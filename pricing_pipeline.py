"""
pricing_pipeline.py
===================
Полный пайплайн динамического ценообразования.

Вход: item_code (int), date (str, формат YYYY-MM-DD)
Выход: рекомендованные цены для товара и заменителя + таблицы по каждому шагу.

Шаги:
  1. Найти лучшего заменителя из similar_products.csv
  2. Проверить KVI-статус (якорный товар?)
  3. Модель цены от времени (обучение до date, предсказание + CI)
  4. Модель спроса от цены (обучение до date)
  5. Кросс-эффект между товарами (γ-коэффициент)
  6. Оптимизация: максимизация полезности (profit + penalty) по сетке цен
"""


#python pricing_pipeline.py 5 2024-06-01 --grid 20
from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI — рисуем в файл
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Пути
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT  = ROOT / "output" / "pipeline"
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Параметры пайплайна
# ─────────────────────────────────────────────────────────────────────────────

PRICE_MODEL_POLY_DEGREE  = 2
DEMAND_MODEL_POLY_DEGREE = 2
CI_ALPHA = 0.05                # 95% prediction interval
MAX_PRICE_MONTHLY_DRIFT = 0.05 # макс. 5% в месяц от последней известной цены при экстраполяции

DEMAND_FULL_FEATURES  = ["UNITPRICE", "month", "dayofweek", "day", "year", "quarter"]
DEMAND_PRICE_FEATURES = ["UNITPRICE"]   # fallback: только цена, без временного тренда
N_GRID_POINTS = 15             # точек сетки для каждого товара
KVI_BAND = 0.08                # ±8% вокруг предсказанной цены при KVI-ограничении
MIN_DEMAND_FLOOR = 0.0         # нижняя граница спроса (не уходить в минус)

# Веса функции полезности
W_PROFIT     = 1.0             # вес суммарной прибыли
W_REVENUE    = 0.1             # бонус за выручку (сглаживает)
W_STABILITY  = 0.05            # штраф за слишком высокую цену относительно CI-центра


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _date_features(dates: pd.Series) -> pd.DataFrame:
    """Разобрать Series дат в признаки для моделей."""
    dt = pd.to_datetime(dates)
    return pd.DataFrame({
        "month":      dt.dt.month,
        "year":       dt.dt.year,
        "day":        dt.dt.day,
        "dayofweek":  dt.dt.dayofweek,
        "quarter":    dt.dt.quarter,
    })


def _build_price_time_model(df_item: pd.DataFrame):
    """
    Обучить модель цены от времени (Ridge + Poly(2)).

    Returns dict с:
      model, scaler, poly,
      predicted_price, lower_95, upper_95 (для каждой строки обучения),
      mse, t_value, n, p
    """
    feature_cols = ["month", "year", "day", "dayofweek", "quarter"]
    X = df_item[feature_cols].values
    y = df_item["UNITPRICE"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=PRICE_MODEL_POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    model = Ridge(alpha=1.0)
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    n, p = X_poly.shape
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / max(n - p, 1)
    s_err = np.sqrt(mse)

    XtX_inv = np.linalg.pinv(X_poly.T @ X_poly)
    h = np.sum(X_poly @ XtX_inv * X_poly, axis=1)
    t_val = t.ppf(1 - CI_ALPHA / 2, df=max(n - p, 1))
    pi = t_val * s_err * np.sqrt(1 + h)

    return {
        "model": model,
        "scaler": scaler,
        "poly": poly,
        "y_pred": y_pred,
        "lower_95": y_pred - pi,
        "upper_95": y_pred + pi,
        "mse": mse,
        "t_val": t_val,
        "s_err": s_err,
        "XtX_inv": XtX_inv,
        "n": n,
        "p": p,
    }


def _predict_price_with_ci(
    fit: dict,
    date_row: pd.DataFrame,
) -> dict:
    """
    Предсказать цену для одного дня + CI.

    date_row — DataFrame с одной строкой (признаки: month, year, day, dayofweek, quarter).
    """
    feature_cols = ["month", "year", "day", "dayofweek", "quarter"]
    X = date_row[feature_cols].values
    X_scaled = fit["scaler"].transform(X)
    X_poly = fit["poly"].transform(X_scaled)

    y_pred = fit["model"].predict(X_poly)[0]

    # Prediction interval для новой точки (X_poly имеет форму (1, p))
    h_new = float((X_poly @ fit["XtX_inv"] @ X_poly.T).ravel()[0])
    pi = fit["t_val"] * fit["s_err"] * np.sqrt(1 + h_new)

    return {
        "predicted_price": y_pred,
        "lower_95": y_pred - pi,
        "upper_95": y_pred + pi,
    }


def _fit_demand_ridge(df_item: pd.DataFrame, feature_cols: list, alpha: float = 1.0) -> dict:
    """Внутренняя: обучить Ridge-модель спроса на заданном наборе признаков."""
    X = df_item[feature_cols].values
    y = df_item["AMOUNT"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=DEMAND_MODEL_POLY_DEGREE, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    n, p = X_poly.shape
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / max(n - p, 1)
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)
    s_err = np.sqrt(mse)

    XtX_inv = np.linalg.pinv(X_poly.T @ X_poly)
    h = np.sum(X_poly @ XtX_inv * X_poly, axis=1)
    t_val = t.ppf(1 - CI_ALPHA / 2, df=max(n - p, 1))
    pi = t_val * s_err * np.sqrt(1 + h)

    return {
        "model": model,
        "scaler": scaler,
        "poly": poly,
        "feature_cols": feature_cols,
        "r2": r2,
        "mse": mse,
        "s_err": s_err,
        "t_val": t_val,
        "XtX_inv": XtX_inv,
        "n": n,
        "p": p,
        "y_pred": y_pred,
        "lower_95": y_pred - pi,
        "upper_95": y_pred + pi,
    }


def _has_positive_elasticity(fit: dict, df_item: pd.DataFrame) -> bool:
    """
    Вернуть True, если модель предсказывает рост спроса при росте цены
    (аномальная положительная эластичность).
    Проверяем в точке средней цены: Q(avg+σ) > Q(avg-σ).
    """
    avg_p = float(df_item["UNITPRICE"].mean())
    std_p = float(max(df_item["UNITPRICE"].std(), avg_p * 0.05))
    # Средние значения временных признаков (используются для полной модели)
    avg_date = {
        "month":     int(round(df_item["month"].mean())),
        "dayofweek": int(round(df_item["dayofweek"].mean())),
        "day":       int(round(df_item["day"].mean())),
        "year":      int(round(df_item["year"].mean())),
        "quarter":   int(round(df_item["quarter"].mean())),
    }
    d_lo = _predict_demand(fit, avg_p - std_p, avg_date)
    d_hi = _predict_demand(fit, avg_p + std_p, avg_date)
    return d_hi > d_lo


def _build_demand_model(df_item: pd.DataFrame) -> dict:
    """
    Обучить модель спроса от цены (Ridge + Poly(2)).

    Автоматически обнаруживает аномальную положительную эластичность
    (характерна при мультиколлинеарности цены и времени) и применяет
    резервную модель:
      1. Полная модель (UNITPRICE + временны́е признаки)
      2. Откат: модель только по цене (убираем временной тренд)
      3. Если всё ещё аномалия: почти-константная модель (большой alpha)

    Поле anomalous_elasticity: False | "price_only" | "constant"
    """
    # ── Попытка 1: полная модель ──────────────────────────────────────────────
    fit = _fit_demand_ridge(df_item, DEMAND_FULL_FEATURES, alpha=1.0)
    fit["anomalous_elasticity"] = False

    if not _has_positive_elasticity(fit, df_item):
        return fit

    # ── Попытка 2: только цена (без временного тренда) ───────────────────────
    fit2 = _fit_demand_ridge(df_item, DEMAND_PRICE_FEATURES, alpha=1.0)
    fit2["anomalous_elasticity"] = "price_only"

    if not _has_positive_elasticity(fit2, df_item):
        return fit2

    # ── Попытка 3: почти-константная (сильная регуляризация) ─────────────────
    fit3 = _fit_demand_ridge(df_item, DEMAND_PRICE_FEATURES, alpha=1e6)
    fit3["anomalous_elasticity"] = "constant"
    return fit3


def _predict_demand(fit: dict, price: float, date_features: dict) -> float:
    """Предсказать спрос для конкретной цены и даты."""
    row = pd.DataFrame([{"UNITPRICE": price, **date_features}])
    feature_cols = fit.get("feature_cols", DEMAND_FULL_FEATURES)
    X = row[feature_cols].values
    X_scaled = fit["scaler"].transform(X)
    X_poly = fit["poly"].transform(X_scaled)
    demand = float(fit["model"].predict(X_poly)[0])
    return max(demand, MIN_DEMAND_FLOOR)


def _predict_demand_with_ci(fit: dict, price: float, date_features: dict) -> dict:
    """
    Предсказать спрос с 95% prediction interval для конкретной цены и даты.
    Returns: dict с ключами demand, lower_95, upper_95.
    """
    row = pd.DataFrame([{"UNITPRICE": price, **date_features}])
    feature_cols = fit.get("feature_cols", DEMAND_FULL_FEATURES)
    X = row[feature_cols].values
    X_scaled = fit["scaler"].transform(X)
    X_poly = fit["poly"].transform(X_scaled)

    demand = float(fit["model"].predict(X_poly)[0])

    h_new = float((X_poly @ fit["XtX_inv"] @ X_poly.T).ravel()[0])
    pi = fit["t_val"] * fit["s_err"] * np.sqrt(1 + h_new)

    return {
        "demand":    max(demand, MIN_DEMAND_FLOOR),
        "lower_95":  max(demand - pi, MIN_DEMAND_FLOOR),
        "upper_95":  demand + pi,
    }


def _compute_gamma(sim_score: float, high_thresh: float = 0.85, low_thresh: float = 0.70, gamma_max: float = 1.5) -> float:
    """Вычислить γ-коэффициент кросс-эффекта из similarity score."""
    if sim_score >= high_thresh:
        return gamma_max
    elif sim_score >= low_thresh:
        return gamma_max * (sim_score - low_thresh) / (high_thresh - low_thresh)
    return 0.0


def _cross_demand(base_demand: float, gamma: float,
                  price_other: float, ref_price_other: float,
                  price_self: float,  ref_price_self: float) -> float:
    """
    Относительный кросс-эффект между заменителями.

    Если оба товара подорожали на одинаковый % от базовой цены — эффект равен нулю.
    Если j стал относительно дороже i → часть покупателей j переходит к i → q_i растёт.
    Формула: delta = gamma * (rel_j - rel_i) * base_demand_i
      rel = (price - ref_price) / ref_price
    """
    rel_other = (price_other - ref_price_other) / max(ref_price_other, 1.0)
    rel_self  = (price_self  - ref_price_self)  / max(ref_price_self,  1.0)
    delta = gamma * (rel_other - rel_self) * base_demand
    return max(base_demand + delta, MIN_DEMAND_FLOOR)


def _utility(
    price_i: float,
    price_j: float,
    cost_i: float,
    cost_j: float,
    demand_i: float,
    demand_j: float,
    predicted_price_i: float,
    predicted_price_j: float,
) -> float:
    """
    Функция полезности для пары цен.

    U = W_PROFIT  * суммарная_прибыль
      + W_REVENUE * суммарная_выручка
      - W_STABILITY * отклонение_от_предсказанной_цены^2
    """
    profit_i = (price_i - cost_i) * demand_i
    profit_j = (price_j - cost_j) * demand_j
    revenue_i = price_i * demand_i
    revenue_j = price_j * demand_j

    stability_penalty = (
        ((price_i - predicted_price_i) / max(predicted_price_i, 1e-6)) ** 2 +
        ((price_j - predicted_price_j) / max(predicted_price_j, 1e-6)) ** 2
    )

    return (
        W_PROFIT  * (profit_i + profit_j)
        + W_REVENUE * (revenue_i + revenue_j)
        - W_STABILITY * stability_penalty
    )


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции построения и сохранения графиков
# ─────────────────────────────────────────────────────────────────────────────

def _save_price_time_plot(table: pd.DataFrame, item_code: int, cutoff: pd.Timestamp,
                          predicted: float, lower_95: float, upper_95: float,
                          out_path: Path) -> None:
    """Сохранить график цены от времени (шаг 3) с 95% CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(table["DATE_"], table["UNITPRICE"], s=15, alpha=0.5, color="steelblue", label="Факт")
    ax.plot(table["DATE_"], table["predicted_price"], color="darkorange", linewidth=1.5, label="Модель")
    ax.fill_between(table["DATE_"], table["lower_95"], table["upper_95"],
                    color="darkorange", alpha=0.15, label="95% CI")

    ax.axvline(cutoff, color="red", linestyle="--", linewidth=1.2, label=f"Целевая дата: {cutoff.date()}")
    ax.errorbar([cutoff], [predicted], yerr=[[predicted - lower_95], [upper_95 - predicted]],
                fmt="o", color="red", capsize=5, label=f"Прогноз: {predicted:.2f} [{lower_95:.2f}–{upper_95:.2f}]")

    ax.set_title(f"Шаг 3: Модель цены от времени — товар {item_code}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена (UNITPRICE)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_demand_plot(table: pd.DataFrame, item_code: int, out_path: Path) -> None:
    """Сохранить график спроса от цены (шаг 4) с 95% PI."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(table["UNITPRICE"], table["AMOUNT"], s=15, alpha=0.5,
               color="steelblue", label="Факт")

    df_sorted = table.sort_values("UNITPRICE")
    ax.plot(df_sorted["UNITPRICE"], df_sorted["pred_demand"],
            color="darkorange", linewidth=1.5, label="Модель")
    ax.fill_between(df_sorted["UNITPRICE"], df_sorted["lower_95"], df_sorted["upper_95"],
                    color="darkorange", alpha=0.15, label="95% PI")

    ax.set_title(f"Шаг 4: Модель спроса от цены — товар {item_code}")
    ax.set_xlabel("Цена (UNITPRICE)")
    ax.set_ylabel("Спрос (AMOUNT)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс пайплайна
# ─────────────────────────────────────────────────────────────────────────────

class DynamicPricingPipeline:
    """
    Пайплайн динамического ценообразования для пары товаров (товар + заменитель).

    Использование:
        pipeline = DynamicPricingPipeline()
        result = pipeline.run(item_code=5, date="2024-06-01")
    """

    def __init__(self, n_grid: int = N_GRID_POINTS):
        self.n_grid = n_grid
        self._cache: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка данных (один раз, кэшируем)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_transactions(self) -> pd.DataFrame:
        if "transactions" not in self._cache:
            od = pd.read_csv(DATA / "Order_Details.csv")
            o  = pd.read_csv(DATA / "Orders.csv")
            merged = od.merge(o[["ORDERID", "DATE_"]], on="ORDERID", how="left")
            merged["DATE_"] = pd.to_datetime(merged["DATE_"])
            df_feat = _date_features(merged["DATE_"])
            merged = pd.concat([merged.reset_index(drop=True), df_feat], axis=1)
            self._cache["transactions"] = merged
        return self._cache["transactions"]

    def _historical_profit(self, item_code: int, cutoff: pd.Timestamp) -> dict:
        """
        Считает историческую прибыль за ближайшую к cutoff дату,
        на которой были реальные продажи.

        Берём агрегат за тот же день (DATE_ == cutoff).
        Если в этот день продаж нет — ищем последний день с продажами до cutoff
        (не более 7 дней назад).

        Returns: dict с ключами hist_date, hist_price, hist_quantity, hist_profit.
        """
        df = self._load_transactions()
        item_df = df[df["ITEMCODE"] == item_code].copy()

        # Сначала пробуем точную дату
        day_df = item_df[item_df["DATE_"].dt.normalize() == cutoff.normalize()]

        if day_df.empty:
            # Ищем последний день с продажами в окне [cutoff-7d, cutoff)
            window = item_df[
                (item_df["DATE_"].dt.normalize() < cutoff.normalize()) &
                (item_df["DATE_"].dt.normalize() >= cutoff.normalize() - pd.Timedelta(days=7))
            ]
            if window.empty:
                return {
                    "hist_date":     None,
                    "hist_price":    None,
                    "hist_quantity": None,
                    "hist_cost":     None,
                    "hist_profit":   None,
                }
            last_date = window["DATE_"].dt.normalize().max()
            day_df = window[window["DATE_"].dt.normalize() == last_date]

        hist_date     = day_df["DATE_"].dt.normalize().iloc[0]
        hist_price    = float(day_df["UNITPRICE"].mean())
        hist_quantity = float(day_df["AMOUNT"].sum())
        hist_cost     = self._get_cost(item_code, hist_date)
        if hist_cost is None:
            hist_cost = hist_price * 0.5

        hist_profit = (hist_price - hist_cost) * hist_quantity

        return {
            "hist_date":     hist_date.strftime("%Y-%m-%d"),
            "hist_price":    round(hist_price, 4),
            "hist_quantity": round(hist_quantity, 4),
            "hist_cost":     round(hist_cost, 4),
            "hist_profit":   round(hist_profit, 4),
        }

    def _load_similarities(self) -> pd.DataFrame:
        if "similarities" not in self._cache:
            self._cache["similarities"] = pd.read_csv(
                ROOT / "similar_products.csv",
                dtype={"item_code_i": int, "item_code_j": int},
            )
        return self._cache["similarities"]

    def _load_kvi(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает (kvi_full, kvi_candidates)."""
        if "kvi" not in self._cache:
            full = pd.read_csv(ROOT / "output" / "kvi" / "kvi_scores_full.csv",
                               dtype={"ITEMID": str})
            cand = pd.read_csv(ROOT / "output" / "kvi" / "kvi_candidates.csv",
                               dtype={"ITEMID": str})
            cat  = pd.read_csv(DATA / "Categories_ENG.csv", sep=";",
                               dtype={"ITEMID": str, "ITEMCODE": int})
            # Добавляем ITEMCODE в kvi_full
            full = full.merge(cat[["ITEMID", "ITEMCODE"]], on="ITEMID", how="left")
            self._cache["kvi"] = (full, cand, cat)
        return self._cache["kvi"]

    def _load_costs(self) -> pd.DataFrame:
        if "costs" not in self._cache:
            df = pd.read_csv(DATA / "cost.csv", dtype={"ITEMCODE": int})
            df["DATE_"] = pd.to_datetime(df["DATE_"])
            df = df.sort_values(["ITEMCODE", "DATE_"]).reset_index(drop=True)
            self._cache["costs"] = df
        return self._cache["costs"]

    def _get_cost(self, item_code: int, cutoff: pd.Timestamp) -> float:
        """
        Вернуть себестоимость товара на дату cutoff (без утечки):
        берём последнюю запись с DATE_ <= cutoff.
        Если записей нет — возвращаем None.
        """
        costs = self._load_costs()
        rows = costs[(costs["ITEMCODE"] == item_code) & (costs["DATE_"] <= cutoff)]
        if rows.empty:
            return None
        return float(rows.iloc[-1]["cost"])

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 1: найти заменителя
    # ──────────────────────────────────────────────────────────────────────────

    def _step1_find_substitute(self, item_code: int) -> dict:
        sim_df = self._load_similarities()
        candidates = (
            sim_df[sim_df["item_code_i"] == item_code]
            .sort_values("score", ascending=False)
            .copy()
        )
        # Исключаем пары с самим собой (score ≈ 1.0) и score > 1.0 (числовые артефакты)
        candidates = candidates[candidates["item_code_j"] != item_code]
        candidates = candidates[candidates["score"] <= 1.0]

        if candidates.empty:
            return {"found": False, "substitute_code": None, "similarity_score": None,
                    "table": pd.DataFrame()}

        best = candidates.iloc[0]
        table = candidates.head(10).copy()
        table.columns = ["item_code", "substitute_code", "similarity_score"]

        return {
            "found": True,
            "substitute_code": int(best["item_code_j"]),
            "similarity_score": float(best["score"]),
            "table": table,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 2: KVI-статус
    # ──────────────────────────────────────────────────────────────────────────

    def _step2_kvi_status(self, item_code: int) -> dict:
        kvi_full, kvi_cand, cat = self._load_kvi()

        # ITEMCODE → ITEMID
        row = cat[cat["ITEMCODE"] == item_code]
        if row.empty:
            return {"is_kvi": False, "kvi_score": None, "itemid": None,
                    "item_name": "unknown", "table": pd.DataFrame()}

        itemid = str(row.iloc[0]["ITEMID"])
        item_name = str(row.iloc[0]["ITEMNAME"])

        kvi_row = kvi_full[kvi_full["ITEMID"] == itemid]
        kvi_score = float(kvi_row["kvi_score_final"].iloc[0]) if not kvi_row.empty else None

        is_kvi = itemid in kvi_cand["ITEMID"].values

        table = pd.DataFrame([{
            "item_code":  item_code,
            "ITEMID":     itemid,
            "ITEMNAME":   item_name,
            "is_kvi":     is_kvi,
            "kvi_score":  kvi_score,
            "elasticity": float(kvi_row["elasticity"].iloc[0]) if not kvi_row.empty and "elasticity" in kvi_row.columns else None,
        }])

        return {
            "is_kvi":   is_kvi,
            "kvi_score": kvi_score,
            "itemid":   itemid,
            "item_name": item_name,
            "table":    table,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 3: модель цены от времени
    # ──────────────────────────────────────────────────────────────────────────

    def _step3_price_time_model(self, item_code: int, cutoff: pd.Timestamp) -> dict:
        df = self._load_transactions()
        df_item = (
            df[(df["ITEMCODE"] == item_code) & (df["DATE_"] < cutoff)]
            .groupby("DATE_")
            .agg(
                UNITPRICE=("UNITPRICE", "mean"),
                month=("month", "first"),
                year=("year", "first"),
                day=("day", "first"),
                dayofweek=("dayofweek", "first"),
                quarter=("quarter", "first"),
            )
            .reset_index()
            .sort_values("DATE_")
        )

        if len(df_item) < 5:
            return {"ok": False, "reason": f"Слишком мало данных ({len(df_item)} строк)"}

        fit = _build_price_time_model(df_item)

        # Предсказание для целевой даты
        target_feats = pd.DataFrame([{
            "month":     cutoff.month,
            "year":      cutoff.year,
            "day":       cutoff.day,
            "dayofweek": cutoff.dayofweek,
            "quarter":   cutoff.quarter,
        }])
        pred = _predict_price_with_ci(fit, target_feats)

        # Ограничить экстраполяцию: не уходить дальше MAX_PRICE_MONTHLY_DRIFT в месяц
        # от последней наблюдаемой цены (полином агрессивно экстраполирует при большом gap)
        last_date  = df_item["DATE_"].iloc[-1]
        last_price = float(df_item["UNITPRICE"].iloc[-1])
        days_ahead = max((cutoff - last_date).days, 0)
        max_drift  = last_price * MAX_PRICE_MONTHLY_DRIFT * (days_ahead / 30.0)
        pred_raw   = pred["predicted_price"]
        pred_clamped = float(np.clip(pred_raw, last_price - max_drift, last_price + max_drift))
        if abs(pred_clamped - pred_raw) > 0.01:
            ci_half = (pred["upper_95"] - pred["lower_95"]) / 2.0
            pred = {
                "predicted_price": pred_clamped,
                "lower_95":        pred_clamped - ci_half,
                "upper_95":        pred_clamped + ci_half,
            }
            print(f"   [CLIP] Экстраполяция обрезана: полином={pred_raw:.2f} → {pred_clamped:.2f} "
                  f"(посл.цена={last_price:.2f}, дней={days_ahead})")
        df_item = df_item.copy()
        df_item["predicted_price"] = fit["y_pred"]
        df_item["lower_95"] = fit["lower_95"]
        df_item["upper_95"] = fit["upper_95"]

        return {
            "ok": True,
            "fit": fit,
            "predicted_price": pred["predicted_price"],
            "lower_95": pred["lower_95"],
            "upper_95": pred["upper_95"],
            "r2": fit["model"].score(
                fit["poly"].transform(fit["scaler"].transform(
                    df_item[["month", "year", "day", "dayofweek", "quarter"]].values
                )),
                df_item["UNITPRICE"].values,
            ),
            "n_train": len(df_item),
            "table": df_item[["DATE_", "UNITPRICE", "predicted_price", "lower_95", "upper_95"]].copy(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 4: модель спроса от цены
    # ──────────────────────────────────────────────────────────────────────────

    def _step4_demand_model(self, item_code: int, cutoff: pd.Timestamp) -> dict:
        df = self._load_transactions()
        df_item = (
            df[(df["ITEMCODE"] == item_code) & (df["DATE_"] < cutoff)]
            .groupby("DATE_")
            .agg(
                AMOUNT=("AMOUNT", "sum"),
                UNITPRICE=("UNITPRICE", "mean"),
                month=("month", "first"),
                year=("year", "first"),
                day=("day", "first"),
                dayofweek=("dayofweek", "first"),
                quarter=("quarter", "first"),
            )
            .reset_index()
            .sort_values("DATE_")
        )

        if len(df_item) < 5:
            return {"ok": False, "reason": f"Слишком мало данных ({len(df_item)} строк)"}

        fit = _build_demand_model(df_item)

        df_item = df_item.copy()
        df_item["pred_demand"] = fit["y_pred"]
        df_item["lower_95"] = fit["lower_95"]
        df_item["upper_95"] = fit["upper_95"]

        anomaly = fit["anomalous_elasticity"]
        if anomaly == "price_only":
            print(f"   [ANOMALY] Товар {item_code}: положительная эластичность — "
                  f"модель переобучена на price-only (без временнóго тренда)")
        elif anomaly == "constant":
            print(f"   [ANOMALY] Товар {item_code}: положительная эластичность даже "
                  f"в price-only модели — используется константный спрос")

        # Средний спрос и цена (для кросс-эффекта используем как ref)
        avg_demand = float(df_item["AMOUNT"].mean())
        avg_price  = float(df_item["UNITPRICE"].mean())

        return {
            "ok": True,
            "fit": fit,
            "r2": fit["r2"],
            "n_train": len(df_item),
            "avg_demand": avg_demand,
            "avg_price": avg_price,
            "anomalous_elasticity": anomaly,
            "table": df_item[["DATE_", "UNITPRICE", "AMOUNT", "pred_demand", "lower_95", "upper_95"]].copy(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 5: кросс-эффект
    # ──────────────────────────────────────────────────────────────────────────

    def _step5_cross_effect(
        self,
        item_code_i: int,
        item_code_j: int,
        sim_score: float,
    ) -> dict:
        gamma_ij = _compute_gamma(sim_score)
        gamma_ji = _compute_gamma(sim_score)  # симметрично (оба заменители)

        table = pd.DataFrame([{
            "item_code_i": item_code_i,
            "item_code_j": item_code_j,
            "similarity_score": sim_score,
            "pair_type": "strong_substitute" if sim_score >= 0.85 else "weak_substitute",
            "gamma_ij": gamma_ij,
            "gamma_ji": gamma_ji,
        }])

        return {
            "gamma_ij": gamma_ij,  # влияние j на i
            "gamma_ji": gamma_ji,  # влияние i на j
            "table": table,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Шаг 6: оптимизация цен
    # ──────────────────────────────────────────────────────────────────────────

    def _step6_optimize(
        self,
        *,
        demand_fit_i: dict,
        demand_fit_j: dict,
        price_bounds_i: tuple,
        price_bounds_j: tuple,
        predicted_price_i: float,
        predicted_price_j: float,
        ref_price_i: float,
        ref_price_j: float,
        cost_i: float,
        cost_j: float,
        gamma_ij: float,
        gamma_ji: float,
        date_feats: dict,
        is_kvi_i: bool,
        is_kvi_j: bool,
    ) -> dict:

        lo_i, hi_i = price_bounds_i
        lo_j, hi_j = price_bounds_j

        # ── KVI: сужаем диапазон ──────────────────────────────────────────────
        if is_kvi_i:
            lo_i = max(lo_i, predicted_price_i * (1 - KVI_BAND))
            hi_i = min(hi_i, predicted_price_i * (1 + KVI_BAND))

        if is_kvi_j:
            lo_j = max(lo_j, predicted_price_j * (1 - KVI_BAND))
            hi_j = min(hi_j, predicted_price_j * (1 + KVI_BAND))

        # Защита: если bounds некорректны (например, lo > hi после сужения)
        lo_i, hi_i = min(lo_i, hi_i), max(lo_i, hi_i)
        lo_j, hi_j = min(lo_j, hi_j), max(lo_j, hi_j)

        # ── Создание сетки цен ────────────────────────────────────────────────
        grid_i = np.linspace(lo_i, hi_i, self.n_grid)
        grid_j = np.linspace(lo_j, hi_j, self.n_grid)

        rows = []

        for pi in grid_i:
            # Базовый спрос i при данной цене pi
            q_i_base = _predict_demand(demand_fit_i, pi, date_feats)
            for pj in grid_j:
                # Базовый спрос j при данной цене pj
                q_j_base = _predict_demand(demand_fit_j, pj, date_feats)

                # Скорректированный спрос с кросс-эффектом
                # Покупатели перетекают к i, если j ОТНОСИТЕЛЬНО дороже i
                q_i = _cross_demand(q_i_base, gamma_ij,  pj, ref_price_j,  pi, ref_price_i)
                q_j = _cross_demand(q_j_base, gamma_ji,  pi, ref_price_i,  pj, ref_price_j)

                u = _utility(
                    price_i=pi,
                    price_j=pj,
                    cost_i=cost_i,
                    cost_j=cost_j,
                    demand_i=q_i,
                    demand_j=q_j,
                    predicted_price_i=predicted_price_i,
                    predicted_price_j=predicted_price_j,
                )

                rows.append({
                    "price_i":         round(pi, 4),
                    "price_j":         round(pj, 4),
                    "demand_i":        round(q_i, 4),
                    "demand_j":        round(q_j, 4),
                    "profit_i":        round((pi - cost_i) * q_i, 4),
                    "profit_j":        round((pj - cost_j) * q_j, 4),
                    "total_profit":    round((pi - cost_i) * q_i + (pj - cost_j) * q_j, 4),
                    "revenue_i":       round(pi * q_i, 4),
                    "revenue_j":       round(pj * q_j, 4),
                    "total_revenue":   round(pi * q_i + pj * q_j, 4),
                    "utility":         round(u, 6),
                })

        grid_df = pd.DataFrame(rows)
        best_idx = grid_df["utility"].idxmax()
        best_row = grid_df.loc[best_idx]

        # ── Точный оптимум через scipy (на основе лучшей точки сетки) ─────────
        def neg_utility(prices):
            pi_, pj_ = prices
            q_i_b = _predict_demand(demand_fit_i, pi_, date_feats)
            q_j_b = _predict_demand(demand_fit_j, pj_, date_feats)
            q_i_ = _cross_demand(q_i_b, gamma_ij, pj_, ref_price_j, pi_, ref_price_i)
            q_j_ = _cross_demand(q_j_b, gamma_ji, pi_, ref_price_i, pj_, ref_price_j)
            return -_utility(pi_, pj_, cost_i, cost_j, q_i_, q_j_,
                              predicted_price_i, predicted_price_j)

        x0 = [best_row["price_i"], best_row["price_j"]]
        bounds = [(lo_i, hi_i), (lo_j, hi_j)]
        opt = minimize(neg_utility, x0=x0, bounds=bounds, method="L-BFGS-B")

        opt_pi, opt_pj = opt.x
        opt_q_i = _cross_demand(
            _predict_demand(demand_fit_i, opt_pi, date_feats),
            gamma_ij, opt_pj, ref_price_j, opt_pi, ref_price_i
        )
        opt_q_j = _cross_demand(
            _predict_demand(demand_fit_j, opt_pj, date_feats),
            gamma_ji, opt_pi, ref_price_i, opt_pj, ref_price_j
        )

        # CI для спроса при оптимальных ценах
        q_i_ci = _predict_demand_with_ci(demand_fit_i, opt_pi, date_feats)
        q_j_ci = _predict_demand_with_ci(demand_fit_j, opt_pj, date_feats)

        # Применяем кросс-эффект к границам CI
        q_i_lo = _cross_demand(q_i_ci["lower_95"], gamma_ij, opt_pj, ref_price_j, opt_pi, ref_price_i)
        q_i_hi = _cross_demand(q_i_ci["upper_95"], gamma_ij, opt_pj, ref_price_j, opt_pi, ref_price_i)
        q_j_lo = _cross_demand(q_j_ci["lower_95"], gamma_ji, opt_pi, ref_price_i, opt_pj, ref_price_j)
        q_j_hi = _cross_demand(q_j_ci["upper_95"], gamma_ji, opt_pi, ref_price_i, opt_pj, ref_price_j)

        optimal = {
            "price_i":           round(opt_pi, 4),
            "price_j":           round(opt_pj, 4),
            "demand_i":          round(opt_q_i, 4),
            "demand_i_lower_95": round(q_i_lo, 4),
            "demand_i_upper_95": round(q_i_hi, 4),
            "demand_j":          round(opt_q_j, 4),
            "demand_j_lower_95": round(q_j_lo, 4),
            "demand_j_upper_95": round(q_j_hi, 4),
            "profit_i":          round((opt_pi - cost_i) * opt_q_i, 4),
            "profit_i_lower_95": round((opt_pi - cost_i) * q_i_lo, 4),
            "profit_i_upper_95": round((opt_pi - cost_i) * q_i_hi, 4),
            "profit_j":          round((opt_pj - cost_j) * opt_q_j, 4),
            "profit_j_lower_95": round((opt_pj - cost_j) * q_j_lo, 4),
            "profit_j_upper_95": round((opt_pj - cost_j) * q_j_hi, 4),
            "total_profit":      round((opt_pi - cost_i) * opt_q_i + (opt_pj - cost_j) * opt_q_j, 4),
            "total_profit_lower_95": round((opt_pi - cost_i) * q_i_lo + (opt_pj - cost_j) * q_j_lo, 4),
            "total_profit_upper_95": round((opt_pi - cost_i) * q_i_hi + (opt_pj - cost_j) * q_j_hi, 4),
            "utility":           round(-opt.fun, 6),
        }

        return {
            "grid_df":           grid_df,
            "optimal":           optimal,
            "price_bounds_i":    (round(lo_i, 4), round(hi_i, 4)),
            "price_bounds_j":    (round(lo_j, 4), round(hi_j, 4)),
            "is_kvi_constrained_i": is_kvi_i,
            "is_kvi_constrained_j": is_kvi_j,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Главный метод: run()
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, item_code: int, date: str) -> dict:
        """
        Запустить полный пайплайн.

        Parameters
        ----------
        item_code : int   — ITEMCODE целевого товара
        date      : str   — дата в формате YYYY-MM-DD (всё до этой даты — обучение)

        Returns
        -------
        dict с полными результатами каждого шага.
        Tables сохраняются в output/pipeline/<item_code>_<date>/
        """
        cutoff = pd.Timestamp(date)
        run_id = f"{item_code}_{date}"
        out_dir = OUT / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        results: dict = {"item_code": item_code, "date": date, "steps": {}}

        print(f"\n{'='*65}")
        print(f"  DYNAMIC PRICING PIPELINE  |  item={item_code}  date={date}")
        print(f"{'='*65}\n")

        # ── ШАГ 1 ─────────────────────────────────────────────────────────────
        print(">> ШАГ 1: Поиск заменителя...")
        s1 = self._step1_find_substitute(item_code)
        results["steps"]["step1_substitute"] = {k: v for k, v in s1.items() if k != "table"}

        if not s1["found"]:
            print("   [WARN] Заменитель не найден. Пайплайн будет работать только для основного товара.")
            sub_code = None
            sim_score = 0.0
        else:
            sub_code  = s1["substitute_code"]
            sim_score = s1["similarity_score"]
            print(f"   Заменитель: {sub_code}  (similarity={sim_score:.4f})")

        s1["table"].to_csv(out_dir / "step1_substitutes.csv", index=False)
        print(f"   Сохранено: step1_substitutes.csv")

        # ── ШАГ 2 ─────────────────────────────────────────────────────────────
        print("\n>> ШАГ 2: Проверка KVI-статуса...")
        s2_i = self._step2_kvi_status(item_code)
        results["steps"]["step2_kvi_item"] = {k: v for k, v in s2_i.items() if k != "table"}
        s2_i["table"].to_csv(out_dir / "step2_kvi_item.csv", index=False)
        print(f"   Товар {item_code} ({s2_i['item_name']}): is_kvi={s2_i['is_kvi']}  score={s2_i['kvi_score']}")

        s2_j: dict = {"is_kvi": False, "kvi_score": None, "item_name": "N/A"}
        if sub_code is not None:
            s2_j = self._step2_kvi_status(sub_code)
            results["steps"]["step2_kvi_substitute"] = {k: v for k, v in s2_j.items() if k != "table"}
            s2_j["table"].to_csv(out_dir / "step2_kvi_substitute.csv", index=False)
            print(f"   Заменитель {sub_code} ({s2_j['item_name']}): is_kvi={s2_j['is_kvi']}  score={s2_j['kvi_score']}")
        print(f"   Сохранено: step2_kvi_*.csv")

        # ── ШАГ 3 ─────────────────────────────────────────────────────────────
        print("\n>> ШАГ 3: Модель цены от времени...")
        s3_i = self._step3_price_time_model(item_code, cutoff)
        if not s3_i["ok"]:
            print(f"   [ERROR] Товар {item_code}: {s3_i['reason']}")
            return {"error": s3_i["reason"], **results}

        s3_i["table"].to_csv(out_dir / "step3_price_time_item.csv", index=False)
        _save_price_time_plot(s3_i["table"], item_code, cutoff,
                              s3_i["predicted_price"], s3_i["lower_95"], s3_i["upper_95"],
                              out_dir / "step3_price_time_item.png")
        print(f"   Товар {item_code}: предсказанная цена={s3_i['predicted_price']:.4f}"
              f"  CI=[{s3_i['lower_95']:.4f}, {s3_i['upper_95']:.4f}]  R²={s3_i['r2']:.4f}")

        s3_j: dict = {"ok": False, "predicted_price": None, "lower_95": None, "upper_95": None}
        if sub_code is not None:
            s3_j = self._step3_price_time_model(sub_code, cutoff)
            if s3_j["ok"]:
                s3_j["table"].to_csv(out_dir / "step3_price_time_substitute.csv", index=False)
                _save_price_time_plot(s3_j["table"], sub_code, cutoff,
                                      s3_j["predicted_price"], s3_j["lower_95"], s3_j["upper_95"],
                                      out_dir / "step3_price_time_substitute.png")
                print(f"   Заменитель {sub_code}: предсказанная цена={s3_j['predicted_price']:.4f}"
                      f"  CI=[{s3_j['lower_95']:.4f}, {s3_j['upper_95']:.4f}]  R²={s3_j['r2']:.4f}")
            else:
                print(f"   [WARN] Заменитель {sub_code}: {s3_j['reason']}")

        results["steps"]["step3_price_time"] = {
            "item":       {k: v for k, v in s3_i.items() if k not in ("fit", "table")},
            "substitute": {k: v for k, v in s3_j.items() if k not in ("fit", "table")},
        }
        print(f"   Сохранено: step3_price_time_*.csv")

        # ── ШАГ 4 ─────────────────────────────────────────────────────────────
        print("\n>> ШАГ 4: Модель спроса от цены...")
        s4_i = self._step4_demand_model(item_code, cutoff)
        if not s4_i["ok"]:
            print(f"   [ERROR] Товар {item_code}: {s4_i['reason']}")
            return {"error": s4_i["reason"], **results}

        s4_i["table"].to_csv(out_dir / "step4_demand_item.csv", index=False)
        _save_demand_plot(s4_i["table"], item_code, out_dir / "step4_demand_item.png")
        anom_i = s4_i.get('anomalous_elasticity') or 'нет'
        print(f"   Товар {item_code}: R²={s4_i['r2']:.4f}  avg_demand={s4_i['avg_demand']:.2f}"
              f"  avg_price={s4_i['avg_price']:.4f}  n_train={s4_i['n_train']}  аномалия={anom_i}")

        s4_j: dict = {"ok": False, "fit": None, "avg_price": s3_j.get("predicted_price", 0.0) or 0.0}
        if sub_code is not None:
            s4_j = self._step4_demand_model(sub_code, cutoff)
            if s4_j["ok"]:
                s4_j["table"].to_csv(out_dir / "step4_demand_substitute.csv", index=False)
                _save_demand_plot(s4_j["table"], sub_code, out_dir / "step4_demand_substitute.png")
                anom_j = s4_j.get('anomalous_elasticity') or 'нет'
                print(f"   Заменитель {sub_code}: R²={s4_j['r2']:.4f}  avg_demand={s4_j['avg_demand']:.2f}"
                      f"  avg_price={s4_j['avg_price']:.4f}  n_train={s4_j['n_train']}  аномалия={anom_j}")
            else:
                print(f"   [WARN] Заменитель {sub_code}: {s4_j['reason']}")

        results["steps"]["step4_demand"] = {
            "item":       {k: v for k, v in s4_i.items() if k not in ("fit", "table")},
            "substitute": {k: v for k, v in s4_j.items() if k not in ("fit", "table")},
        }
        print(f"   Сохранено: step4_demand_*.csv")

        # ── ШАГ 5 ─────────────────────────────────────────────────────────────
        print("\n>> ШАГ 5: Кросс-эффект...")
        if sub_code is not None:
            s5 = self._step5_cross_effect(item_code, sub_code, sim_score)
            s5["table"].to_csv(out_dir / "step5_cross_effect.csv", index=False)
            print(f"   γ_ij (j→i)={s5['gamma_ij']:.4f}   γ_ji (i→j)={s5['gamma_ji']:.4f}")
            print(f"   Сохранено: step5_cross_effect.csv")
        else:
            s5 = {"gamma_ij": 0.0, "gamma_ji": 0.0}
            print("   Заменитель не найден, кросс-эффекты = 0")

        results["steps"]["step5_cross_effect"] = {k: v for k, v in s5.items() if k != "table"}

        # ── ШАГ 6 ─────────────────────────────────────────────────────────────
        print("\n>> ШАГ 6: Оптимизация цен...")

        # Загрузка себестоимости (date-aware: последнее значение до cutoff)
        cost_i = self._get_cost(item_code, cutoff)
        if cost_i is None:
            cost_i = s3_i["predicted_price"] * 0.5
            print(f"   [WARN] Себестоимость для {item_code} не найдена, использую 50% от цены")
        else:
            print(f"   Себестоимость {item_code} на {date}: {cost_i:.4f}")

        cost_j = 0.0
        if sub_code is not None:
            cost_j = self._get_cost(sub_code, cutoff)
            if cost_j is None:
                cost_j = (s3_j.get("predicted_price") or 0) * 0.5
                print(f"   [WARN] Себестоимость для {sub_code} не найдена, использую 50% от цены")
            else:
                print(f"   Себестоимость {sub_code} на {date}: {cost_j:.4f}")

        # Если нет данных по заменителю — делаем фиктивный fit (нулевой спрос)
        demand_fit_j = s4_j["fit"] if (s4_j.get("ok") and s4_j.get("fit")) else None
        if demand_fit_j is None:
            # Заглушка: нулевой спрос заменителя
            class _ZeroDemand:
                scaler  = StandardScaler().fit([[0]*6])
                poly    = PolynomialFeatures(degree=1, include_bias=False).fit([[0]*6])
                model   = Ridge().fit([[0]*6], [0])
            demand_fit_j = {"model": _ZeroDemand.model,
                            "scaler": _ZeroDemand.scaler,
                            "poly":   _ZeroDemand.poly}

        price_bounds_i = (s3_i["lower_95"], s3_i["upper_95"])
        price_bounds_j_base = (
            (s3_j["lower_95"] or s3_i["lower_95"] * 0.8,
             s3_j["upper_95"] or s3_i["upper_95"] * 1.2)
            if s3_j.get("ok") else (cost_j * 1.1, cost_j * 2.0)
        )

        date_feats = {
            "month":     cutoff.month,
            "year":      cutoff.year,
            "day":       cutoff.day,
            "dayofweek": cutoff.dayofweek,
            "quarter":   cutoff.quarter,
        }

        ref_price_i = s4_i["avg_price"]
        ref_price_j = s4_j["avg_price"] if s4_j.get("ok") else (s3_j.get("predicted_price") or cost_j * 1.5)

        s6 = self._step6_optimize(
            demand_fit_i=s4_i["fit"],
            demand_fit_j=demand_fit_j,
            price_bounds_i=price_bounds_i,
            price_bounds_j=price_bounds_j_base,
            predicted_price_i=s3_i["predicted_price"],
            predicted_price_j=s3_j.get("predicted_price") or ref_price_j,
            ref_price_i=ref_price_i,
            ref_price_j=ref_price_j,
            cost_i=cost_i,
            cost_j=cost_j,
            gamma_ij=s5["gamma_ij"],
            gamma_ji=s5["gamma_ji"],
            date_feats=date_feats,
            is_kvi_i=s2_i["is_kvi"],
            is_kvi_j=s2_j["is_kvi"],
        )

        s6["grid_df"].to_csv(out_dir / "step6_optimization_grid.csv", index=False)
        print(f"   Диапазон i:  [{s6['price_bounds_i'][0]:.4f}, {s6['price_bounds_i'][1]:.4f}]"
              f"  (KVI={'ДА' if s6['is_kvi_constrained_i'] else 'НЕТ'})")
        if sub_code is not None:
            print(f"   Диапазон j:  [{s6['price_bounds_j'][0]:.4f}, {s6['price_bounds_j'][1]:.4f}]"
                  f"  (KVI={'ДА' if s6['is_kvi_constrained_j'] else 'НЕТ'})")
        print(f"   Точек сетки: {len(s6['grid_df'])}")

        opt = s6["optimal"]
        print(f"\n{'─'*65}")
        print(f"  ОПТИМАЛЬНАЯ СТРАТЕГИЯ")
        print(f"{'─'*65}")
        print(f"  Товар    {item_code}: цена={opt['price_i']:.4f}  спрос={opt['demand_i']:.2f}"
              f"  прибыль={opt['profit_i']:.4f}")
        if sub_code is not None:
            print(f"  Замен.   {sub_code}: цена={opt['price_j']:.4f}  спрос={opt['demand_j']:.2f}"
                  f"  прибыль={opt['profit_j']:.4f}")
        print(f"  Суммарная прибыль: {opt['total_profit']:.4f}")
        print(f"  Utility:           {opt['utility']:.6f}")
        print(f"{'─'*65}")

        results["steps"]["step6_optimization"] = {
            "optimal":           opt,
            "price_bounds_i":    s6["price_bounds_i"],
            "price_bounds_j":    s6["price_bounds_j"],
            "is_kvi_constrained_i": s6["is_kvi_constrained_i"],
            "is_kvi_constrained_j": s6["is_kvi_constrained_j"],
            "n_grid_points":     len(s6["grid_df"]),
        }

        print(f"\n   Сохранено: step6_optimization_grid.csv  ({len(s6['grid_df'])} строк)")

        # ── Историческая прибыль ───────────────────────────────────────────────
        print("\n>> Историческая прибыль (Order Details)...")
        hist_i = self._historical_profit(item_code, cutoff)
        hist_j: dict = {
            "hist_date": None, "hist_price": None,
            "hist_quantity": None, "hist_cost": None, "hist_profit": None,
        }
        if sub_code is not None:
            hist_j = self._historical_profit(sub_code, cutoff)

        if hist_i["hist_profit"] is not None:
            lift_i = opt["profit_i"] - hist_i["hist_profit"]
            lift_i_pct = lift_i / abs(hist_i["hist_profit"]) * 100 if hist_i["hist_profit"] != 0 else None
            print(f"   Товар {item_code}: ист.цена={hist_i['hist_price']}  "
                  f"ист.кол-во={hist_i['hist_quantity']}  "
                  f"ист.прибыль={hist_i['hist_profit']}  "
                  f"→ прирост={lift_i:+.4f} ({lift_i_pct:+.1f}%)" if lift_i_pct is not None
                  else f"   Товар {item_code}: ист.прибыль={hist_i['hist_profit']}  "
                       f"→ прирост={lift_i:+.4f}")
        else:
            lift_i = None
            lift_i_pct = None
            print(f"   [WARN] Нет исторических продаж для {item_code} вблизи {date}")

        if sub_code is not None and hist_j["hist_profit"] is not None:
            lift_j = opt["profit_j"] - hist_j["hist_profit"]
            lift_j_pct = lift_j / abs(hist_j["hist_profit"]) * 100 if hist_j["hist_profit"] != 0 else None
            print(f"   Замен. {sub_code}: ист.цена={hist_j['hist_price']}  "
                  f"ист.кол-во={hist_j['hist_quantity']}  "
                  f"ист.прибыль={hist_j['hist_profit']}  "
                  f"→ прирост={lift_j:+.4f} ({lift_j_pct:+.1f}%)" if lift_j_pct is not None
                  else f"   Замен. {sub_code}: ист.прибыль={hist_j['hist_profit']}")
        else:
            lift_j = None
            lift_j_pct = None

        hist_total = (
            (hist_i["hist_profit"] or 0) + (hist_j["hist_profit"] or 0)
            if (hist_i["hist_profit"] is not None or hist_j["hist_profit"] is not None)
            else None
        )

        # ── Сводная таблица результатов ────────────────────────────────────────
        summary = pd.DataFrame([{
            "run_id":                run_id,
            "item_code":             item_code,
            "item_name":             s2_i["item_name"],
            "substitute_code":       sub_code,
            "substitute_name":       s2_j.get("item_name", "N/A"),
            "similarity_score":      sim_score,
            "is_kvi_item":           s2_i["is_kvi"],
            "is_kvi_substitute":     s2_j.get("is_kvi", False),
            "predicted_price_item":  round(s3_i["predicted_price"], 4),
            "ci_lower_item":         round(s3_i["lower_95"], 4),
            "ci_upper_item":         round(s3_i["upper_95"], 4),
            "predicted_price_sub":   round(s3_j["predicted_price"], 4) if s3_j.get("ok") else None,
            "ci_lower_sub":          round(s3_j["lower_95"], 4) if s3_j.get("ok") else None,
            "ci_upper_sub":          round(s3_j["upper_95"], 4) if s3_j.get("ok") else None,
            "gamma_ij":              s5["gamma_ij"],
            "gamma_ji":              s5["gamma_ji"],
            # ── Оптимальная стратегия ──────────────────────────────────────────
            "optimal_price_item":            opt["price_i"],
            "optimal_price_sub":             opt["price_j"],
            "optimal_demand_item":           opt["demand_i"],
            "optimal_demand_item_lower_95":  opt["demand_i_lower_95"],
            "optimal_demand_item_upper_95":  opt["demand_i_upper_95"],
            "optimal_demand_sub":            opt["demand_j"],
            "optimal_demand_sub_lower_95":   opt["demand_j_lower_95"],
            "optimal_demand_sub_upper_95":   opt["demand_j_upper_95"],
            "optimal_cost_item":             round(cost_i, 4),
            "optimal_cost_sub":              round(cost_j, 4),
            "optimal_profit_item":           opt["profit_i"],
            "optimal_profit_item_lower_95":  opt["profit_i_lower_95"],
            "optimal_profit_item_upper_95":  opt["profit_i_upper_95"],
            "optimal_profit_sub":            opt["profit_j"],
            "optimal_profit_sub_lower_95":   opt["profit_j_lower_95"],
            "optimal_profit_sub_upper_95":   opt["profit_j_upper_95"],
            "total_optimal_profit":          opt["total_profit"],
            "total_optimal_profit_lower_95": opt["total_profit_lower_95"],
            "total_optimal_profit_upper_95": opt["total_profit_upper_95"],
            "utility":                       opt["utility"],
            # ── Качество модели спроса ─────────────────────────────────────────
            "demand_model_item":     "price_only" if s4_i.get("anomalous_elasticity") else "full",
            "anomalous_item":        str(s4_i.get("anomalous_elasticity") or False),
            "demand_model_sub":      "price_only" if s4_j.get("anomalous_elasticity") else "full",
            "anomalous_sub":         str(s4_j.get("anomalous_elasticity") or False),
            # ── Историческая базовая линия ─────────────────────────────────────
            "hist_date_item":        hist_i["hist_date"],
            "hist_price_item":       hist_i["hist_price"],
            "hist_quantity_item":    hist_i["hist_quantity"],
            "hist_cost_item":        hist_i["hist_cost"],
            "hist_profit_item":      hist_i["hist_profit"],
            "hist_date_sub":         hist_j["hist_date"],
            "hist_price_sub":        hist_j["hist_price"],
            "hist_quantity_sub":     hist_j["hist_quantity"],
            "hist_cost_sub":         hist_j["hist_cost"],
            "hist_profit_sub":       hist_j["hist_profit"],
            "hist_total_profit":     hist_total,
            # ── Прирост ───────────────────────────────────────────────────────
            "profit_lift_item":      round(lift_i, 4) if lift_i is not None else None,
            "profit_lift_item_pct":  round(lift_i_pct, 2) if lift_i_pct is not None else None,
            "profit_lift_sub":       round(lift_j, 4) if lift_j is not None else None,
            "profit_lift_sub_pct":   round(lift_j_pct, 2) if lift_j_pct is not None else None,
            "total_profit_lift":     round(opt["total_profit"] - (hist_total or 0), 4) if hist_total is not None else None,
        }])
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"   Сохранено: summary.csv")

        # ── Дамп JSON ──────────────────────────────────────────────────────────
        def _jsonify(v):
            if isinstance(v, (np.integer,)):   return int(v)
            if isinstance(v, (np.floating,)):  return float(v)
            if isinstance(v, (np.ndarray,)):   return v.tolist()
            if isinstance(v, pd.DataFrame):    return v.to_dict(orient="records")
            if isinstance(v, pd.Timestamp):    return str(v)
            return v

        def _deep_jsonify(d):
            if isinstance(d, dict):
                return {k: _deep_jsonify(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_deep_jsonify(i) for i in d]
            return _jsonify(d)

        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(_deep_jsonify({k: v for k, v in results.items() if k != "steps" or True}),
                      f, ensure_ascii=False, indent=2, default=str)

        print(f"   Сохранено: results.json")
        print(f"\n   Все файлы: output/pipeline/{run_id}/\n")

        results["summary"] = summary
        results["output_dir"] = str(out_dir)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Pricing Pipeline")
    parser.add_argument("item_code", type=int, help="ITEMCODE товара")
    parser.add_argument("date",      type=str, help="Дата отсечки YYYY-MM-DD")
    parser.add_argument("--grid",    type=int, default=N_GRID_POINTS,
                        help=f"Количество точек сетки на ось (default={N_GRID_POINTS})")
    args = parser.parse_args()

    pipeline = DynamicPricingPipeline(n_grid=args.grid)
    result   = pipeline.run(item_code=args.item_code, date=args.date)
