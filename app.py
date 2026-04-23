"""
Price Optimizer — Web Interface (Streamlit)
============================================
Запуск (ОБЯЗАТЕЛЬНО через streamlit):
    cd /Users/kirilllipin/Dynamic-pricing
    streamlit run app.py

Версия 2: DemandModel + PriceOptimizer с кривой эластичности ε(P),
средней оптимальной ценой и ценой единичной эластичности (|ε| = 1).
"""

import sys
from pathlib import Path
import json

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Добавляем путь к модулям ───────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent / "Elasticity"
sys.path.insert(0, str(PIPELINE_DIR))

from DataPreprocessor import preprocessor
from ETL import etl_with_demand_target
from DemandModel import demand_model
from PriceOptimizer import price_optimizer
from warehouse.byer import init_warehouse, update_warehouse_day
from Baseline import compare_baselines, SimResult, MLOptimizerBaseline

# ─── Пути к данным ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "full_data.csv"
COST_PATH = BASE_DIR / "data" / "cost.csv"

# ─── Колонки, исключаемые из признаков ──────────────────────────────────────
DROP_COLS = [
    "DATE_", "CATEGORY1", "CATEGORY2",
    "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
    "AMOUNT_0D_target", "AMOUNT_1D_target", "AMOUNT_7D_target",
    "AMOUNT_15D_target", "AMOUNT_30D_target",
    "AMOUNT_1D", "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D",
]

SIGNAL_DIAGNOSTIC_COLS = [
    "price_cv_30d",
    "price_range_pct_30d",
    "price_change_share_30d",
    "selling_days_30d",
    "demand_cv_30d",
    "AMOUNT_mean_30d",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Загрузка данных (кэшируется Streamlit)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_raw_data():
    """Загружает и объединяет данные + cost. Кэшируется на 1 час."""
    data = pd.read_csv(DATA_PATH)
    cost = pd.read_csv(COST_PATH)
    merged = pd.merge(data, cost[["ITEMCODE", "cost"]], how="left", on="ITEMCODE")
    merged["DATE_"] = pd.to_datetime(merged["DATE_"])
    return merged


@st.cache_data(ttl=3600)
def get_itemcodes():
    """Список уникальных ITEMCODE."""
    raw = load_raw_data()
    return sorted(raw["ITEMCODE"].dropna().unique())


# ═════════════════════════════════════════════════════════════════════════════
#  Утилиты
# ═════════════════════════════════════════════════════════════════════════════

def find_unitary_elasticity_price(prices, elasticity_values):
    """
    Находит цену, где эластичность ε(P) = -1, через линейную интерполяцию.

    Parameters
    ----------
    prices : list/np.ndarray
        Ценовая сетка.
    elasticity_values : list/np.ndarray
        Значения эластичности для каждой цены.

    Returns
    -------
    float or None
        Цена, где ε(P) = -1, или None если пересечения нет.
    """
    eps = np.array(elasticity_values, dtype=float)
    P = np.array(prices, dtype=float)
    # Ищем знак изменения (eps + 1) — где кривая пересекает -1
    mask = (eps[:-1] + 1) * (eps[1:] + 1) < 0
    if mask.any():
        idx = np.where(mask)[0][0]
        P1, E1 = P[idx], eps[idx]
        P2, E2 = P[idx + 1], eps[idx + 1]
        # Линейная интерполяция
        return float(P1 + (P2 - P1) * (-1 - E1) / (E2 - E1))
    return None


def _create_future_row(day, itemcode, df_hist):
    """Создаёт синтетическую строку для будущего дня."""
    last_date = df_hist["DATE_"].max()
    last_row = df_hist[df_hist["DATE_"] == last_date].iloc[0].copy()
    new_row = last_row.to_dict()
    new_row["DATE_"] = pd.to_datetime(day)
    new_row["ITEMCODE"] = itemcode
    for c in ["GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D"]:
        new_row[c] = np.nan
    new_row["AMOUNT_1D"] = 0
    new_row["TOTALPRICE_1D"] = 0
    return pd.DataFrame([new_row])


def _build_actual_daily_summary(data, first_day, last_day):
    """Агрегирует исторический факт на дневной уровень для UI."""
    data = data.copy()
    data["AMOUNT"] = pd.to_numeric(data["AMOUNT"], errors="coerce")
    data["TOTALPRICE"] = pd.to_numeric(
        data["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    data["UNITPRICE"] = pd.to_numeric(
        data["UNITPRICE"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    period = data[(data["DATE_"] >= first_day) & (data["DATE_"] < last_day)].copy()
    if len(period) == 0:
        return pd.DataFrame(columns=["DATE_", "actual_amount", "actual_gmv", "actual_price"])

    period["DATE_"] = pd.to_datetime(period["DATE_"]).dt.normalize()
    sold = period[period["AMOUNT"] > 0].copy()

    daily = (
        period
        .groupby("DATE_", as_index=False)
        .agg(
            actual_amount=("AMOUNT", "sum"),
            actual_gmv=("TOTALPRICE", "sum"),
        )
    )

    if len(sold) > 0:
        actual_price = (
            sold
            .groupby("DATE_", as_index=False)
            .agg(actual_price=("UNITPRICE", "median"))
        )
        daily = daily.merge(actual_price, on="DATE_", how="left")
    else:
        daily["actual_price"] = np.nan

    return daily.sort_values("DATE_").reset_index(drop=True)


def _lookup_historical_price(data, day):
    data = data.copy()
    data["UNITPRICE"] = pd.to_numeric(
        data["UNITPRICE"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    data["AMOUNT"] = pd.to_numeric(data["AMOUNT"], errors="coerce")

    day_norm = pd.Timestamp(day).normalize()
    same_day_mask = (data["DATE_"].dt.normalize() == day_norm) & (data["AMOUNT"] > 0)
    hist_prices = data.loc[same_day_mask, "UNITPRICE"].dropna()
    if len(hist_prices) > 0:
        return float(hist_prices.median())

    # Fallback: последняя наблюдаемая цена продаж до текущего дня.
    past_mask = (data["DATE_"].dt.normalize() < day_norm) & (data["AMOUNT"] > 0)
    past_sales = data.loc[past_mask, ["DATE_", "UNITPRICE"]].dropna().sort_values("DATE_")
    if len(past_sales) > 0:
        return float(past_sales.iloc[-1]["UNITPRICE"])

    return None


def _summarize_elasticity_by_band(prices, elasticity_values, n_bands=4):
    prices = pd.Series(prices, dtype=float)
    elasticity = pd.Series(elasticity_values, dtype=float)
    valid = prices.notna() & elasticity.notna() & np.isfinite(elasticity)
    if valid.sum() == 0:
        return []

    n_unique = int(prices[valid].nunique())
    n_bins = max(1, min(n_bands, n_unique))
    labels = [f"Q{i+1}" for i in range(n_bins)]

    if n_bins == 1:
        return [{
            "price_band": labels[0],
            "price_min": float(prices[valid].min()),
            "price_max": float(prices[valid].max()),
            "mean_elasticity": float(elasticity[valid].mean()),
            "n_points": int(valid.sum()),
        }]

    bands = pd.qcut(prices[valid], q=n_bins, labels=labels, duplicates="drop")
    out = []
    for label in bands.cat.categories:
        idx = bands == label
        band_prices = prices.loc[bands.index[idx]]
        band_eps = elasticity.loc[bands.index[idx]]
        out.append({
            "price_band": str(label),
            "price_min": float(band_prices.min()),
            "price_max": float(band_prices.max()),
            "mean_elasticity": float(band_eps.mean()),
            "n_points": int(idx.sum()),
        })
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Ядро симуляции
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(itemcode, first_day, n_days, progress_bar, status_text):
    """
    Запускает полный цикл симуляции для одного ITEMCODE.

    Использует новый пайплайн:
    - ETL: etl_with_demand_target (таргет = AMOUNT_7D_target)
      - Модель: DemandModel (Ridge regression, Q(P_next_day))
      - Оптимизатор: price_optimizer (grid search + scipy refinement)

    Возвращает (history_pred_df, baseline_table, elasticity_data).

    elasticity_data содержит:
      - prices: ценовая сетка (последний день)
      - qty: предсказанный спрос при каждой цене
      - gmv: P * Q при каждой цене
      - elasticity: ε(P) при каждой цене
      - optimal_prices: список оптимальных цен по дням
      - unitary_price: цена где ε = -1
    """

    raw = load_raw_data()
    raw_item = raw[raw["ITEMCODE"] == itemcode].copy()
    data = preprocessor(raw_item.copy())

    last_day = first_day + pd.Timedelta(days=n_days)

    # ── 1. ETL на всех данных ──────────────────────────────────────────
    status_text.text("🔧 Подготовка features…")
    progress_bar.progress(5)

    etl_with_demand_target(data[data["DATE_"] < last_day].copy())  # прогреваем кэш ETL
    progress_bar.progress(10)

    # ── 2. Инициализация склада ──────────────────────────────────────────
    status_text.text("🏭 Инициализация склада…")
    warehouse_state_path = BASE_DIR / f"warehouse_state_{itemcode}.pkl"
    warehouse_store_csv  = BASE_DIR / f"warehouse_store_{itemcode}.csv"

    warehouse = init_warehouse(
        data.copy(),
        itemcode=itemcode,
        base_dir=BASE_DIR,
        state_path=warehouse_state_path,
        store_csv_path=warehouse_store_csv,
    )
    progress_bar.progress(15)

    # ── 3. Основной цикл ─────────────────────────────────────────────────
    history_pred = pd.DataFrame(
        columns=["DATE_", "ITEMCODE", "unitprice", "gmv", "margin",
                  "quantity", "margin_percent", "cost", "elasticity",
                  "historical_price", "price_delta_pct",
                  "quantity_lower", "quantity_upper",
                  "gmv_lower", "gmv_upper",
                  "margin_lower", "margin_upper",
                  "penalty_value"] + SIGNAL_DIAGNOSTIC_COLS
    )

    # Кэш моделей (JSON с датами последнего обучения)
    model_cache_path = BASE_DIR / "model_cache.json"
    if model_cache_path.exists():
        with open(model_cache_path, "r") as f:
            model_cache = json.load(f)
    else:
        model_cache = {}

    day = first_day
    total_steps = n_days
    step = 0
    model_demand = None

    # Для графиков эластичности (последний день симуляции)
    elasticity_data = {
        "prices": [],
        "qty": [],
        "qty_lower": [],
        "qty_upper": [],
        "gmv": [],
        "gmv_lower": [],
        "gmv_upper": [],
        "elasticity": [],
        "optimal_prices": [],   # оптимальная цена каждого дня
        "unitary_price": None,  # цена где ε = -1
        "elasticity_by_band": [],
        "interval_level": None,
    }

    while day < last_day:
        step += 1
        pct = 15 + int(70 * step / total_steps)
        progress_bar.progress(pct)
        status_text.text(f"🔄 День {day.date()} ({step}/{total_steps})")

        # ── Склад ────────────────────────────────────────────────────────
        warehouse_record = update_warehouse_day(
            warehouse, day, actual_sales=None,
            state_path=warehouse_state_path,
            store_csv_path=warehouse_store_csv,
        )

        # ── Данные для модели ────────────────────────────────────────────
        df_hist = data[data["DATE_"] < day].copy()
        future_row = _create_future_row(day, itemcode, df_hist)
        df_hist = pd.concat([df_hist, future_row], ignore_index=True)
        df_hist = df_hist.sort_values("DATE_").reset_index(drop=True)

        # ETL с таргетом AMOUNT_1D_target
        df_etl = etl_with_demand_target(df_hist)

        # Строка-шаблон для сегодняшнего дня
        today_template = (
            df_etl[df_etl["DATE_"] == day]
            .drop(columns=DROP_COLS, errors="ignore")
            .copy()
        )
        today_template["Id"] = 0

        # ── Обучение модели спроса Q(P) ──────────────────────────────────
        cutoff = day - pd.Timedelta(days=1)
        train_full = df_etl[(df_etl["DATE_"] < day) & (df_etl["DATE_"] < cutoff)].copy()

        target = train_full["AMOUNT_7D_target"].copy()
        train_X = train_full.drop(columns=DROP_COLS, errors="ignore")

        valid = train_X.notna().all(axis=1) & target.notna()
        train_X, target = train_X[valid], target[valid]

        if len(train_X) > 0 and len(target) > 0:
            model_demand = demand_model(train=train_X, target=target)
        elif model_demand is None:
            st.warning(f"⚠️ Недостаточно данных для обучения на {day.date()}, пропускаем день.")
            day += pd.Timedelta(days=1)
            continue

        progress_bar.progress(min(pct + 3, 85))

        # ── Оптимизация цены ─────────────────────────────────────────────
        hist_price_day = _lookup_historical_price(raw_item, day)
        item_cost = float(data["cost"].iloc[0]) if "cost" in data.columns else 0.0
        item_baseprice = (
            hist_price_day
            if hist_price_day is not None
            else float(today_template["UNITPRICE"].iloc[0]) if "UNITPRICE" in today_template.columns else None
        )

        d = price_optimizer(
            demand_model=model_demand,
            row_features=today_template,
            stock=float(warehouse_record["available"]),
            cost=item_cost,
            base_price=item_baseprice,
            objective="profit",
            n_grid=300,
        )

        optimal_price = round(float(np.nan_to_num(d["unitprice"], nan=0.0)), 2)
        realized_qty = int(round(float(np.nan_to_num(d["quantity"], nan=0.0))))

        # Пересчитываем GMV и маржу через целое количество
        realized_gmv = round(optimal_price * realized_qty, 2)
        realized_margin = round((optimal_price - item_cost) * realized_qty, 2)
        realized_margin_pct = (
            round((optimal_price - item_cost) / optimal_price * 100, 2)
            if optimal_price > 0 else 0.0
        )

        # Сохраняем оптимальную цену
        elasticity_data["optimal_prices"].append(optimal_price)

        # ── Кривая эластичности ε(P) для последнего дня ──────────────────
        if day == last_day - pd.Timedelta(days=1):
            result_obj = d.get("_result_obj")
            if result_obj is not None and len(result_obj.price_grid) > 0:
                price_grid = np.asarray(result_obj.price_grid, dtype=float)
                q_values = np.asarray(result_obj.demand_grid, dtype=float)
                q_lower = np.asarray(
                    getattr(result_obj, "demand_lower_grid", result_obj.demand_grid),
                    dtype=float,
                )
                q_upper = np.asarray(
                    getattr(result_obj, "demand_upper_grid", result_obj.demand_grid),
                    dtype=float,
                )
                eps_values = np.asarray(
                    getattr(result_obj, "elasticity_grid", np.full_like(price_grid, np.nan)),
                    dtype=float,
                )

                # Fallback: если оптимизатор не вернул сетку эластичности,
                # вычисляем её напрямую через модель спроса на ценовой сетке.
                if not np.isfinite(eps_values).any():
                    eps_rows = []
                    for p in price_grid:
                        row = today_template.copy()
                        row["UNITPRICE"] = p
                        if "BASEPRICE" in row.columns:
                            row["discount"] = 1.0 - row["UNITPRICE"] / row["BASEPRICE"].replace(0, 1e-8)
                        if "cost" in row.columns:
                            row["margin_%"] = (row["UNITPRICE"] - row["cost"]) / row["cost"].replace(0, 1e-8)
                        eps_rows.append(row)

                    eps_batch = pd.concat(eps_rows, ignore_index=True)
                    eps_values = np.asarray(model_demand.elasticity(eps_batch), dtype=float)

                q_max = q_values.max() if len(q_values) else 0.0
                q_threshold = max(1.0, 0.05 * q_max)
                valid_mask = q_values > q_threshold

                if optimal_price > 0:
                    price_mask = (price_grid >= 0.5 * optimal_price) & (price_grid <= 2.0 * optimal_price)
                    valid_mask = valid_mask & price_mask

                if valid_mask.sum() < 5:
                    valid_mask = q_values > max(0.5, 0.02 * q_max)

                price_grid_filtered = price_grid[valid_mask]
                q_values_filtered = q_values[valid_mask]
                q_lower_filtered = q_lower[valid_mask]
                q_upper_filtered = q_upper[valid_mask]
                eps_clipped = eps_values[valid_mask]

                # Сохраняем ОТФИЛЬТРОВАННЫЕ данные (без выбросов от деления на Q≈0)
                elasticity_data["prices"] = price_grid_filtered.tolist()
                elasticity_data["qty"] = q_values_filtered.tolist()
                elasticity_data["qty_lower"] = q_lower_filtered.tolist()
                elasticity_data["qty_upper"] = q_upper_filtered.tolist()
                elasticity_data["gmv"] = (price_grid_filtered * q_values_filtered).tolist()
                elasticity_data["gmv_lower"] = (price_grid_filtered * q_lower_filtered).tolist()
                elasticity_data["gmv_upper"] = (price_grid_filtered * q_upper_filtered).tolist()
                elasticity_data["elasticity"] = eps_clipped.tolist()
                elasticity_data["elasticity_by_band"] = _summarize_elasticity_by_band(
                    price_grid_filtered, eps_clipped
                )
                summary = getattr(model_demand, "get_uncertainty_summary", lambda: {})()
                if "interval_level" in summary:
                    elasticity_data["interval_level"] = float(summary["interval_level"])

                # Находим цену единичной эластичности (ε = -1)
                elasticity_data["unitary_price"] = find_unitary_elasticity_price(
                    price_grid_filtered, eps_clipped
                )

        # Разница между оптимальной и исторической ценой
        if hist_price_day is not None and hist_price_day > 0:
            price_delta_pct = (optimal_price - hist_price_day) / hist_price_day * 100
        else:
            price_delta_pct = None

        # ── Запись в историю ─────────────────────────────────────────────
        history_pred = pd.concat(
            [history_pred,
             pd.DataFrame([{
                  "DATE_": day,
                  "ITEMCODE": itemcode,
                 "unitprice": optimal_price,
                  "gmv": realized_gmv,
                  "gmv_lower": d.get("gmv_lower", realized_gmv),
                  "gmv_upper": d.get("gmv_upper", realized_gmv),
                  "margin": realized_margin,
                  "margin_lower": d.get("margin_lower", realized_margin),
                  "margin_upper": d.get("margin_upper", realized_margin),
                  "quantity": realized_qty,
                  "quantity_lower": d.get("quantity_lower", float(realized_qty)),
                  "quantity_upper": d.get("quantity_upper", float(realized_qty)),
                  "margin_percent": realized_margin_pct,
                  "cost": item_cost,
                  "baseprice": item_baseprice,
                  "elasticity": d.get("elasticity", float("nan")),
                  "penalty_value": d.get("penalty_value", 0.0),
                  "historical_price": hist_price_day,
                  "price_delta_pct": price_delta_pct,
                  **{
                      col: float(today_template[col].iloc[0]) if col in today_template.columns else np.nan
                      for col in SIGNAL_DIAGNOSTIC_COLS
                  },
              }])],
            ignore_index=True,
        )

        # Обновляем склад
        warehouse.rest = max(warehouse_record["available"] - realized_qty, 0)
        day += pd.Timedelta(days=1)

    # ── 4. Сохранение результатов ────────────────────────────────────────
    actual_daily = _build_actual_daily_summary(raw_item, first_day, last_day)
    if len(actual_daily) > 0:
        history_pred = history_pred.copy()
        actual_daily = actual_daily.copy()

        # Merge требует одинаковый dtype ключа; приводим обе стороны к daily datetime.
        history_pred["DATE_"] = pd.to_datetime(history_pred["DATE_"], errors="coerce").dt.normalize()
        actual_daily["DATE_"] = pd.to_datetime(actual_daily["DATE_"], errors="coerce").dt.normalize()

        history_pred = history_pred.merge(actual_daily, on="DATE_", how="left")

    out_csv = BASE_DIR / f"opt_results_{itemcode}.csv"
    history_pred.to_csv(out_csv, index=False)

    # ── 5. Baseline comparison ───────────────────────────────────────────
    status_text.text("📊 Сравнение с baseline…")
    progress_bar.progress(90)

    df_sim = data[(data["DATE_"] >= first_day) & (data["DATE_"] < last_day)].copy()

    baseline_table = None
    try:
        baseline_table = compare_baselines(
            df=df_sim,
            ml_results=history_pred,
            demand_col="AMOUNT",
            price_col="UNITPRICE",
            cost_col="cost",
            date_col="DATE_",
            initial_stock=0.0,
            markup=0.30,
        )
    except Exception as e:
        st.warning(f"⚠️ Baseline-сравнение не удалось: {e}. Показываю только ML-результаты.")
        if len(history_pred) > 0:
            ml_res = SimResult(
                strategy_name="ML_Optimizer",
                total_gmv=float(history_pred["gmv"].sum()),
                total_margin=float(history_pred["margin"].sum()),
                avg_margin_pct=float(history_pred["margin_percent"].mean()) / 100,
            )
            baseline_table = ml_res.summary().to_frame().T.set_index("strategy")
        else:
            baseline_table = pd.DataFrame()

    # Проверяем что ML_Optimizer есть в таблице
    if "ML_Optimizer" not in baseline_table.index:
        st.warning("⚠️ ML_Optimizer не попал в таблицу baseline. Добавляю вручную…")
        ml_strat = MLOptimizerBaseline(history_pred)
        ml_res = ml_strat.simulate(
            df=df_sim,
            demand_col="AMOUNT",
            price_col="UNITPRICE",
            cost_col="cost",
            date_col="DATE_",
            initial_stock=0.0,
        )
        baseline_table = pd.concat([
            baseline_table,
            ml_res.summary().to_frame().T.set_index("strategy")
        ])

    progress_bar.progress(100)
    status_text.text("✅ Готово!")

    return history_pred, baseline_table, elasticity_data


# ═════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Price Optimizer", layout="wide")

st.title("🎯 Price Optimizer")
st.markdown("Выберите товар, задайте период и запустите прогноз.")

# ─── Sidebar: настройки ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Параметры")

    itemcodes = get_itemcodes()

    selected_itemcode = st.selectbox(
        "📦 ITEMCODE",
        options=itemcodes,
        index=itemcodes.index(107) if 107 in itemcodes else 0,
    )

    @st.cache_data(ttl=3600)
    def _get_date_range(ic):
        raw = load_raw_data()
        item_data = raw[raw["ITEMCODE"] == ic]
        sold = item_data[item_data["AMOUNT"] > 0]
        if len(sold) == 0:
            sold = item_data
        return sold["DATE_"].min().date(), sold["DATE_"].max().date()

    min_date, max_date = _get_date_range(selected_itemcode)

    start_date = st.date_input(
        "📅 Дата начала",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    n_days = st.slider(
        "📆 Период прогноза (дней)",
        min_value=1,
        max_value=30,
        value=10,
    )

    st.divider()

    run_btn = st.button("🚀 Спрогнозировать", type="primary", use_container_width=True)

# ─── Основная область ────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    progress_bar = st.progress(0, text="Инициализация…")
    status_text = st.empty()

    try:
        first_day = pd.Timestamp(start_date)
        hist_pred, baseline_tbl, elast_data = run_simulation(
            itemcode=int(selected_itemcode),
            first_day=first_day,
            n_days=n_days,
            progress_bar=progress_bar,
            status_text=status_text,
        )
        st.session_state.results = {
            "history_pred": hist_pred,
            "baseline": baseline_tbl,
            "elasticity_data": elast_data,
            "itemcode": int(selected_itemcode),
            "period": f"{start_date} → {start_date + pd.Timedelta(days=n_days)}",
        }
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

# ─── Отображение результатов ─────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results

    st.divider()
    st.subheader(f"📊 Результаты: ITEMCODE {res['itemcode']}  |  {res['period']}")

    # 1. Таблица baseline comparison
    st.markdown("### Сравнение стратегий")
    st.caption("Считается на дневном уровне по всему выбранному периоду и использует исторический спрос/склад, а не прогнозный GMV модели.")
    st.dataframe(res["baseline"], use_container_width=True)

    # 2. ML vs лучший baseline — дельта
    st.markdown("### 📈 ML vs лучший baseline")

    if "ML_Optimizer" not in res["baseline"].index:
        st.error("❌ ML_Optimizer отсутствует в таблице baseline. Проверьте логи выше.")
        st.dataframe(res["baseline"], use_container_width=True)
    else:
        ml_row = res["baseline"].loc["ML_Optimizer"]
        other_strategies = res["baseline"].drop(index="ML_Optimizer")
        if len(other_strategies) == 0:
            st.warning("Нет baseline-стратегий для сравнения.")
            best_baseline_name = "N/A"
            best_baseline = None
        else:
            best_baseline_name = other_strategies["total_gmv"].idxmax()
            best_baseline = other_strategies.loc[best_baseline_name]

        def _delta(ml, base):
            if abs(base) < 1e-8:
                return 0.0
            return (ml - base) / abs(base) * 100

        if best_baseline is not None:
            delta_gmv = _delta(ml_row["total_gmv"], best_baseline["total_gmv"])
            delta_margin = _delta(ml_row["total_margin"], best_baseline["total_margin"])
            delta_margin_pct = _delta(ml_row["avg_margin_pct"], best_baseline["avg_margin_pct"])
            delta_fill = _delta(ml_row["fill_rate_%"], best_baseline["fill_rate_%"])

            delta_cols = st.columns(4)
            delta_cols[0].metric(
                "Δ GMV", f"{delta_gmv:+.1f}%",
                help=f"ML: {ml_row['total_gmv']:,.0f} vs {best_baseline_name}: {best_baseline['total_gmv']:,.0f}"
            )
            delta_cols[1].metric(
                "Δ Margin", f"{delta_margin:+.1f}%",
                help=f"ML: {ml_row['total_margin']:,.0f} vs {best_baseline_name}: {best_baseline['total_margin']:,.0f}"
            )
            delta_cols[2].metric(
                "Δ Margin %", f"{delta_margin_pct:+.1f} п.п.",
                help=f"ML: {ml_row['avg_margin_pct']:.1f}% vs {best_baseline_name}: {best_baseline['avg_margin_pct']:.1f}%"
            )
            delta_cols[3].metric(
                "Δ Fill Rate", f"{delta_fill:+.1f}%",
                help=f"ML: {ml_row['fill_rate_%']:.1f}% vs {best_baseline_name}: {best_baseline['fill_rate_%']:.1f}%"
            )

            st.caption(f"Лучший baseline: **{best_baseline_name}**")
        else:
            st.caption("Нет baseline для сравнения")

        # 3. Ключевые метрики ML
        st.markdown("### Ключевые метрики ML-оптимизатора")

        # Вычисляем дополнительные метрики из elasticity_data и history_pred
        elast = res.get("elasticity_data", {})
        opt_prices = elast.get("optimal_prices", [])
        avg_optimal_price = float(np.mean(opt_prices)) if opt_prices else None
        unitary_price = elast.get("unitary_price")
        interval_level = elast.get("interval_level")

        # Средняя разница цены vs исторической
        hist_pred_df = res.get("history_pred")
        avg_price_delta_pct = None
        if hist_pred_df is not None and "price_delta_pct" in hist_pred_df.columns:
            valid_deltas = hist_pred_df["price_delta_pct"].dropna()
            if len(valid_deltas) > 0:
                avg_price_delta_pct = float(valid_deltas.mean())

        # 7 колонок: GMV, Margin, Margin%, Fill Rate, Avg Opt Price, Avg Δ vs Hist, Unitary ε Price
        cols = st.columns(7)
        cols[0].metric("GMV", f"{ml_row['total_gmv']:,.0f}")
        cols[1].metric("Margin", f"{ml_row['total_margin']:,.0f}")
        cols[2].metric("Margin %", f"{ml_row['avg_margin_pct']:.1f}%")
        cols[3].metric("Fill Rate", f"{ml_row['fill_rate_%']:.1f}%")

        if avg_optimal_price is not None:
            cols[4].metric(
                "Avg Optimal Price",
                f"{avg_optimal_price:,.2f}",
                help="Средняя оптимальная цена за период симуляции"
            )
        else:
            cols[4].metric("Avg Optimal Price", "N/A")

        if avg_price_delta_pct is not None:
            cols[5].metric(
                "Avg Δ vs Hist",
                f"{avg_price_delta_pct:+.1f}%",
                help="Средняя разница: насколько ML-цена выше/ниже исторической"
            )
        else:
            cols[5].metric("Avg Δ vs Hist", "N/A")

        if unitary_price is not None:
            cols[6].metric(
                "Unitary ε Price",
                f"{unitary_price:,.2f}",
                help="Цена, где эластичность спроса ε(P) = −1 (максимум выручки)"
            )
        else:
            cols[6].metric("Unitary ε Price", "N/A")

        # 3b. Диагностика сигнала по SKU
        st.markdown("### 🔎 Диагностика сигнала SKU")
        st.caption("Показатели рассчитываются из прошлых ~30 дней относительно каждого дня оптимизации и помогают понять, насколько надёжно интерпретировать ценовую кривую.")

        latest_diag = hist_pred_df[SIGNAL_DIAGNOSTIC_COLS].dropna(how="all").tail(1)
        latest_diag = latest_diag.iloc[0] if len(latest_diag) else pd.Series(dtype=float)

        diag_cols = st.columns(5)
        diag_cols[0].metric(
            "Price CV 30d",
            f"{latest_diag.get('price_cv_30d', float('nan')):.2f}" if len(latest_diag) else "N/A",
            help="Коэффициент вариации цены за последние 30 дней: чем выше, тем больше ценового сигнала в истории."
        )
        diag_cols[1].metric(
            "Price range 30d",
            f"{latest_diag.get('price_range_pct_30d', float('nan')) * 100:.1f}%" if len(latest_diag) else "N/A",
            help="Относительный диапазон цен max-min к средней цене за последние 30 дней."
        )
        diag_cols[2].metric(
            "Price changes 30d",
            f"{latest_diag.get('price_change_share_30d', float('nan')) * 100:.1f}%" if len(latest_diag) else "N/A",
            help="Доля дней с изменением цены в окне; низкое значение означает слабый идентифицируемый ценовой сигнал."
        )
        diag_cols[3].metric(
            "Selling days 30d",
            f"{latest_diag.get('selling_days_30d', float('nan')):.0f}" if len(latest_diag) else "N/A",
            help="Число наблюдаемых дней продаж в недавнем окне."
        )
        diag_cols[4].metric(
            "Demand CV 30d",
            f"{latest_diag.get('demand_cv_30d', float('nan')):.2f}" if len(latest_diag) else "N/A",
            help="Волатильность спроса: std/mean за последние 30 дней. Высокое значение означает более шумный спрос."
        )

        if interval_level is not None:
            st.caption(f"Интервалы ниже — лёгкие residual-based bands уровня примерно {interval_level:.0%}; это ориентир по неопределённости, а не строгая вероятностная гарантия.")

    # 4. Графики эластичности и оптимизации (3 подграфика)
    st.markdown("### 📈 Графики эластичности")

    if elast.get("prices") and len(elast["prices"]) > 0:
        prices = elast["prices"]
        qty = elast["qty"]
        qty_lower = elast.get("qty_lower", qty)
        qty_upper = elast.get("qty_upper", qty)
        gmv = elast["gmv"]
        gmv_lower = elast.get("gmv_lower", gmv)
        gmv_upper = elast.get("gmv_upper", gmv)
        eps = elast["elasticity"]
        opt_price = opt_prices[-1] if opt_prices else None  # последний день
        unitary_p = elast.get("unitary_price")

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Цена → Количество  Q(P)",
                "Цена → GMV  P·Q(P)",
                "Цена → Эластичность  ε(P)"
            )
        )

        # ── График 1: Цена → Количество ──────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=prices, y=qty, mode="lines",
                name="Количество",
                line=dict(color="steelblue", width=2.5),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=prices + prices[::-1],
                y=qty_upper + qty_lower[::-1],
                fill="toself",
                fillcolor="rgba(70,130,180,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=1,
        )
        # Вертикальная линия оптимальной цены
        if opt_price is not None:
            fig.add_vline(
                x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}",
                annotation_position="top right",
                row=1, col=1,
            )
        # Вертикальная линия единичной эластичности
        if unitary_p is not None:
            fig.add_vline(
                x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"|ε|=1: {unitary_p:.2f}",
                annotation_position="bottom left",
                row=1, col=1,
            )
        fig.update_xaxes(title_text="Цена (UNITPRICE)", row=1, col=1)
        fig.update_yaxes(title_text="Количество Q", row=1, col=1)

        # ── График 2: Цена → GMV ─────────────────────────────────────
        fig.add_trace(
            go.Scatter(
                x=prices, y=gmv, mode="lines",
                name="GMV",
                line=dict(color="crimson", width=2.5),
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=prices + prices[::-1],
                y=gmv_upper + gmv_lower[::-1],
                fill="toself",
                fillcolor="rgba(220,20,60,0.14)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1, col=2,
        )
        if opt_price is not None:
            fig.add_vline(
                x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}",
                annotation_position="top right",
                row=1, col=2,
            )
        if unitary_p is not None:
            fig.add_vline(
                x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"|ε|=1: {unitary_p:.2f}",
                annotation_position="bottom left",
                row=1, col=2,
            )
        fig.update_xaxes(title_text="Цена (UNITPRICE)", row=1, col=2)
        fig.update_yaxes(title_text="GMV (P·Q)", row=1, col=2)

        # ── График 3: Цена → Эластичность ε(P) ───────────────────────
        fig.add_trace(
            go.Scatter(
                x=prices, y=eps, mode="lines",
                name="Эластичность ε(P)",
                line=dict(color="darkviolet", width=2.5),
            ),
            row=1, col=3,
        )
        # Горизонтальная линия ε = -1
        fig.add_hline(
            y=-1, line_dash="dash", line_color="red", line_width=1.5,
            annotation_text="ε = −1",
            annotation_position="right",
            row=1, col=3,
        )
        # Вертикальная линия оптимальной цены
        if opt_price is not None:
            fig.add_vline(
                x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}",
                annotation_position="top right",
                row=1, col=3,
            )
        # Вертикальная линия единичной эластичности
        if unitary_p is not None:
            fig.add_vline(
                x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"P_unitary = {unitary_p:.2f}",
                annotation_position="bottom left",
                row=1, col=3,
            )
        fig.update_xaxes(title_text="Цена (UNITPRICE)", row=1, col=3)
        fig.update_yaxes(title_text="Эластичность ε(P)", row=1, col=3)

        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Легенда-пояснение под графиками
        st.caption(
            "🟢 **P\\*** — оптимальная цена (максимум прибыли)  |  "
            "🟠 **|ε|=1** — цена единичной эластичности (максимум выручки)  |  "
            "🔴 **ε = −1** — граница эластичного/неэластичного спроса  |  "
            "полупрозрачные области — residual-based интервалы спроса/GMV"
        )

        band_table = pd.DataFrame(elast.get("elasticity_by_band", []))
        if len(band_table) > 0:
            st.markdown("### 📐 Эластичность по ценовым диапазонам")
            st.caption("Q1 — самые низкие цены в сетке, последний диапазон — самые высокие.")
            st.dataframe(
                band_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "price_band": "Диапазон",
                    "price_min": st.column_config.NumberColumn("Цена min", format="%.2f"),
                    "price_max": st.column_config.NumberColumn("Цена max", format="%.2f"),
                    "mean_elasticity": st.column_config.NumberColumn("Средняя ε", format="%.3f"),
                    "n_points": st.column_config.NumberColumn("Точек", format="%d"),
                },
            )

    # 5. Детализация по дням
    st.markdown("### Прогноз модели по дням")
    st.caption("Это прогноз ML-модели для выбранных цен. Исторический факт за день показан в отдельных колонках для сопоставления.")
    daily = res["history_pred"].copy()
    daily["DATE_"] = pd.to_datetime(daily["DATE_"]).dt.date
    
    # Определяем какие колонки доступны
    display_cols = ["DATE_", "unitprice", "historical_price", "actual_price", "price_delta_pct",
                    "baseprice", "cost", "quantity", "quantity_lower", "quantity_upper",
                    "gmv", "gmv_lower", "gmv_upper", "margin", "margin_lower", "margin_upper",
                    "actual_amount", "actual_gmv",
                    "margin_percent", "elasticity", "penalty_value"] + SIGNAL_DIAGNOSTIC_COLS
    available_cols = [c for c in display_cols if c in daily.columns]
    
    st.dataframe(
        daily[available_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "DATE_": "Дата",
            "unitprice": st.column_config.NumberColumn("ML Цена", format="%.2f",
                                                       help="Оптимальная цена от ML-оптимизатора"),
            "historical_price": st.column_config.NumberColumn("Историческая цена", format="%.2f",
                                                               help="Медианная цена за этот день в истории"),
            "actual_price": st.column_config.NumberColumn("Факт. цена", format="%.2f",
                                                          help="Медианная фактическая цена продаж в этот день"),
            "price_delta_pct": st.column_config.NumberColumn("Δ vs Hist", format="%+.1f%%",
                                                             help="Разница ML цены vs исторической: +завышаем, -занижаем"),
            "baseprice": st.column_config.NumberColumn("Base Price", format="%.2f"),
            "cost": st.column_config.NumberColumn("Cost", format="%.2f"),
            "quantity": st.column_config.NumberColumn("Прогноз кол-ва", format="%d"),
            "quantity_lower": st.column_config.NumberColumn("Кол-во low", format="%.2f",
                                                            help="Нижняя граница residual-based интервала спроса"),
            "quantity_upper": st.column_config.NumberColumn("Кол-во high", format="%.2f",
                                                            help="Верхняя граница residual-based интервала спроса"),
            "gmv": st.column_config.NumberColumn("Прогноз GMV", format="%.2f"),
            "gmv_lower": st.column_config.NumberColumn("GMV low", format="%.2f"),
            "gmv_upper": st.column_config.NumberColumn("GMV high", format="%.2f"),
            "margin": st.column_config.NumberColumn("Маржа", format="%.2f"),
            "margin_lower": st.column_config.NumberColumn("Маржа low", format="%.2f"),
            "margin_upper": st.column_config.NumberColumn("Маржа high", format="%.2f"),
            "actual_amount": st.column_config.NumberColumn("Факт. кол-во", format="%.2f",
                                                           help="Суммарный фактический спрос за день"),
            "actual_gmv": st.column_config.NumberColumn("Факт. GMV", format="%.2f",
                                                         help="Суммарный фактический GMV за день"),
            "margin_percent": st.column_config.NumberColumn("Маржа %", format="%.1f"),
            "elasticity": st.column_config.NumberColumn("ε(P*)", format="%.3f"),
            "penalty_value": st.column_config.NumberColumn("Soft penalty", format="%.2f",
                                                           help="Штраф от мягких ограничений: низкая маржа + сильное отклонение от базовой цены"),
            "price_cv_30d": st.column_config.NumberColumn("Price CV 30d", format="%.2f"),
            "price_range_pct_30d": st.column_config.NumberColumn("Price range 30d", format="%.2f"),
            "price_change_share_30d": st.column_config.NumberColumn("Price changes 30d", format="%.2f"),
            "selling_days_30d": st.column_config.NumberColumn("Selling days 30d", format="%.0f"),
            "demand_cv_30d": st.column_config.NumberColumn("Demand CV 30d", format="%.2f"),
            "AMOUNT_mean_30d": st.column_config.NumberColumn("Avg demand 30d", format="%.2f"),
        },
    )

    # 6. Кнопка нового прогноза
    st.divider()
    if st.button("🔄 Новый прогноз", use_container_width=True):
        st.session_state.results = None
        st.rerun()
