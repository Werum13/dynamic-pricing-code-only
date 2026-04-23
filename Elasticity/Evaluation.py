"""
Evaluation.py
=============
Модуль оценки модели эластичности / спроса.

Методология
-----------
Для временных рядов случайный train/test сплит недопустим — он создаёт
«утечку будущего» (data leakage): модель видит будущие наблюдения при
обучении. Поэтому используется **временной сплит**:

    train: t ∈ [t_0, t_0 + (1-τ)·T)
    test:  t ∈ [t_0 + (1-τ)·T, t_0 + T]

где τ — доля тестовой выборки (по умолчанию 0.2).

Метрики
-------
R²   = 1 - SS_res / SS_tot
     = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²

MAE  = (1/n) Σ |yᵢ - ŷᵢ|

RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]

MAPE = (100/n) Σ |yᵢ - ŷᵢ| / max(|yᵢ|, ε)
     где ε = 1e-8 — защита от деления на ноль

Точечная ценовая эластичность (PED)
------------------------------------
Для GMV-модели эластичность восстанавливается через Q̂(P) = GMV(P)/P.
Для demand-модели Q̂(P) получается напрямую.

    ε̂(P) = dQ̂/dP · P/Q̂
          ≈ [Q̂(P+δ) - Q̂(P-δ)] / (2δ) · P / Q̂(P)

где δ = 0.01·P — малое приращение цены (1 %).
Нормально ε < 0 (закон спроса). Значение ε < -1 — эластичный товар.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from DemandModel import DemandModel


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass для хранения результатов
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Контейнер метрик для одного прогона оценки."""
    r2:   float = float('nan')
    mae:  float = float('nan')
    rmse: float = float('nan')
    mape: float = float('nan')
    n_train: int = 0
    n_test:  int = 0
    # Средняя эластичность по тестовой выборке
    mean_ped: float = float('nan')
    ped_by_price_band: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        base = asdict(self)
        ped_by_band = base.pop('ped_by_price_band', {})
        return {**base, **ped_by_band}

    def __str__(self) -> str:
        return (
            f"EvalMetrics | n_train={self.n_train}, n_test={self.n_test}\n"
            f"  R²   = {self.r2:+.4f}\n"
            f"  MAE  = {self.mae:.4f}\n"
            f"  RMSE = {self.rmse:.4f}\n"
            f"  MAPE = {self.mape:.2f}%\n"
            f"  Mean PED = {self.mean_ped:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции метрик
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-8


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Коэффициент детерминации R².

    R² = 1 - SS_res / SS_tot
    SS_res = Σ(yᵢ - ŷᵢ)²
    SS_tot = Σ(yᵢ - ȳ)²

    R² = 1  → идеальная подгонка
    R² = 0  → модель равна константе ȳ
    R² < 0  → модель хуже константы (переобучение или неправильная спецификация)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < _EPS:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE = (1/n) Σ |yᵢ - ŷᵢ|"""
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = √[(1/n) Σ (yᵢ - ŷᵢ)²]"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE = (100/n) Σ |yᵢ - ŷᵢ| / max(|yᵢ|, ε)

    Защита от деления на 0: знаменатель обрезается снизу на ε=1e-8.
    Нули в y_true завышают MAPE — для GMV-задач это типично.
    """
    denom = np.maximum(np.abs(y_true), _EPS)
    return float(100.0 * np.mean(np.abs(y_true - y_pred) / denom))


# ─────────────────────────────────────────────────────────────────────────────
# Оценка эластичности (PED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_point_elasticity(
    model,
    df_test: pd.DataFrame,
    delta_frac: float = 0.01
) -> np.ndarray:
    """
    Численная оценка точечной ценовой эластичности спроса.

    Формула центральной разности:
        ε̂(P) = [Q̂(P+δ) - Q̂(P-δ)] / (2δ) · P / Q̂(P)

    Параметры
    ---------
    model : DemandModel
        Обученная модель спроса Q(P).
    df_test : pd.DataFrame
        Тестовая выборка (содержит UNITPRICE).
    delta_frac : float
        Доля изменения цены для численной производной (1 % по умолчанию).

    Возвращает
    ----------
    np.ndarray
        Вектор точечных эластичностей для каждого наблюдения.
    """
    if not isinstance(model, DemandModel):
        raise TypeError(f"Ожидается DemandModel, получен {type(model).__name__}")
    return model.elasticity(df_test.copy(), delta_frac=delta_frac)


def summarize_elasticity_by_price_band(
    df_test: pd.DataFrame,
    ped: np.ndarray,
    price_col: str = 'UNITPRICE',
    n_bands: int = 4,
) -> Dict[str, float]:
    """
    Сводка средней эластичности по квантильным ценовым диапазонам.

    Возвращает плоский словарь:
      ped_band_q1, ped_band_q1_n, ..., ped_band_q4, ped_band_q4_n
    где q1 — самый низкий ценовой диапазон.
    """
    if price_col not in df_test.columns:
        return {}

    price = df_test[price_col]
    ped_series = pd.Series(np.asarray(ped, dtype=float), index=df_test.index)
    valid = price.notna() & ped_series.notna() & np.isfinite(ped_series)

    if valid.sum() == 0:
        return {}

    n_unique = int(price[valid].nunique())
    n_bins = max(1, min(n_bands, n_unique))
    labels = [f'q{i + 1}' for i in range(n_bins)]

    if n_bins == 1:
        return {
            'ped_band_q1': float(np.nanmean(ped_series[valid])),
            'ped_band_q1_n': int(valid.sum()),
        }

    try:
        bands = pd.qcut(price[valid], q=n_bins, labels=labels, duplicates='drop')
    except ValueError:
        return {
            'ped_band_q1': float(np.nanmean(ped_series[valid])),
            'ped_band_q1_n': int(valid.sum()),
        }

    out: Dict[str, float] = {}
    for label in bands.cat.categories:
        mask = bands == label
        values = ped_series.loc[bands.index[mask]]
        out[f'ped_band_{label}'] = float(np.nanmean(values)) if len(values) else float('nan')
        out[f'ped_band_{label}_n'] = int(mask.sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс оценки
# ─────────────────────────────────────────────────────────────────────────────

class ElasticityEvaluator:
    """
    Оценивает модель эластичности на hold-out выборке.

    Методы
    ------
    temporal_split_eval  — однократный временной сплит
    walk_forward_eval    — скользящая оценка (walk-forward validation)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = 'GMV_7D',
        date_col: str = 'DATE_',
        test_fraction: float = 0.2
    ):
        """
        Параметры
        ---------
        df : pd.DataFrame
            Данные после ETL (содержат date_col и target_col).
        target_col : str
            Название целевой колонки (GMV_7D по умолчанию).
        date_col : str
            Название колонки с датой.
        test_fraction : float
            Доля данных для тестовой выборки (0 < τ < 1).
        """
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction должен быть в интервале (0, 1)")

        self.df           = df.copy()
        self.target_col   = target_col
        self.date_col     = date_col
        self.test_fraction = test_fraction

        # Метаданные последней оценки
        self.metrics_: Optional[EvalMetrics] = None
        self.cv_metrics_: list[EvalMetrics]  = []

    # ─────────────────────────────────────────────────────────────────────
    # Временной сплит (однократный)
    # ─────────────────────────────────────────────────────────────────────

    def temporal_split_eval(self) -> EvalMetrics:
        """
        Однократный временной hold-out.

        train: первые (1-τ)·T наблюдений по времени
        test:  последние τ·T наблюдений

        ВАЖНО: сортировка по дате обязательна. Случайный сплит
        нарушает временную причинность и приводит к data leakage.

        Возвращает
        ----------
        EvalMetrics
        """
        df_sorted = self.df.sort_values(self.date_col).reset_index(drop=True)
        n = len(df_sorted)
        split_idx = int(n * (1.0 - self.test_fraction))

        if split_idx < 10:
            raise ValueError(
                f"Обучающая выборка слишком мала ({split_idx} строк). "
                "Уменьшите test_fraction или добавьте данных."
            )

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df  = df_sorted.iloc[split_idx:].copy()

        train_X, train_y = self._prepare_Xy(train_df)
        test_X,  test_y  = self._prepare_Xy(test_df)

        model = self._build_model(train_X, train_y)
        model.fit()

        test_pred = self._predict_target(model, test_X)

        ped = compute_point_elasticity(model, test_X)

        metrics = EvalMetrics(
            r2      = _r2(test_y.values, test_pred),
            mae     = _mae(test_y.values, test_pred),
            rmse    = _rmse(test_y.values, test_pred),
            mape    = _mape(test_y.values, test_pred),
            n_train = len(train_df),
            n_test  = len(test_df),
            mean_ped = float(np.nanmean(ped)),
            ped_by_price_band = summarize_elasticity_by_price_band(test_X, ped),
        )

        self.metrics_ = metrics
        return metrics

    # ─────────────────────────────────────────────────────────────────────
    # Walk-forward validation (скользящая оценка)
    # ─────────────────────────────────────────────────────────────────────

    def walk_forward_eval(self, n_splits: int = 5) -> pd.DataFrame:
        """
        Walk-forward (expanding window) кросс-валидация.

        На каждом фолде k:
          - train: все наблюдения до точки разбивки k
          - test:  следующие n_test наблюдений

        Это единственная корректная форма CV для временных рядов.
        sklearn.TimeSeriesSplit реализует именно такую схему.

        Параметры
        ---------
        n_splits : int
            Количество фолдов.

        Возвращает
        ----------
        pd.DataFrame
            Метрики по каждому фолду + агрегаты (mean ± std).
        """
        df_sorted = self.df.sort_values(self.date_col).reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        records = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_sorted)):
            train_df = df_sorted.iloc[train_idx].copy()
            test_df  = df_sorted.iloc[test_idx].copy()

            if len(train_df) < 10 or len(test_df) < 2:
                warnings.warn(f"Фолд {fold_idx}: пропускаем — недостаточно данных.")
                continue

            train_X, train_y = self._prepare_Xy(train_df)
            test_X,  test_y  = self._prepare_Xy(test_df)

            try:
                model = self._build_model(train_X, train_y)
                model.fit()
                test_pred = self._predict_target(model, test_X)
                ped = compute_point_elasticity(model, test_X)

                m = EvalMetrics(
                    r2      = _r2(test_y.values, test_pred),
                    mae     = _mae(test_y.values, test_pred),
                    rmse    = _rmse(test_y.values, test_pred),
                    mape    = _mape(test_y.values, test_pred),
                    n_train = len(train_df),
                    n_test  = len(test_df),
                    mean_ped = float(np.nanmean(ped)),
                    ped_by_price_band = summarize_elasticity_by_price_band(test_X, ped),
                )
                records.append({'fold': fold_idx, **m.to_dict()})
            except Exception as exc:
                warnings.warn(f"Фолд {fold_idx}: ошибка — {exc}")

        if not records:
            raise RuntimeError("Ни один фолд не завершился успешно.")

        result_df = pd.DataFrame(records)

        # Агрегаты
        numeric_cols = [c for c in result_df.columns if c != 'fold']
        agg_mean = result_df[numeric_cols].mean()
        agg_std  = result_df[numeric_cols].std()
        agg_mean['fold'] = 'mean'
        agg_std['fold']  = 'std'

        result_df = pd.concat(
            [result_df,
             agg_mean.to_frame().T,
             agg_std.to_frame().T],
            ignore_index=True
        )

        self.cv_metrics_ = records
        return result_df

    # ─────────────────────────────────────────────────────────────────────
    # Вспомогательный метод
    # ─────────────────────────────────────────────────────────────────────

    def _prepare_Xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Разделяет DataFrame на матрицу признаков X и вектор целевой переменной y.
        Удаляет строки с NaN в любой из колонок.
        """
        drop_cols = [
            self.date_col, self.target_col,
            'CATEGORY1', 'CATEGORY2',
            'GMV_1D', 'GMV_7D', 'GMV_15D', 'GMV_30D',
            'AMOUNT_0D_target', 'AMOUNT_1D_target', 'AMOUNT_7D_target',
            'AMOUNT_15D_target', 'AMOUNT_30D_target',
            'AMOUNT_1D', 'AMOUNT_7D', 'AMOUNT_15D', 'AMOUNT_30D',
        ]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[self.target_col].copy().fillna(0)

        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    def _build_model(self, train_X: pd.DataFrame, train_y: pd.Series):
        return DemandModel(df=train_X, target=train_y)

    def _predict_target(self, model, X: pd.DataFrame) -> np.ndarray:
        pred = model.predict(X.copy())
        if isinstance(model, DemandModel):
            return pred['pred_Q'].values
        return pred['pred_GMV'].values
