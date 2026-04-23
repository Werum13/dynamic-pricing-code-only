"""
PriceOptimizer.py
=================
Явная оптимизация цены через максимизацию прибыли / выручки.

НИКАКИХ нейросетей для выбора цены. Только:
  1. Сетка ценовых кандидатов → вычисляем π(P) для каждого → argmax
  2. Уточнение через scipy.optimize.minimize_scalar (опционально)

Задача оптимизации
------------------
Максимизация прибыли с ограничением склада:

    max  π(P)  = (P − MC) · min(Q̂(P), S)
     P
    s.t. P ∈ [P_min, P_max]
         Q̂(P) ≥ 0

где:
  Q̂(P)  — предсказание модели спроса
  MC    — себестоимость (marginal cost)
  S     — остаток на складе

Ограничение склада: фактический спрос = min(Q̂(P), S).
При Q̂(P) > S дополнительная выручка от роста спроса не реализуется.

Максимизация выручки (альтернатива):
    max  R(P)  = P · min(Q̂(P), S)
     P

Расчёт эластичности в точке оптимума
--------------------------------------
После нахождения P* вычисляем ε(P*) через численную производную модели.
Это позволяет интерпретировать решение экономически.

Предупреждения
--------------
- Деление на маленькие цены (P → 0): использовать P_min > 0.
- Шум в Q̂(P) может создавать ложные локальные максимумы:
  grid search находит глобальный, scipy уточняет локально.
- Монотонность: если Q̂(P) немонотонна, π(P) может быть многомодальной.
  Рекомендуется вызвать DemandModel.enforce_monotonicity() перед оптимизацией.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

_EPS = 1e-8


@dataclass
class OptimizationResult:
    """Результат оптимизации для одной позиции (Id)."""
    item_id: int

    optimal_price:    float = 0.0
    predicted_demand: float = 0.0
    predicted_demand_lower: float = 0.0
    predicted_demand_upper: float = 0.0
    realized_demand:  float = 0.0   # min(Q̂, S)
    realized_demand_lower: float = 0.0
    realized_demand_upper: float = 0.0
    revenue:          float = 0.0   # P * realized_demand
    revenue_lower:    float = 0.0
    revenue_upper:    float = 0.0
    profit:           float = 0.0   # (P - MC) * realized_demand
    profit_lower:     float = 0.0
    profit_upper:     float = 0.0
    margin_pct:       float = 0.0   # (P - MC) / P
    elasticity:       float = float('nan')
    objective_value:  float = 0.0
    penalty_value:    float = 0.0

    stock_binding: bool = False   # True если S < Q̂ (склад ограничивает)

    # Кривые для визуализации (опционально)
    price_grid:  np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    profit_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    demand_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    demand_lower_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    demand_upper_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    gmv_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    gmv_lower_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    gmv_upper_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    elasticity_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    realized_demand_grid: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def summary(self) -> dict:
        return {
            'item_id':         self.item_id,
            'optimal_price':   round(self.optimal_price, 4),
            'realized_demand': round(self.realized_demand, 2),
            'revenue':         round(self.revenue, 2),
            'profit':          round(self.profit, 2),
            'margin_pct':      round(self.margin_pct * 100, 2),
            'elasticity':      round(self.elasticity, 4),
            'penalty_value':   round(self.penalty_value, 2),
            'stock_binding':   self.stock_binding,
        }


class PriceOptimizer:
    """
    Оптимизатор цены через явную оптимизацию.

    Алгоритм
    --------
    1. Для каждой позиции (Id) строим сетку из n_grid цен в диапазоне
       [price_min_factor · P_base, price_max_factor · P_base].
    2. Вычисляем Q̂(P) для каждой цены через DemandModel.
    3. Вычисляем π(P) = (P − MC) · min(Q̂(P), S).
    4. Находим P* = argmax π(P) по сетке.
    5. (Опционально) Уточняем P* через scipy.minimize_scalar на интервале
       [P*_grid_left, P*_grid_right].

    Parameters
    ----------
    demand_model : DemandModel
        Обученная модель спроса.
    objective : 'profit' | 'revenue'
        Целевая функция.
        'profit'  → max (P − MC) · Q_realized
        'revenue' → max P · Q_realized
    n_grid : int
        Количество точек в ценовой сетке.
    price_min_factor : float
        Нижняя граница цены как доля от базовой цены (например 0.5 = −50%).
    price_max_factor : float
        Верхняя граница цены как доля от базовой цены (например 2.0 = +100%).
    refine_with_scipy : bool
        Уточнять ли найденный максимум через minimize_scalar.
    enforce_monotonicity : bool
        Применять ли монотонизацию кривой спроса перед оптимизацией.
    min_margin_pct, margin_penalty_weight : optional
        Мягкий штраф за слишком низкую маржу. При weight=0 поведение не меняется.
    price_deviation_penalty_weight : float
        Мягкий штраф за сильное отклонение от base_price. Не является hard cap.
    """

    def __init__(
        self,
        demand_model,
        objective:             Literal['profit', 'revenue'] = 'profit',
        n_grid:                int   = 200,
        price_min_factor:      float = 0.5,
        price_max_factor:      float = 2.0,
        refine_with_scipy:     bool  = True,
        enforce_monotonicity:  bool  = True,
        min_margin_pct: Optional[float] = None,
        margin_penalty_weight: float = 0.0,
        price_deviation_penalty_weight: float = 0.0,
    ):
        self.demand_model          = demand_model
        self.objective             = objective
        self.n_grid                = n_grid
        self.price_min_factor      = price_min_factor
        self.price_max_factor      = price_max_factor
        self.refine_with_scipy     = refine_with_scipy
        self.enforce_monotonicity  = enforce_monotonicity
        self.min_margin_pct        = min_margin_pct
        self.margin_penalty_weight = margin_penalty_weight
        self.price_deviation_penalty_weight = price_deviation_penalty_weight

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def optimize(
        self,
        data: pd.DataFrame,
        stock: float,
        base_price_col: str = 'UNITPRICE',
        cost_col:       str = 'cost',
    ) -> list[OptimizationResult]:
        """
        Оптимизирует цену для каждой позиции в data.

        Parameters
        ----------
        data : pd.DataFrame
            Признаки, включая UNITPRICE, cost, Id.
        stock : float
            Остаток на складе (одно значение для всего батча или dict {Id: S}).
        base_price_col : str
            Колонка с базовой ценой для определения диапазона поиска.
        cost_col : str
            Колонка с себестоимостью.

        Returns
        -------
        list[OptimizationResult]
        """
        results = []

        for item_id, group in data.groupby('Id'):
            S = stock[item_id] if isinstance(stock, dict) else float(stock)
            result = self._optimize_single(
                item_id     = int(item_id),
                group       = group,
                stock       = S,
                base_price_col = base_price_col,
                cost_col    = cost_col,
            )
            results.append(result)

        return results

    def optimize_single_row(
        self,
        row_features: pd.DataFrame,
        stock: float,
        base_price: Optional[float] = None,
        cost: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Оптимизирует цену для одной позиции (одна строка features).

        Это основной метод для использования в pipeline.simulation().

        Parameters
        ----------
        row_features : pd.DataFrame
            Строка-шаблон признаков (цена в ней будет заменяться).
        stock : float
            Остаток на складе.
        base_price : float, optional
            Базовая цена для определения диапазона. Если None — берётся
            из 'UNITPRICE' в row_features.
        cost : float, optional
            Себестоимость. Если None — берётся из 'cost' в row_features.

        Returns
        -------
        OptimizationResult
        """
        if base_price is None:
            base_price = float(row_features['UNITPRICE'].iloc[0])
        if cost is None:
            cost = float(row_features['cost'].iloc[0]) if 'cost' in row_features.columns else 0.0

        item_id = int(row_features['Id'].iloc[0]) if 'Id' in row_features.columns else 0

        # Строим ценовую сетку
        P_min = base_price * self.price_min_factor
        P_max = base_price * self.price_max_factor
        P_min = max(P_min, cost * 1.001, _EPS)  # цена > MC (иначе убыток всегда)

        price_grid = np.linspace(P_min, P_max, self.n_grid)

        # Вычисляем Q для каждой цены
        pred_grid = self._predict_grid(row_features, price_grid)
        curve = self._prepare_curve(price_grid, pred_grid)
        Q_grid = curve['pred_Q']

        # Применяем ограничение склада
        Q_realized = np.minimum(Q_grid, stock)

        # Вычисляем целевую функцию
        obj_grid = self._objective_values(price_grid, Q_realized, cost, base_price)

        # Шаг 1: argmax по сетке
        max_obj = float(np.nanmax(obj_grid))
        near_opt_tol = max(1e-8, 0.005 * abs(max_obj))
        candidate_idx = np.where(obj_grid >= max_obj - near_opt_tol)[0]
        best_idx = int(candidate_idx[0]) if len(candidate_idx) > 0 else int(np.argmax(obj_grid))
        P_star   = price_grid[best_idx]

        # Шаг 2: уточнение через scipy (если включено и есть простор)
        if (
            self.refine_with_scipy and
            len(candidate_idx) <= 1 and
            best_idx > 0 and
            best_idx < len(price_grid) - 1
        ):
            P_star = self._refine_scipy(
                price_grid   = price_grid,
                demand_grid  = Q_grid,
                P_lo         = price_grid[best_idx - 1],
                P_hi         = price_grid[best_idx + 1],
                stock        = stock,
                cost         = cost,
                base_price   = base_price,
            )

        # Финальный расчёт в оптимальной точке
        Q_star      = float(np.interp(P_star, price_grid, curve['pred_Q']))
        Q_star_low  = float(np.interp(P_star, price_grid, curve['pred_Q_lower']))
        Q_star_high = float(np.interp(P_star, price_grid, curve['pred_Q_upper']))
        Q_real_star = min(Q_star, stock)
        Q_real_low  = min(Q_star_low, stock)
        Q_real_high = min(Q_star_high, stock)
        revenue     = P_star * Q_real_star
        revenue_low = P_star * Q_real_low
        revenue_high = P_star * Q_real_high
        profit      = (P_star - cost) * Q_real_star
        profit_low  = (P_star - cost) * Q_real_low
        profit_high = (P_star - cost) * Q_real_high
        margin_pct  = (P_star - cost) / max(P_star, _EPS)
        penalty_val = float(self._soft_penalty(P_star, Q_real_star, cost, base_price))
        objective_value = float(self._objective_values(
            np.array([P_star]),
            np.array([Q_real_star]),
            cost,
            base_price,
        )[0])

        # Эластичность в точке оптимума
        eps_val = float(np.interp(P_star, price_grid, curve['elasticity']))

        return OptimizationResult(
            item_id          = item_id,
            optimal_price    = P_star,
            predicted_demand = Q_star,
            predicted_demand_lower = Q_star_low,
            predicted_demand_upper = Q_star_high,
            realized_demand  = Q_real_star,
            realized_demand_lower = Q_real_low,
            realized_demand_upper = Q_real_high,
            revenue          = revenue,
            revenue_lower    = revenue_low,
            revenue_upper    = revenue_high,
            profit           = profit,
            profit_lower     = profit_low,
            profit_upper     = profit_high,
            margin_pct       = margin_pct,
            elasticity       = eps_val,
            objective_value  = objective_value,
            penalty_value    = penalty_val,
            stock_binding    = Q_star > stock,
            price_grid       = price_grid,
            profit_grid      = obj_grid,
            demand_grid      = Q_grid,
            demand_lower_grid = curve['pred_Q_lower'],
            demand_upper_grid = curve['pred_Q_upper'],
            gmv_grid         = price_grid * Q_grid,
            gmv_lower_grid   = price_grid * curve['pred_Q_lower'],
            gmv_upper_grid   = price_grid * curve['pred_Q_upper'],
            elasticity_grid  = curve['elasticity'],
            realized_demand_grid = Q_realized,
        )

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _optimize_single(
        self,
        item_id: int,
        group: pd.DataFrame,
        stock: float,
        base_price_col: str,
        cost_col: str,
    ) -> OptimizationResult:
        """Оптимизация для одной группы (одного Id)."""
        base_price = float(group[base_price_col].median())
        cost       = float(group[cost_col].iloc[0]) if cost_col in group.columns else 0.0

        # Берём строку-шаблон (первую строку с медианными признаками)
        template = group.iloc[[0]].copy()
        template['UNITPRICE'] = base_price

        return self.optimize_single_row(
            row_features = template,
            stock        = stock,
            base_price   = base_price,
            cost         = cost,
        )

    def _predict_grid(
        self,
        template: pd.DataFrame,
        price_grid: np.ndarray,
    ) -> pd.DataFrame:
        """
        Предсказывает Q для массива цен, копируя template для каждой цены.

        Это быстрее, чем вызывать predict() n_grid раз по одной строке.
        """
        rows = []
        for P in price_grid:
            row = template.copy()
            row['UNITPRICE'] = P
            _refresh_price_features_inplace(row)
            rows.append(row)

        batch = pd.concat(rows, ignore_index=True)
        
        # Ensure batch has all required columns before prediction
        if hasattr(self.demand_model, 'columns_') and self.demand_model.columns_ is not None:
            missing_cols = set(self.demand_model.columns_) - set(batch.columns)
            if missing_cols:
                # Add missing columns with NaN or 0
                for col in missing_cols:
                    batch[col] = 0
        
        pred = self.demand_model.predict(batch)

        if 'pred_Q' not in pred.columns:
            raise ValueError(f"Model prediction missing 'pred_Q' column. Available columns: {pred.columns.tolist()}")

        return pred

    def _batch_predict_Q(
        self,
        template: pd.DataFrame,
        price_grid: np.ndarray,
    ) -> np.ndarray:
        return self._predict_grid(template, price_grid)['pred_Q'].values

    def _predict_Q_at_price(
        self,
        template: pd.DataFrame,
        price: float,
    ) -> float:
        return float(self._predict_at_price(template, price)['pred_Q'].iloc[0])

    def _predict_at_price(
        self,
        template: pd.DataFrame,
        price: float,
    ) -> pd.DataFrame:
        row = _set_price(template, price)
        return self.demand_model.predict(row)

    def _objective_values(
        self,
        prices:     np.ndarray,
        Q_realized: np.ndarray,
        cost:       float,
        base_price: float,
    ) -> np.ndarray:
        """Вычисляет целевую функцию для всей сетки цен."""
        if self.objective == 'profit':
            base_obj = (prices - cost) * Q_realized
        elif self.objective == 'revenue':
            base_obj = prices * Q_realized
        else:
            raise ValueError(f"Неизвестный objective: {self.objective}")
        penalties = self._soft_penalty(prices, Q_realized, cost, base_price)
        return base_obj - penalties

    def _objective_scalar(
        self,
        price: float,
        price_grid: np.ndarray,
        demand_grid: np.ndarray,
        stock: float,
        cost: float,
        base_price: float,
    ) -> float:
        """Скалярная целевая функция для scipy (возвращает -π, т.к. минимизация)."""
        Q = float(np.interp(price, price_grid, demand_grid))
        Q_real = min(Q, stock)
        return -float(self._objective_values(
            np.array([price]),
            np.array([Q_real]),
            cost,
            base_price,
        )[0])

    def _refine_scipy(
        self,
        price_grid: np.ndarray,
        demand_grid: np.ndarray,
        P_lo: float,
        P_hi: float,
        stock: float,
        cost: float,
        base_price: float,
    ) -> float:
        """
        Уточняет максимум на интервале [P_lo, P_hi] через golden-section search.

        scipy.optimize.minimize_scalar(method='bounded') — O(log(1/tol)) вызовов
        целевой функции. Быстро и не требует производной.
        """
        result = minimize_scalar(
            fun    = self._objective_scalar,
            bounds = (P_lo, P_hi),
            method = 'bounded',
            args   = (price_grid, demand_grid, stock, cost, base_price),
            options = {'xatol': 1e-4, 'maxiter': 50}
        )
        return float(result.x) if result.success else (P_lo + P_hi) / 2.0

    def _prepare_curve(
        self,
        price_grid: np.ndarray,
        pred_grid: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        q_raw = pred_grid['pred_Q'].to_numpy(dtype=float)
        q_low_raw = pred_grid.get('pred_Q_lower', pred_grid['pred_Q']).to_numpy(dtype=float)
        q_high_raw = pred_grid.get('pred_Q_upper', pred_grid['pred_Q']).to_numpy(dtype=float)

        if self.enforce_monotonicity:
            price_grid, q_raw, q_low_raw, q_high_raw = self.demand_model.monotonicize_curve(
                prices=price_grid,
                demand=q_raw,
                lower=q_low_raw,
                upper=q_high_raw,
            )

        elasticity = self._elasticity_from_curve(price_grid, q_raw)
        return {
            'pred_Q': np.clip(q_raw, 0, np.inf),
            'pred_Q_lower': np.clip(q_low_raw, 0, np.inf),
            'pred_Q_upper': np.clip(q_high_raw, 0, np.inf),
            'elasticity': elasticity,
        }

    @staticmethod
    def _elasticity_from_curve(
        price_grid: np.ndarray,
        demand_grid: np.ndarray,
    ) -> np.ndarray:
        if len(price_grid) < 2:
            return np.full(len(price_grid), np.nan)
        dQ_dP = np.gradient(demand_grid, price_grid)
        return dQ_dP * price_grid / np.maximum(demand_grid, _EPS)

    def _soft_penalty(
        self,
        prices,
        Q_realized,
        cost: float,
        base_price: float,
    ) -> np.ndarray:
        prices_arr = np.asarray(prices, dtype=float)
        qty_arr = np.asarray(Q_realized, dtype=float)
        penalties = np.zeros_like(prices_arr, dtype=float)

        if self.min_margin_pct is not None and self.margin_penalty_weight > 0:
            margin_pct = (prices_arr - cost) / np.maximum(prices_arr, _EPS)
            margin_gap = np.maximum(self.min_margin_pct - margin_pct, 0.0)
            penalties += (
                self.margin_penalty_weight *
                np.maximum(prices_arr, _EPS) *
                np.maximum(qty_arr, 0.0) *
                margin_gap ** 2
            )

        if base_price > 0 and self.price_deviation_penalty_weight > 0:
            deviation = prices_arr / max(base_price, _EPS) - 1.0
            penalties += (
                self.price_deviation_penalty_weight *
                base_price *
                np.maximum(qty_arr, 0.0) *
                deviation ** 2
            )

        return penalties


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def _set_price(df: pd.DataFrame, price: float) -> pd.DataFrame:
    """Клонирует DataFrame и задаёт новую цену, обновляет производные признаки."""
    df = df.copy()
    df['UNITPRICE'] = price
    _refresh_price_features_inplace(df)
    return df


def _refresh_price_features_inplace(df: pd.DataFrame) -> None:
    """Пересчитывает discount и margin_% в месте (inplace)."""
    if 'BASEPRICE' in df.columns:
        df['discount'] = 1.0 - df['UNITPRICE'] / df['BASEPRICE'].replace(0, _EPS)
    if 'cost' in df.columns:
        df['margin_%'] = (df['UNITPRICE'] - df['cost']) / df['cost'].replace(0, _EPS)


# ------------------------------------------------------------------
# Удобная функция-фасад (совместимость с pipeline.py)
# ------------------------------------------------------------------

def price_optimizer(
    demand_model,
    row_features:  pd.DataFrame,
    stock:         float,
    cost:          Optional[float] = None,
    base_price:    Optional[float] = None,
    objective:     str = 'profit',
    n_grid:        int = 300,
    min_margin_pct: Optional[float] = None,
    margin_penalty_weight: float = 0.0,
    price_deviation_penalty_weight: float = 0.0,
) -> dict:
    """
    Оптимизирует цену для одной строки признаков.

    Заменяет вызов optimizer(pred, warehouse_record['available'])
    из старого pipeline.py.

    Returns
    -------
    dict с ключами: unitprice, quantity, gmv, margin, margin_percent, elasticity.
    """
    opt = PriceOptimizer(
        demand_model      = demand_model,
        objective         = objective,
        n_grid            = n_grid,
        price_min_factor  = 0.5,
        price_max_factor  = 2.0,
        refine_with_scipy = True,
        min_margin_pct    = min_margin_pct,
        margin_penalty_weight = margin_penalty_weight,
        price_deviation_penalty_weight = price_deviation_penalty_weight,
    )

    result = opt.optimize_single_row(
        row_features = row_features,
        stock        = stock,
        base_price   = base_price,
        cost         = cost,
    )

    P = result.optimal_price
    Q = result.realized_demand
    return {
        'unitprice':      round(P, 4),
        'quantity':       round(Q, 2),
        'quantity_lower': round(result.realized_demand_lower, 2),
        'quantity_upper': round(result.realized_demand_upper, 2),
        'gmv':            round(result.revenue, 2),
        'gmv_lower':      round(result.revenue_lower, 2),
        'gmv_upper':      round(result.revenue_upper, 2),
        'margin':         round(result.profit, 2),
        'margin_lower':   round(result.profit_lower, 2),
        'margin_upper':   round(result.profit_upper, 2),
        'margin_percent': round(result.margin_pct, 4),
        'elasticity':     round(result.elasticity, 4),
        'penalty_value':  round(result.penalty_value, 2),
        'stock_binding':  result.stock_binding,
        '_result_obj':    result,   # полный объект для визуализации
    }
