"""
Baseline.py
===========
Базовые стратегии ценообразования и управления запасами.

Назначение
----------
Любая ML-система требует сравнения с простыми эвристиками.
Если оптимизатор не превосходит cost+markup на ±5%, это сигнал
пересмотреть модель, а не радоваться красивым графикам.

Реализованные стратегии
-----------------------
1. FixedPriceBaseline    — фиксированная медианная цена за весь период
2. CostPlusBaseline      — cost × (1 + markup)
3. HeuristicReorderBaseline — классический ROP + EOQ из теории запасов

Метрики сравнения (SimResult)
-------------------------------
Для честного сравнения с оптимизатором все стратегии возвращают
одинаковую структуру SimResult с полями:

  total_gmv       — суммарная выручка
  total_margin    — суммарная маржа (GMV − cost·Q)
  avg_margin_pct  — средний процент маржи
  fill_rate       — доля спроса, удовлетворённого со склада
                    fill_rate = Σ sale_real / Σ demand
  stockout_days   — количество дней дефицита (rest = 0 и demand > 0)
  avg_inventory   — средний остаток на складе

Теоретические основы
---------------------
ROP (Reorder Point):
    ROP = μ_d · L + z · σ_d · √L

    μ_d  — средний дневной спрос (скользящее окно)
    σ_d  — стандартное отклонение спроса
    L    — lead time (дней)
    z    — z-квантиль нормального распределения (уровень сервиса)
           z = 1.645 → service level = 95%

EOQ (Economic Order Quantity):
    EOQ = √(2 · D · K / h)

    D — годовой спрос
    K — стоимость размещения заказа (ordering cost)
    h — стоимость хранения единицы в год (holding cost)

При отсутствии параметров K и h используется упрощённая эвристика:
    order = max(μ_d_7 + z·σ_d·√L − rest, 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Контейнер результатов симуляции
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Результаты симуляции одной стратегии за период."""
    strategy_name: str

    total_gmv:      float = 0.0
    total_margin:   float = 0.0
    avg_margin_pct: float = 0.0
    fill_rate:      float = 0.0   # ∈ [0, 1]
    stockout_days:  int   = 0
    avg_inventory:  float = 0.0
    n_orders:       int   = 0

    daily_log: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def summary(self) -> pd.Series:
        return pd.Series({
            'strategy':       self.strategy_name,
            'total_gmv':      round(self.total_gmv, 2),
            'total_margin':   round(self.total_margin, 2),
            'avg_margin_pct': round(self.avg_margin_pct * 100, 2),
            'fill_rate_%':    round(self.fill_rate * 100, 2),
            'stockout_days':  self.stockout_days,
            'avg_inventory':  round(self.avg_inventory, 2),
            'n_orders':       self.n_orders,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный симулятор склада (общий для всех стратегий)
# ─────────────────────────────────────────────────────────────────────────────

class _WarehouseSimulator:
    """
    Простой дискретный симулятор склада (unit: единица товара).

    На каждый день:
      available = rest_prev + order_t
      sale      = min(available, demand_t)
      rest_next = available - sale
    """

    def __init__(self, initial_stock: float = 0.0, lead_time: int = 1):
        self.rest      = initial_stock
        self.lead_time = lead_time
        self.log: list = []

    def step(
        self,
        day: pd.Timestamp,
        price: float,
        demand: float,
        cost: float,
        order: float
    ) -> dict:
        available = self.rest + order
        sale      = min(available, demand)
        self.rest = available - sale
        margin    = (price - cost) * sale

        record = {
            'ds':        day,
            'price':     price,
            'demand':    demand,
            'order':     order,
            'available': available,
            'sale':      sale,
            'rest':      self.rest,
            'gmv':       price * sale,
            'margin':    margin,
            'margin_%':  margin / max(price * sale, 1e-8),
        }
        self.log.append(record)
        return record

    def get_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    @staticmethod
    def aggregate(log: pd.DataFrame, strategy_name: str) -> SimResult:
        total_demand = log['demand'].sum()
        total_sale   = log['sale'].sum()

        return SimResult(
            strategy_name = strategy_name,
            total_gmv      = log['gmv'].sum(),
            total_margin   = log['margin'].sum(),
            avg_margin_pct = log['margin_%'].mean(),
            fill_rate      = total_sale / max(total_demand, 1e-8),
            stockout_days  = int(((log['rest'] == 0) & (log['demand'] > 0)).sum()),
            avg_inventory  = log['rest'].mean(),
            n_orders       = int((log['order'] > 0).sum()),
            daily_log      = log,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка дневных данных для сравнения стратегий
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_daily_simulation_input(
    df: pd.DataFrame,
    demand_col: str = 'AMOUNT',
    price_col: str = 'UNITPRICE',
    cost_col: str = 'cost',
    date_col: str = 'DATE_',
) -> pd.DataFrame:
    """
    Приводит транзакционные данные к дневному уровню.

    compare_baselines и стратегии ниже ожидают одну строку на день.
    В приложении сюда часто попадают сырые продажи по чекам, поэтому
    предварительно агрегируем период на дневной уровень.
    """
    if len(df) == 0:
        return df.copy()

    daily = df.copy()
    daily[date_col] = pd.to_datetime(daily[date_col]).dt.normalize()

    agg_map = {}
    if demand_col in daily.columns:
        agg_map[demand_col] = 'sum'
    if 'TOTALPRICE' in daily.columns:
        agg_map['TOTALPRICE'] = 'sum'
    if price_col in daily.columns:
        agg_map[price_col] = 'median'
    if 'BASEPRICE' in daily.columns:
        agg_map['BASEPRICE'] = 'median'
    if cost_col in daily.columns:
        agg_map[cost_col] = 'median'

    if not agg_map:
        return (
            daily[[date_col]]
            .drop_duplicates()
            .sort_values(date_col)
            .reset_index(drop=True)
        )

    return (
        daily
        .groupby(date_col, as_index=False)
        .agg(agg_map)
        .sort_values(date_col)
        .reset_index(drop=True)
    )


def _prepare_warmup_input(
    history_df: Optional[pd.DataFrame],
    sim_start: pd.Timestamp,
    demand_col: str,
    price_col: str,
    cost_col: str,
    date_col: str,
) -> pd.DataFrame:
    if history_df is None or len(history_df) == 0:
        return pd.DataFrame()

    warmup = history_df.copy()
    warmup[date_col] = pd.to_datetime(warmup[date_col]).dt.normalize()
    warmup = warmup[warmup[date_col] < pd.to_datetime(sim_start).normalize()]
    if len(warmup) == 0:
        return pd.DataFrame()

    return _prepare_daily_simulation_input(
        df=warmup,
        demand_col=demand_col,
        price_col=price_col,
        cost_col=cost_col,
        date_col=date_col,
    )


def _bootstrap_state(
    initial_stock: float,
    warmup_df: Optional[pd.DataFrame],
    demand_col: str,
    z: float,
    lead_time: int,
    demand_window: int,
    cover_days: int,
) -> tuple[float, list[float]]:
    demand_history: list[float] = []
    stock = float(initial_stock)

    if warmup_df is None or len(warmup_df) == 0 or demand_col not in warmup_df.columns:
        return stock, demand_history

    history = (
        warmup_df[demand_col]
        .fillna(0.0)
        .astype(float)
        .tolist()
    )
    demand_history = history[-max(int(demand_window), 1):]

    if stock <= 0 and demand_history:
        window = np.asarray(demand_history, dtype=float)
        mu_d = float(window.mean()) if len(window) else 0.0
        sigma_d = float(window.std()) if len(window) > 1 else 0.0
        safety = z * sigma_d * np.sqrt(max(lead_time, 1))
        stock = max(mu_d * max(int(cover_days), 1) + safety, 0.0)

    return stock, demand_history


# ─────────────────────────────────────────────────────────────────────────────
# 1. Фиксированная цена (Fixed Price)
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalPriceBaseline:
    """
    Стратегия 0: Исходная цена из файла (BASEPRICE) + исторический спрос.

    BASEPRICE — это оригинальная UNITPRICE до пересчёта в DataPreprocessor
    (до того как UNITPRICE = TOTALPRICE / AMOUNT).

    Это «нулевой baseline»: что было бы, если бы мы просто продавали
    по исходной цене из файла с тем спросом, который реально был.

    Параметры
    ---------
    z, lead_time, reorder_window : аналогичны другим baseline.
    """

    def __init__(
        self,
        z: float = 1.645,
        lead_time: int = 1,
        reorder_window: int = 30
    ):
        self.z              = z
        self.lead_time      = lead_time
        self.reorder_window = reorder_window

    def simulate(
        self,
        df: pd.DataFrame,
        demand_col: str = 'AMOUNT',
        price_col: str  = 'UNITPRICE',
        cost_col: str   = 'cost',
        date_col: str   = 'DATE_',
        initial_stock: float = 0.0,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> SimResult:
        df = df.sort_values(date_col).reset_index(drop=True)

        # Используем BASEPRICE если есть, иначе UNITPRICE
        use_col = 'BASEPRICE' if 'BASEPRICE' in df.columns else price_col

        seed_stock, demand_history = _bootstrap_state(
            initial_stock=initial_stock,
            warmup_df=warmup_df,
            demand_col=demand_col,
            z=self.z,
            lead_time=self.lead_time,
            demand_window=self.reorder_window,
            cover_days=7,
        )
        sim = _WarehouseSimulator(seed_stock, self.lead_time)

        for _, row in df.iterrows():
            day    = row[date_col]
            price  = float(row[use_col]) if not pd.isna(row.get(use_col)) else float(row.get(price_col, 0))
            demand = float(row[demand_col]) if not pd.isna(row[demand_col]) else 0.0
            cost   = float(row[cost_col])   if cost_col in row else 0.0

            window = demand_history[-self.reorder_window:]
            mu_d   = np.mean(window) if window else 0.0
            sigma_d = np.std(window) if len(window) > 1 else 0.0
            safety  = self.z * sigma_d * np.sqrt(self.lead_time)
            order   = max(mu_d * 7 + safety - sim.rest, 0.0)

            sim.step(day, price, demand, cost, order)
            demand_history.append(demand)

        return _WarehouseSimulator.aggregate(sim.get_log(), 'HistoricalPrice')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Фиксированная цена (Fixed Price)
# ─────────────────────────────────────────────────────────────────────────────

class FixedPriceBaseline:
    """
    Стратегия 1: Фиксированная медианная цена.

    price_fixed = median(UNITPRICE, train_period)

    Порядок:
      - Цена = медиана исторических цен (не среднее: менее чувствительна к выбросам)
      - Заказ = ожидаемый спрос на неделю + safety stock (z·σ·√L)
      - Переобучение прогноза спроса — каждые 7 дней

    Параметры
    ---------
    z : float
        z-квантиль для safety stock (1.645 → 95% service level)
    lead_time : int
        Срок поставки в днях
    reorder_window : int
        Окно скользящего среднего для оценки спроса (дней)
    demand_fn : callable, optional
        Функция(day, price, cost) → demand.
        Если None — используется исторический спрос из df.
        Если задана — спрос предсказывается через модель эластичности.
    """

    def __init__(
        self,
        z: float = 1.645,
        lead_time: int = 1,
        reorder_window: int = 30,
        demand_fn=None
    ):
        self.z              = z
        self.lead_time      = lead_time
        self.reorder_window = reorder_window
        self.demand_fn      = demand_fn

    def simulate(
        self,
        df: pd.DataFrame,
        demand_col: str = 'AMOUNT',
        price_col: str  = 'UNITPRICE',
        cost_col: str   = 'cost',
        date_col: str   = 'DATE_',
        initial_stock: float = 0.0,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> SimResult:
        """
        Запускает симуляцию на переданных данных.

        Параметры
        ---------
        df : pd.DataFrame
            Дневные данные (одна строка = один день).
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        price_fixed = float(df[price_col].median())

        seed_stock, demand_history = _bootstrap_state(
            initial_stock=initial_stock,
            warmup_df=warmup_df,
            demand_col=demand_col,
            z=self.z,
            lead_time=self.lead_time,
            demand_window=self.reorder_window,
            cover_days=7,
        )
        sim = _WarehouseSimulator(seed_stock, self.lead_time)

        for _, row in df.iterrows():
            day    = row[date_col]
            cost   = float(row[cost_col])   if cost_col in row else 0.0

            # Спрос: через модель эластичности или исторический
            if self.demand_fn is not None:
                demand = self.demand_fn(day, price_fixed, cost)
            else:
                demand = float(row[demand_col]) if not pd.isna(row[demand_col]) else 0.0

            # Прогноз и заказ
            window = demand_history[-self.reorder_window:]
            mu_d   = np.mean(window) if window else 0.0
            sigma_d = np.std(window) if len(window) > 1 else 0.0
            safety  = self.z * sigma_d * np.sqrt(self.lead_time)

            order = max(mu_d * 7 + safety - sim.rest, 0.0)

            sim.step(day, price_fixed, demand, cost, order)
            demand_history.append(demand)

        return _WarehouseSimulator.aggregate(sim.get_log(), 'FixedPrice_Median')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cost-plus markup (Наценка на себестоимость)
# ─────────────────────────────────────────────────────────────────────────────

class CostPlusBaseline:
    """
    Стратегия 2: Правило cost-plus markup.

    price_t = cost_t · (1 + markup)

    Это стандартный подход в ритейле: менеджер устанавливает целевую
    маржу и не меняет цену в зависимости от спроса.

    markup = 0.30  → наценка 30% (margin = 23%)
    markup = 0.50  → наценка 50% (margin = 33%)

    Параметры
    ---------
    markup : float
        Процент наценки к себестоимости (0.3 = 30%).
    z, lead_time, reorder_window : аналогичны FixedPriceBaseline.
    """

    def __init__(
        self,
        markup: float = 0.30,
        z: float = 1.645,
        lead_time: int = 1,
        reorder_window: int = 30,
        demand_fn=None
    ):
        self.markup         = markup
        self.z              = z
        self.lead_time      = lead_time
        self.reorder_window = reorder_window
        self.demand_fn      = demand_fn

    def simulate(
        self,
        df: pd.DataFrame,
        demand_col: str = 'AMOUNT',
        price_col: str  = 'UNITPRICE',
        cost_col: str   = 'cost',
        date_col: str   = 'DATE_',
        initial_stock: float = 0.0,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> SimResult:
        df = df.sort_values(date_col).reset_index(drop=True)

        if cost_col not in df.columns:
            raise KeyError(f"Колонка '{cost_col}' не найдена. Cost-plus требует информацию о себестоимости.")

        seed_stock, demand_history = _bootstrap_state(
            initial_stock=initial_stock,
            warmup_df=warmup_df,
            demand_col=demand_col,
            z=self.z,
            lead_time=self.lead_time,
            demand_window=self.reorder_window,
            cover_days=7,
        )
        sim = _WarehouseSimulator(seed_stock, self.lead_time)

        for _, row in df.iterrows():
            day    = row[date_col]
            cost   = float(row[cost_col])
            price  = cost * (1.0 + self.markup)

            # Спрос: через модель эластичности или исторический
            if self.demand_fn is not None:
                demand = self.demand_fn(day, price, cost)
            else:
                demand = float(row[demand_col]) if not pd.isna(row[demand_col]) else 0.0

            window  = demand_history[-self.reorder_window:]
            mu_d    = np.mean(window) if window else 0.0
            sigma_d = np.std(window) if len(window) > 1 else 0.0
            safety  = self.z * sigma_d * np.sqrt(self.lead_time)
            order   = max(mu_d * 7 + safety - sim.rest, 0.0)

            sim.step(day, price, demand, cost, order)
            demand_history.append(demand)

        return _WarehouseSimulator.aggregate(
            sim.get_log(), f'CostPlus_markup={self.markup:.0%}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Heuristic Reorder Point (Классический ROP)
# ─────────────────────────────────────────────────────────────────────────────

class HeuristicReorderBaseline:
    """
    Стратегия 3: Классический ROP (Reorder Point) из теории запасов.

    ROP = μ_d · L + z · σ_d · √L

    Где:
      μ_d  = (1/W) Σ_{t-W}^{t-1} demand_t  — скользящее среднее спроса
      σ_d  = std(demand_{t-W:t-1})           — стандартное отклонение
      L    = lead_time (дней)
      z    = z-квантиль (уровень сервиса)

    Логика заказа:
      Если rest ≤ ROP → размещаем заказ Q*
      Иначе → не заказываем

    Q* (упрощённый EOQ без параметров K и h):
      Q* = μ_d · review_period + safety_stock − rest
         = μ_d · 7 + z · σ_d · √L − rest

    Цена фиксирована (медиана) — изолируем эффект управления запасами.

    Параметры
    ---------
    z : float
        z-квантиль нормального распределения.
        1.282 → 90%, 1.645 → 95%, 2.326 → 99%.
    lead_time : int
        Срок поставки (дней).
    review_period : int
        Период пересмотра заказа (дней). По умолчанию 7 (еженедельно).
    demand_window : int
        Окно для оценки μ_d и σ_d (дней).
    """

    def __init__(
        self,
        z: float = 1.645,
        lead_time: int = 1,
        review_period: int = 7,
        demand_window: int = 30,
        demand_fn=None
    ):
        self.z             = z
        self.lead_time     = lead_time
        self.review_period = review_period
        self.demand_window = demand_window
        self.demand_fn     = demand_fn

    def simulate(
        self,
        df: pd.DataFrame,
        demand_col: str = 'AMOUNT',
        price_col: str  = 'UNITPRICE',
        cost_col: str   = 'cost',
        date_col: str   = 'DATE_',
        initial_stock: float = 0.0,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> SimResult:
        df = df.sort_values(date_col).reset_index(drop=True)
        price_fixed = float(df[price_col].median())

        seed_stock, demand_history = _bootstrap_state(
            initial_stock=initial_stock,
            warmup_df=warmup_df,
            demand_col=demand_col,
            z=self.z,
            lead_time=self.lead_time,
            demand_window=self.demand_window,
            cover_days=self.review_period,
        )
        sim = _WarehouseSimulator(seed_stock, self.lead_time)

        for i, row in df.iterrows():
            day    = row[date_col]
            cost   = float(row[cost_col]) if cost_col in row else 0.0

            # Спрос: через модель эластичности или исторический
            if self.demand_fn is not None:
                demand = self.demand_fn(day, price_fixed, cost)
            else:
                demand = float(row[demand_col]) if not pd.isna(row[demand_col]) else 0.0

            window  = demand_history[-self.demand_window:]
            mu_d    = np.mean(window) if window else 0.0
            sigma_d = np.std(window) if len(window) > 1 else 0.0

            # ROP = μ_d · L + z · σ_d · √L
            rop = mu_d * self.lead_time + self.z * sigma_d * np.sqrt(self.lead_time)

            # Заказываем только если запас упал до или ниже ROP
            # и только в день пересмотра (каждые review_period дней)
            is_review_day = (i % self.review_period == 0)
            order = 0.0
            if sim.rest <= rop and is_review_day:
                # Q* = покрытие review_period + safety stock − текущий остаток
                safety = self.z * sigma_d * np.sqrt(self.lead_time)
                order  = max(
                    mu_d * self.review_period + safety - sim.rest,
                    0.0
                )

            sim.step(day, price_fixed, demand, cost, order)
            demand_history.append(demand)

        return _WarehouseSimulator.aggregate(sim.get_log(), f'HeuristicROP_z={self.z}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. ML Optimizer (пропущен через _WarehouseSimulator)
# ─────────────────────────────────────────────────────────────────────────────

class MLOptimizerBaseline:
    """
    Стратегия 4: ML-оптимизатор цен, пропущенный через реальный склад.

    В отличие от «сырого» ML_Optimizer (который просто суммирует
    pred_GMV без учёта остатков), эта стратегия:

      1. Берёт оптимизированные цены из opt_results_{itemcode}.csv
      2. Пропускает их через _WarehouseSimulator с тем же reorder-механизмом
         что и baseline-стратегии
      3. Возвращает честные fill_rate, stockout_days, avg_inventory

    Это делает сравнение по-настоящему fair: все стратегии работают
    в одинаковых условиях склада.

    Параметры
    ---------
    ml_results : pd.DataFrame
        Результаты ML-оптимизатора (DATE_, unitprice, quantity, margin, ...).
    z, lead_time, reorder_window : аналогичны другим baseline.
    demand_fn : callable, optional
        Функция(day, price, cost) → demand для предсказания спроса через модель.
    """

    def __init__(
        self,
        ml_results: pd.DataFrame,
        z: float = 1.645,
        lead_time: int = 1,
        reorder_window: int = 30,
        demand_fn=None
    ):
        self.ml_results     = ml_results.copy()
        self.z              = z
        self.lead_time      = lead_time
        self.reorder_window = reorder_window
        self.demand_fn      = demand_fn

    def simulate(
        self,
        df: pd.DataFrame,
        demand_col: str = 'AMOUNT',
        price_col: str  = 'UNITPRICE',
        cost_col: str   = 'cost',
        date_col: str   = 'DATE_',
        initial_stock: float = 0.0,
        warmup_df: Optional[pd.DataFrame] = None,
    ) -> SimResult:
        df = df.sort_values(date_col).reset_index(drop=True)

        # Строим маппинг: дата → ML-цена
        ml_prices = {}
        if len(self.ml_results) > 0:
            ml_daily = self.ml_results.copy()
            ml_daily['DATE_'] = pd.to_datetime(ml_daily['DATE_']).dt.normalize()
            ml_daily = (
                ml_daily
                .sort_values('DATE_')
                .groupby('DATE_', as_index=False)
                .agg(unitprice=('unitprice', 'last'), cost=('cost', 'last'))
            )
            for _, row in ml_daily.iterrows():
                day = pd.to_datetime(row['DATE_'])
                ml_prices[day.date()] = {
                    'price': float(row['unitprice']),
                    'cost':  float(row.get('cost', 0)),
                }

        seed_stock, demand_history = _bootstrap_state(
            initial_stock=initial_stock,
            warmup_df=warmup_df,
            demand_col=demand_col,
            z=self.z,
            lead_time=self.lead_time,
            demand_window=self.reorder_window,
            cover_days=7,
        )
        sim = _WarehouseSimulator(seed_stock, self.lead_time)

        for _, row in df.iterrows():
            day    = row[date_col]
            cost   = float(row[cost_col])   if cost_col in row else 0.0

            # Берём ML-цену если есть, иначе fallback на cost × 1.3
            day_date = day.date() if hasattr(day, 'date') else pd.Timestamp(day).date()
            if day_date in ml_prices:
                price = ml_prices[day_date]['price']
                cost  = ml_prices[day_date].get('cost', cost)
            else:
                price = cost * 1.3

            # Спрос: через модель эластичности или исторический
            if self.demand_fn is not None:
                demand = self.demand_fn(day, price, cost)
            else:
                demand = float(row[demand_col]) if not pd.isna(row[demand_col]) else 0.0

            # Прогноз и заказ (тот же механизм что у baseline)
            window  = demand_history[-self.reorder_window:]
            mu_d    = np.mean(window) if window else 0.0
            sigma_d = np.std(window) if len(window) > 1 else 0.0
            safety  = self.z * sigma_d * np.sqrt(self.lead_time)
            order   = max(mu_d * 7 + safety - sim.rest, 0.0)

            sim.step(day, price, demand, cost, order)
            demand_history.append(demand)

        return _WarehouseSimulator.aggregate(sim.get_log(), 'ML_Optimizer')


# ─────────────────────────────────────────────────────────────────────────────
# Сравнительная таблица
# ─────────────────────────────────────────────────────────────────────────────

def compare_baselines(
    df: pd.DataFrame,
    ml_results: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    demand_col: str = 'AMOUNT',
    price_col: str  = 'UNITPRICE',
    cost_col: str   = 'cost',
    date_col: str   = 'DATE_',
    initial_stock: float = 0.0,
    markup: float = 0.30
) -> pd.DataFrame:
    """
    Запускает все baseline-стратегии и собирает сравнительную таблицу.

    Все baseline-стратегии используют исторический спрос из df.
    ML_Optimizer тоже проходит через _WarehouseSimulator с историческим спросом —
    это даёт честное сравнение при одинаковых условиях склада.

    Параметры
    ---------
    df : pd.DataFrame
        Дневные данные за симулируемый период.
    ml_results : pd.DataFrame, optional
        Результаты ML-оптимизатора (для MLOptimizerBaseline).
    markup : float
        Наценка для CostPlusBaseline.

    Возвращает
    ----------
    pd.DataFrame
        Сводная таблица со строкой на каждую стратегию.
    """
    df = _prepare_daily_simulation_input(
        df=df.copy(),
        demand_col=demand_col,
        price_col=price_col,
        cost_col=cost_col,
        date_col=date_col,
    )

    if len(df) == 0:
        raise ValueError("Нет данных для baseline-сравнения.")

    sim_start = pd.to_datetime(df[date_col]).min()
    warmup_df = _prepare_warmup_input(
        history_df=history_df,
        sim_start=sim_start,
        demand_col=demand_col,
        price_col=price_col,
        cost_col=cost_col,
        date_col=date_col,
    )

    strategies = [
        HistoricalPriceBaseline(),
        FixedPriceBaseline(),
        CostPlusBaseline(markup=markup),
        HeuristicReorderBaseline(),
    ]

    results = []
    for strat in strategies:
        try:
            res = strat.simulate(
                df,
                demand_col=demand_col,
                price_col=price_col,
                cost_col=cost_col,
                date_col=date_col,
                initial_stock=initial_stock,
                warmup_df=warmup_df,
            )
            results.append(res.summary())
        except Exception as exc:
            warnings.warn(f"Стратегия {strat.__class__.__name__} упала: {exc}")

    # ML через _WarehouseSimulator — честное сравнение
    if ml_results is not None and len(ml_results) > 0:
        try:
            ml_strat = MLOptimizerBaseline(ml_results)
            ml_res = ml_strat.simulate(
                df,
                demand_col=demand_col,
                price_col=price_col,
                cost_col=cost_col,
                date_col=date_col,
                initial_stock=initial_stock,
                warmup_df=warmup_df,
            )
            results.append(ml_res.summary())
        except Exception as exc:
            warnings.warn(f"ML_OptimizerBaseline упал: {exc}")

    if not results:
        raise RuntimeError("Ни одна стратегия не выполнилась успешно.")

    table = pd.DataFrame(results).set_index('strategy')

    # Считаем uplift ML над лучшим baseline по GMV
    if 'ML_Optimizer' in table.index and len(results) > 1:
        baseline_gmv = table.drop(index='ML_Optimizer')['total_gmv'].max()
        ml_gmv       = table.loc['ML_Optimizer', 'total_gmv']
        uplift_pct   = (ml_gmv - baseline_gmv) / max(baseline_gmv, 1e-8) * 100
        print(f"\n📊 Uplift ML над лучшим baseline по GMV: {uplift_pct:+.2f}%")

    return table
