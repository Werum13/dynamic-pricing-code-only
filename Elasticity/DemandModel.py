"""
DemandModel.py
==============
Модель спроса Q(P, features) — прямое предсказание количества.

Почему Q(P) лучше, чем R(P) = P * Q(P)
-----------------------------------------
1. Экономическая корректность
   Спрос Q(P) — фундаментальная экономическая концепция.
   Оптимальная цена ищется из задачи:
       max  π(P) = (P − MC) · Q(P)
        P
   Если учить модель на R(P) = P·Q(P), то производная dR/dP = Q + P·dQ/dP
   содержит P как мультипликативный множитель, что смещает градиент:
   модель "думает", что увеличение цены всегда увеличивает выручку,
   пока не видит явного снижения R. Это занижает эластичность.

2. Устойчивость к малым ценам
   Q̂(P) = R̂(P) / P ↗ ∞ при P → 0.
   Деление на цену после обучения усиливает шум при маленьких ценах.
   Прямое предсказание Q не имеет этой проблемы.

3. Интерпретируемость
   Эластичность считается напрямую через dQ/dP, без цепного правила
   дифференцирования составной функции (P · Q(P)).

4. Монотонность проще обеспечить
   Закон спроса: dQ/dP < 0. Для Q(P) это легче проверить / наложить
   постпроцессингом, чем для выпуклости R(P).

Целевая переменная
------------------
Для daily repricing используем AMOUNT_0D_target — спрос в тот же день, что и цена.
Признаки уже сдвинуты на 1 день назад (shift(1)), поэтому утечки нет:
в строке за день T признаки = история до T, цена = P_T, таргет = Q_T.
Это прямая Q(P) без временного смещения между ценой и спросом.

Формула эластичности
--------------------
Центральная разность (численная производная):
    ε̂(P) = dQ/dP · P / Q
          ≈ [Q̂(P+δ) − Q̂(P−δ)] / (2δ) · P / Q̂(P)
    δ = 0.01 · P  (1% от цены)

Норма: ε < 0 (закон спроса). |ε| > 1 — эластичный товар.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV, Ridge
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator

_EPS = 1e-8


class DemandModel:
    """
    Модель спроса Q(P, features).

    Архитектура: Poly(2) → StandardScaler → LassoCV.
    LassoCV автоматически регуляризует → устойчивость к мультиколлинеарности.

    Parameters
    ----------
    df : pd.DataFrame
        Признаки (без целевой переменной и служебных колонок).
    target : pd.Series
        Целевая переменная — AMOUNT_0D_target (количество, тот же день, что и цена).
    random_state : int
        Для воспроизводимости.
    """

    _DROP_COLS = {'ITEMCODE', 'Id', 'n_sku'}

    def __init__(self, df: pd.DataFrame, target: pd.Series, random_state: int = 42):
        self.df           = df
        self.target       = target
        self.random_state = random_state
        self.pipeline_    = None   # sklearn Pipeline после fit()
        self.columns_     = None   # Список признаков для predict()
        self.residual_quantiles_ = (-0.0, 0.0)
        self.residual_rmse_      = 0.0
        self.interval_level_     = 0.80

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self) -> "DemandModel":
        """
        Обучение модели Q(P, features).

        Обнуляем NaN в target (редкие пропуски в future-target),
        обрезаем снизу нулём — количество не может быть отрицательным.
        """
        target = self.target.fillna(0).clip(lower=0)

        X = self._prepare_X(self.df)
        self.columns_ = X.columns.tolist()

        self.pipeline_ = Pipeline([
            ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=0.01))
        ])

        self.pipeline_.fit(X, target)
        fitted = np.clip(self.pipeline_.predict(X), 0, np.inf)
        residuals = target.to_numpy() - fitted
        if len(residuals) > 0:
            q_low, q_high = np.quantile(residuals, [0.1, 0.9])
            self.residual_quantiles_ = (float(q_low), float(q_high))
            self.residual_rmse_ = float(np.sqrt(np.mean(residuals ** 2)))
        return self

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание спроса Q(P, features).

        Returns
        -------
        data с колонками:
          - pred_Q
          - pred_Q_lower / pred_Q_upper
          - pred_GMV и соответствующие lower / upper
        """
        if self.pipeline_ is None:
            raise RuntimeError("Вызовите .fit() перед .predict()")

        X = data[self.columns_]
        pred = self.pipeline_.predict(X)
        pred = np.clip(pred, 0, np.inf)   # Q ≥ 0
        q_low, q_high = self.residual_quantiles_
        pred_lower = np.clip(pred + q_low, 0, np.inf)
        pred_upper = np.clip(pred + q_high, 0, np.inf)
        pred_upper = np.maximum(pred_upper, pred_lower)

        data = data.copy()
        data['pred_Q']         = pred
        data['pred_Q_lower']   = pred_lower
        data['pred_Q_upper']   = pred_upper
        data['pred_GMV']       = data['UNITPRICE'] * pred   # для совместимости
        data['pred_GMV_lower'] = data['UNITPRICE'] * pred_lower
        data['pred_GMV_upper'] = data['UNITPRICE'] * pred_upper
        data['pred_interval_level'] = self.interval_level_
        return data

    # ------------------------------------------------------------------
    # Эластичность (численная производная)
    # ------------------------------------------------------------------

    def elasticity(
        self,
        data: pd.DataFrame,
        delta_frac: float = 0.01
    ) -> np.ndarray:
        """
        Точечная ценовая эластичность спроса.

        ε̂(P) = dQ/dP · P / Q
              ≈ [Q̂(P+δ) − Q̂(P−δ)] / (2δ) · P / Q̂(P)

        Формула центральной разности второго порядка точности O(δ²).
        δ = delta_frac · P  (по умолчанию 1% от цены).

        Parameters
        ----------
        data : pd.DataFrame
            Данные с UNITPRICE и всеми признаками.
        delta_frac : float
            Доля приращения цены для численной производной.

        Returns
        -------
        np.ndarray : вектор эластичностей (обычно < 0).

        Предупреждения
        --------------
        - При очень маленькой цене (P → 0) δ → 0, погрешность растёт.
          Используйте min_price в price_optimizer для отсечения таких случаев.
        - Шум в данных → шумная производная. Сгладьте Q(P) или добавьте
          регуляризацию перед вызовом.
        """
        P     = data['UNITPRICE'].values
        delta = np.maximum(delta_frac * P, _EPS)

        df_up   = data.copy()
        df_down = data.copy()
        df_up['UNITPRICE']   = P + delta
        df_down['UNITPRICE'] = P - delta

        # Обновляем производные признаки после изменения цены
        for df_ in [df_up, df_down]:
            _refresh_price_features(df_)

        Q_up   = self.predict(df_up)['pred_Q'].values
        Q_down = self.predict(df_down)['pred_Q'].values
        Q_base = self.predict(data.copy())['pred_Q'].values

        dQ_dP = (Q_up - Q_down) / (2.0 * delta)
        eps   = dQ_dP * P / np.maximum(Q_base, _EPS)
        return eps

    # ------------------------------------------------------------------
    # Постобработка: монотонность спроса
    # ------------------------------------------------------------------

    def enforce_monotonicity(
        self,
        data: pd.DataFrame,
        group_col: str = 'Id',
        min_diff: float = 1e-3,
        tail_mult: float = 1.5
    ) -> pd.DataFrame:
        """
        Пост-процессинг: монотонное убывание Q(P) по каждой позиции.

        Закон спроса требует dQ/dP ≤ 0.
        Если модель нарушает это — принудительно накладываем cumulative min
        (moving minimum справа налево) и интерполируем PCHIP.

        Parameters
        ----------
        min_diff : float
            Минимальная разница между соседними точками кривой.
            Убирает "плато" и вырожденные случаи.
        tail_mult : float
            Множитель для нулевой цены хвоста (Q → 0 при P_tail = P_max · tail_mult).
        """
        def _process(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < 2:
                return group

            group = group.sort_values('UNITPRICE').copy()
            P = group['UNITPRICE'].values
            Q = group['pred_Q'].values

            # Isotonic regression (decreasing) → минимально искажаем кривую,
            # но гарантируем закон спроса: Q(P_low) >= Q(P_high).
            try:
                iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
                Q_mono = iso.fit_transform(P, Q)
            except Exception:
                # Fallback: cumulative maximum справа налево
                Q_mono = np.maximum.accumulate(Q[::-1])[::-1]

            # Убираем точки без изменения Q
            keep_P, keep_Q = [P[0]], [Q_mono[0]]
            for i in range(1, len(Q_mono)):
                if abs(keep_Q[-1] - Q_mono[i]) >= min_diff:
                    keep_P.append(P[i])
                    keep_Q.append(Q_mono[i])

            # Добавляем нулевой хвост (Q=0 при P_tail)
            keep_P.append(keep_P[-1] * tail_mult)
            keep_Q.append(0.0)

            if len(keep_P) >= 2:
                try:
                    interp = PchipInterpolator(keep_P, keep_Q)
                    group['pred_Q'] = np.clip(interp(P), 0, None)
                except Exception:
                    pass

            return group

        return (
            data
            .groupby(group_col, group_keys=False)
            .apply(_process)
            .sort_index()
        )

    @staticmethod
    def monotonicize_curve(
        prices,
        demand,
        lower=None,
        upper=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Монотонизирует кривую спроса Q(P) на возрастающей сетке цен.

        Возвращает:
          prices_sorted, q_mono, q_lower_mono, q_upper_mono
        """
        P = np.asarray(prices, dtype=float)
        Q = np.asarray(demand, dtype=float)

        if len(P) == 0:
            return P, Q, None if lower is None else np.asarray(lower), None if upper is None else np.asarray(upper)

        order = np.argsort(P)
        P = P[order]
        Q = np.clip(Q[order], 0, np.inf)

        def _fit(values):
            arr = np.clip(np.asarray(values, dtype=float)[order], 0, np.inf)
            if len(arr) < 2:
                return arr
            try:
                iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
                return np.clip(iso.fit_transform(P, arr), 0, np.inf)
            except Exception:
                return np.maximum.accumulate(arr[::-1])[::-1]

        Q_mono = _fit(Q)
        Q_low = _fit(lower) if lower is not None else None
        Q_high = _fit(upper) if upper is not None else None

        if Q_low is not None:
            Q_low = np.minimum(Q_low, Q_mono)
        if Q_high is not None:
            Q_high = np.maximum(Q_high, Q_mono)
        if Q_low is not None and Q_high is not None:
            Q_high = np.maximum(Q_high, Q_low)

        return P, Q_mono, Q_low, Q_high

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self._DROP_COLS if c in df.columns]
        return df.drop(columns=drop)

    def get_sklearn_pipeline(self):
        """Возвращает обученный sklearn Pipeline."""
        return self.pipeline_

    def get_uncertainty_summary(self) -> dict:
        """Краткая сводка по residual-based uncertainty bands."""
        return {
            'interval_level': self.interval_level_,
            'residual_q10': self.residual_quantiles_[0],
            'residual_q90': self.residual_quantiles_[1],
            'residual_rmse': self.residual_rmse_,
        }


# ------------------------------------------------------------------
# Вспомогательная функция: обновление производных признаков
# ------------------------------------------------------------------

def _refresh_price_features(df: pd.DataFrame) -> None:
    """
    Пересчитывает discount и margin_% после изменения UNITPRICE.
    Вызывается при численном дифференцировании.
    """
    if 'BASEPRICE' in df.columns:
        df['discount'] = 1.0 - df['UNITPRICE'] / df['BASEPRICE'].replace(0, _EPS)
    if 'cost' in df.columns:
        df['margin_%'] = (df['UNITPRICE'] - df['cost']) / df['cost'].replace(0, _EPS)


# ------------------------------------------------------------------
# Функция-фабрика (совместимость с pipeline.py)
# ------------------------------------------------------------------

def demand_model(train: pd.DataFrame, target: pd.Series) -> DemandModel:
    """
    Обёртка: создаёт и обучает модель спроса.

    Parameters
    ----------
    train : pd.DataFrame
        Признаки (результат ETL, без целевой).
    target : pd.Series
        AMOUNT_0D_target — спрос в тот же день, что и цена (Q(P), без временного сдвига).

    Returns
    -------
    DemandModel : обученная модель.
    """
    model = DemandModel(df=train, target=target)
    model.fit()
    return model
