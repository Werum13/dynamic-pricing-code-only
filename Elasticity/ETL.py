import pandas as pd
from typing import Iterable
import numpy as np

_EPS = 1e-8

class ETL:
    """
    Production-ready ETL для фичей цен и временных агрегатов
    """

    # -----------------------------
    # ВСПОМОГАТЕЛЬНЫЕ ПРОВЕРКИ
    # -----------------------------
    @staticmethod
    def _check_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    # -----------------------------
    # BINNING ЦЕН
    # -----------------------------
    @staticmethod
    def aggregate_price_bins(df: pd.DataFrame, n_bins: int = 10, min_unique_prices: int = 2):
        """
        Биннинг UNITPRICE внутри ITEMCODE и агрегация AMOUNT / TOTALPRICE
        """

        ETL._check_columns(
            df,
            ['ITEMCODE', 'DATE_', 'UNITPRICE', 'AMOUNT', 'BASEPRICE']
        )

        def _bin_item(group: pd.DataFrame) -> pd.DataFrame:
            if group['UNITPRICE'].nunique() < min_unique_prices:
                return group

            prices = group['UNITPRICE']

            bins = np.linspace(prices.min(), prices.max(), n_bins + 1)
            labels = (bins[:-1] + bins[1:]) / 2

            group = group.copy()
            group['UNITPRICE'] = pd.cut(
                prices,
                bins=bins,
                labels=labels,
                include_lowest=True
            ).astype(float)

            return group
        df = (df
            .groupby(['ITEMCODE', 'DATE'], group_keys=False)
            .apply(_bin_item)
            .agg({
                'UNITPRICE': 'first',
                'AMOUNT': 'sum',
                'BASEPRICE': 'mean'
            })
            .reset_index()
        )





        # обрезка краёв
        min_date = df['DATE_'].min() + pd.Timedelta('7D')
        max_date = df['DATE_'].max() - pd.Timedelta('7D')
       
        return df[(df['DATE_'] >= min_date) & ((df['DATE_'] <= max_date) | (df['DATE_'] == df['DATE_'].max()))].sort_values(['ITEMCODE', 'DATE_']).reset_index(drop=True)


def _first_non_null(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.iloc[0]


def etl_with_demand_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит данные к дневному уровню и добавляет лаговые/целевые признаки.
    """
    required_cols = ['ITEMCODE', 'DATE_', 'UNITPRICE', 'TOTALPRICE', 'AMOUNT']
    ETL._check_columns(df, required_cols)

    df = df.copy()
    df['DATE_'] = pd.to_datetime(df['DATE_'], errors='coerce')
    df = df.dropna(subset=['ITEMCODE', 'DATE_'])
    if df.empty:
        return df

    numeric_cols = ['UNITPRICE', 'TOTALPRICE', 'AMOUNT', 'BASEPRICE', 'cost']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'BASEPRICE' not in df.columns:
        df['BASEPRICE'] = df['UNITPRICE']

    agg_map = {
        'UNITPRICE': 'mean',
        'TOTALPRICE': 'sum',
        'AMOUNT': 'sum',
        'BASEPRICE': 'mean',
    }
    if 'cost' in df.columns:
        agg_map['cost'] = 'mean'
    if 'CATEGORY1' in df.columns:
        agg_map['CATEGORY1'] = _first_non_null
    if 'CATEGORY2' in df.columns:
        agg_map['CATEGORY2'] = _first_non_null

    daily = (
        df.sort_values(['ITEMCODE', 'DATE_'])
        .groupby(['ITEMCODE', 'DATE_'])
        .agg(agg_map)
        .reset_index()
        .sort_values(['ITEMCODE', 'DATE_'])
        .reset_index(drop=True)
    )

    def _past_rolling(values: pd.Series, window: int) -> pd.Series:
        return values.shift(1).rolling(window=window, min_periods=1).sum()

    def _future_rolling(values: pd.Series, window: int) -> pd.Series:
        reversed_values = values.iloc[::-1]
        forward_sum = reversed_values.rolling(window=window + 1, min_periods=1).sum().iloc[::-1]
        return forward_sum - values

    def _add_item_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values('DATE_').reset_index(drop=True).copy()

        group['Id'] = 0
        group['dayofweek'] = group['DATE_'].dt.dayofweek
        group['month'] = group['DATE_'].dt.month
        group['day'] = group['DATE_'].dt.day
        group['weekofyear'] = group['DATE_'].dt.isocalendar().week.astype(int)
        group['is_weekend'] = (group['dayofweek'] >= 5).astype(int)

        if 'BASEPRICE' not in group.columns:
            group['BASEPRICE'] = group['UNITPRICE']

        baseprice = group['BASEPRICE'].replace(0, np.nan)
        group['discount'] = 1.0 - group['UNITPRICE'] / baseprice

        if 'cost' in group.columns:
            cost = group['cost'].replace(0, np.nan)
            group['margin_%'] = (group['UNITPRICE'] - group['cost']) / cost
            group['margin_%'] = group['margin_%'].fillna(0.0)

        for window in [1, 7, 15, 30]:
            amount_roll = _past_rolling(group['AMOUNT'], window).fillna(0.0)
            totalprice_roll = _past_rolling(group['TOTALPRICE'], window).fillna(0.0)

            group[f'AMOUNT_{window}D'] = amount_roll
            group[f'GMV_{window}D'] = totalprice_roll
            group[f'TOTALPRICE_{window}D'] = totalprice_roll

            group[f'AMOUNT_{window}D_target'] = _future_rolling(group['AMOUNT'], window)

        group['AMOUNT_0D_target'] = group['AMOUNT']
        return group

    processed_groups = []
    for itemcode, group in daily.groupby('ITEMCODE', sort=False):
        item_group = _add_item_features(group)
        item_group['ITEMCODE'] = itemcode
        processed_groups.append(item_group)

    if not processed_groups:
        return daily.iloc[0:0].copy()

    return (
        pd.concat(processed_groups, ignore_index=True)
        .sort_values(['ITEMCODE', 'DATE_'])
        .reset_index(drop=True)
    )
