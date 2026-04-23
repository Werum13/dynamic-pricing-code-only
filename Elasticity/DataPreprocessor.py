import pandas as pd
from typing import Iterable

class DataPreprocessor:
    """
    Production-ready preprocessing:
    - приведение типов
    - расчёт UNITPRICE
    - удаление выбросов (IQR) внутри ITEMCODE
    """

    # -----------------------------
    # VALIDATION
    # -----------------------------
    @staticmethod
    def _check_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    # -----------------------------
    # TYPE CASTING
    # -----------------------------
    @staticmethod
    def cast_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Приведение типов:
        - TOTALPRICE → float
        - DATE_ → datetime
        """

        DataPreprocessor._check_columns(df, ['TOTALPRICE', 'DATE_'])

        df = df.copy()

        df['TOTALPRICE'] = (
            df['TOTALPRICE']
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

        df['UNITPRICE'] = (
            df['UNITPRICE']
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )

        df['DATE_'] = pd.to_datetime(df['DATE_'], errors='raise')

        return df

    # -----------------------------
    # UNITPRICE
    # -----------------------------
    @staticmethod
    def compute_unit_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        UNITPRICE = TOTALPRICE / AMOUNT
        """

        DataPreprocessor._check_columns(df, ['TOTALPRICE', 'AMOUNT'])

        if (df['AMOUNT'] <= 0).any():
            raise ValueError("AMOUNT contains zero or negative values")

        df = df.copy()
        df['BASEPRICE'] = df['UNITPRICE']
        df['UNITPRICE'] = df['TOTALPRICE'] / df['AMOUNT']

        return df

    # -----------------------------
    # OUTLIERS (IQR)
    # -----------------------------
    @staticmethod
    def remove_iqr_outliers(
        df: pd.DataFrame,
        value_col: str = 'UNITPRICE',
        group_col: str = 'ITEMCODE',
        k: float = 1.5,
        min_obs: int = 5
    ) -> pd.DataFrame:
        """
        Удаление выбросов по IQR внутри группы
        """

        DataPreprocessor._check_columns(df, [value_col, group_col])

        def _filter(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) < min_obs:
                return group

            q1, q3 = group[value_col].quantile([0.25, 0.75])
            iqr = q3 - q1

            if iqr == 0:
                return group

            lower = q1 - k * iqr
            upper = q3 + k * iqr

            return group[
                (group[value_col] >= lower) &
                (group[value_col] <= upper)
            ]

        filtered_groups = []
        for _, group in df.groupby(group_col, sort=False):
            filtered_groups.append(_filter(group))

        if not filtered_groups:
            return df.iloc[0:0].copy()

        return pd.concat(filtered_groups, ignore_index=True)
    


def preprocessor(data):
    df = DataPreprocessor.cast_types(data)
    df = DataPreprocessor.compute_unit_price(df)
    df = DataPreprocessor.remove_iqr_outliers(df)

    return df