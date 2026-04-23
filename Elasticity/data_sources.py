from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterator

import pandas as pd
from pandas.api.types import is_numeric_dtype


ORDER_DETAILS_FILE = "Order_Details.csv"
ORDERS_FILE = "Orders.csv"
CATEGORIES_FILE = "Categories_ENG.csv"

DETAIL_COLS = ["ORDERID", "ITEMID", "ITEMCODE", "UNITPRICE", "TOTALPRICE", "AMOUNT"]
ORDER_COLS = ["ORDERID", "DATE_"]
CATEGORY_COLS = ["ITEMID", "CATEGORY1", "CATEGORY2"]
VALID_OUTPUT_COLS = ["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"]
DEFAULT_CHUNKSIZE = 200_000


def _data_files(data_dir: Path) -> dict[str, Path]:
    return {
        "order_details": data_dir / ORDER_DETAILS_FILE,
        "orders": data_dir / ORDERS_FILE,
        "categories": data_dir / CATEGORIES_FILE,
    }


def _require_data_files(data_dir: Path) -> dict[str, Path]:
    files = _data_files(data_dir)
    missing = [path for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required data files: " + ", ".join(str(path.resolve()) for path in missing)
        )
    return files


def source_signature(workspace_root: Path) -> dict[str, dict[str, float | int]]:
    """Return source-file metadata used to invalidate cached derived data."""
    files = _require_data_files(workspace_root / "data")
    return {
        name: {"mtime": path.stat().st_mtime, "size": path.stat().st_size}
        for name, path in files.items()
    }


def _read_orders(orders_path: Path) -> pd.DataFrame:
    orders = pd.read_csv(orders_path, dtype=str, usecols=ORDER_COLS, low_memory=False)
    orders["ORDERID"] = orders["ORDERID"].astype("string").str.strip()
    orders = orders.dropna(subset=["ORDERID", "DATE_"])
    return orders


def _read_categories(categories_path: Path) -> pd.DataFrame:
    categories = pd.read_csv(
        categories_path,
        sep=";",
        dtype=str,
        usecols=CATEGORY_COLS,
        low_memory=False,
    )
    categories["ITEMID"] = categories["ITEMID"].astype("string").str.strip()
    categories = categories.dropna(subset=["ITEMID"]).drop_duplicates(subset=["ITEMID"], keep="first")
    return categories


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    """Parse numeric values, supporting comma decimal separators."""
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _normalize_itemcode(series: pd.Series) -> pd.Series:
    """Convert item codes to nullable integers; invalid codes become <NA>."""
    # to_numeric(..., errors="coerce") marks invalid codes as NaN; Int64 keeps them as <NA>.
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def iter_elasticity_source_chunks(
    workspace_root: Path,
    *,
    chunksize: int,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield merged/normalized chunks built from Order_Details, Orders, and Categories files."""
    data_dir = workspace_root / "data"
    files = _require_data_files(data_dir)

    orders = _read_orders(files["orders"])
    categories = _read_categories(files["categories"])

    requested_cols = usecols or [
        "ITEMCODE",
        "DATE_",
        "UNITPRICE",
        "TOTALPRICE",
        "AMOUNT",
        "CATEGORY1",
        "CATEGORY2",
    ]
    selected_cols = [col for col in requested_cols if col in VALID_OUTPUT_COLS]

    for chunk in pd.read_csv(
        files["order_details"],
        dtype=str,
        usecols=DETAIL_COLS,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["ORDERID"] = chunk["ORDERID"].astype("string").str.strip()
        chunk["ITEMID"] = chunk["ITEMID"].astype("string").str.strip()

        merged = chunk.merge(orders, on="ORDERID", how="left")
        merged = merged.merge(categories, on="ITEMID", how="left")
        merged["ITEMCODE"] = _normalize_itemcode(merged["ITEMCODE"])
        merged = merged[merged["ITEMCODE"].notna()].copy()
        for numeric_col in ("UNITPRICE", "TOTALPRICE", "AMOUNT"):
            merged[numeric_col] = _normalize_numeric_series(merged[numeric_col])

        yield merged[selected_cols].copy()


def load_elasticity_source_data(
    workspace_root: Path,
    *,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load full elasticity source data by concatenating normalized chunks."""
    chunks = iter_elasticity_source_chunks(workspace_root, chunksize=DEFAULT_CHUNKSIZE, usecols=usecols)
    first_chunk = next(chunks, None)
    if first_chunk is None:
        return pd.DataFrame(columns=usecols or VALID_OUTPUT_COLS)
    return pd.concat(chain([first_chunk], chunks), ignore_index=True)
