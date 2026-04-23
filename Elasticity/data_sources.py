from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


ORDER_DETAILS_FILE = "Order_Details.csv"
ORDERS_FILE = "Orders.csv"
CATEGORIES_FILE = "Categories_ENG.csv"

DETAIL_COLS = ["ORDERID", "ITEMID", "ITEMCODE", "UNITPRICE", "TOTALPRICE", "AMOUNT"]
ORDER_COLS = ["ORDERID", "DATE_"]
CATEGORY_COLS = ["ITEMID", "CATEGORY1", "CATEGORY2"]


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


def iter_elasticity_source_chunks(
    workspace_root: Path,
    *,
    chunksize: int,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
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
    selected_cols = [col for col in requested_cols if col in {"ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"}]

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
        merged["ITEMCODE"] = pd.to_numeric(merged["ITEMCODE"], errors="coerce").astype("Int64")

        yield merged[selected_cols].copy()


def load_elasticity_source_data(
    workspace_root: Path,
    *,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    frames = list(iter_elasticity_source_chunks(workspace_root, chunksize=200_000, usecols=usecols))
    if not frames:
        return pd.DataFrame(columns=usecols or [])
    return pd.concat(frames, ignore_index=True)
