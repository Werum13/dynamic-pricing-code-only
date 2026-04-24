from __future__ import annotations

import hashlib
import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

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
ORDER_CHUNKSIZE = 500_000
CACHE_VERSION = 1


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


def _read_orders_subset(orders_path: Path, order_ids: set[str]) -> pd.DataFrame:
    """Read only orders that are needed for filtered order-details rows."""
    if not order_ids:
        return pd.DataFrame(columns=ORDER_COLS)

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        orders_path,
        dtype=str,
        usecols=ORDER_COLS,
        chunksize=ORDER_CHUNKSIZE,
        low_memory=False,
    ):
        chunk["ORDERID"] = chunk["ORDERID"].astype("string").str.strip()
        chunk = chunk[chunk["ORDERID"].isin(order_ids)]
        chunk = chunk.dropna(subset=["ORDERID", "DATE_"])
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=ORDER_COLS)

    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["ORDERID"], keep="last")


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


def _normalize_itemcodes_filter(itemcodes: Iterable[int] | None) -> set[int] | None:
    """Normalize itemcode filter values to plain int set."""
    if itemcodes is None:
        return None
    values = pd.to_numeric(pd.Series(list(itemcodes), dtype="object"), errors="coerce")
    values = values.dropna().astype("int64")
    return set(values.tolist())


def _itemcode_string_variants(itemcodes: set[int]) -> set[str]:
    """Build robust string representations for quick pre-filtering in raw CSV chunks."""
    variants: set[str] = set()
    for code in itemcodes:
        text = str(int(code))
        variants.add(text)
        variants.add(f"{text}.0")
        variants.add(f"{text},0")
    return variants


def _cache_dir(workspace_root: Path) -> Path:
    d = workspace_root / "output" / "elasticity" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(usecols: list[str] | None, itemcodes: set[int]) -> str:
    payload = json.dumps(
        {
            "usecols": usecols or VALID_OUTPUT_COLS,
            "itemcodes": sorted(itemcodes),
            "v": CACHE_VERSION,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_paths(workspace_root: Path, usecols: list[str] | None, itemcodes: set[int]) -> tuple[Path, Path]:
    key = _cache_key(usecols, itemcodes)
    base = _cache_dir(workspace_root)
    return base / f"source_{key}.pkl", base / f"source_{key}.meta.json"


def _load_cached_source_data(
    workspace_root: Path,
    usecols: list[str] | None,
    itemcodes: set[int],
) -> pd.DataFrame | None:
    data_path, meta_path = _cache_paths(workspace_root, usecols, itemcodes)
    if not data_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("cache_version") != CACHE_VERSION:
            return None
        if meta.get("source_signature") != source_signature(workspace_root):
            return None
        return pd.read_pickle(data_path)
    except Exception:
        return None


def _save_cached_source_data(
    workspace_root: Path,
    usecols: list[str] | None,
    itemcodes: set[int],
    df: pd.DataFrame,
) -> None:
    data_path, meta_path = _cache_paths(workspace_root, usecols, itemcodes)
    df.to_pickle(data_path)
    meta = {
        "cache_version": CACHE_VERSION,
        "source_signature": source_signature(workspace_root),
        "rows": int(len(df)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)


def _normalize_selected_chunk(
    chunk: pd.DataFrame,
    selected_cols: list[str],
) -> pd.DataFrame:
    for numeric_col in ("UNITPRICE", "TOTALPRICE", "AMOUNT"):
        chunk[numeric_col] = _normalize_numeric_series(chunk[numeric_col])
    return chunk[selected_cols].copy()


def iter_elasticity_source_chunks(
    workspace_root: Path,
    *,
    chunksize: int,
    usecols: list[str] | None = None,
    itemcodes: Iterable[int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield merged/normalized chunks built from Order_Details, Orders, and Categories files."""
    data_dir = workspace_root / "data"
    files = _require_data_files(data_dir)
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
    itemcode_filter = _normalize_itemcodes_filter(itemcodes)

    if itemcode_filter is not None and len(itemcode_filter) == 0:
        return

    if itemcode_filter is None:
        orders = _read_orders(files["orders"])

        for chunk in pd.read_csv(
            files["order_details"],
            dtype=str,
            usecols=DETAIL_COLS,
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk["ITEMCODE"] = _normalize_itemcode(chunk["ITEMCODE"])
            chunk = chunk[chunk["ITEMCODE"].notna()].copy()

            chunk["ORDERID"] = chunk["ORDERID"].astype("string").str.strip()
            chunk["ITEMID"] = chunk["ITEMID"].astype("string").str.strip()

            merged = chunk.merge(orders, on="ORDERID", how="left")
            merged = merged.merge(categories, on="ITEMID", how="left")
            yield _normalize_selected_chunk(merged, selected_cols)
        return

    # Fast path for filtered ITEMCODE requests.
    # 1) Scan order_details and keep only matching rows.
    # 2) Read only needed orders.
    # 3) Merge once and yield.
    filtered_detail_parts: list[pd.DataFrame] = []
    needed_order_ids: set[str] = set()
    itemcode_variants = _itemcode_string_variants(itemcode_filter)
    chunks_seen = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        files["order_details"],
        dtype=str,
        usecols=DETAIL_COLS,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunks_seen += 1

        raw_itemcode = chunk["ITEMCODE"].astype("string").str.strip()
        quick_mask = raw_itemcode.isin(itemcode_variants)
        if not quick_mask.any():
            if chunks_seen % 20 == 0:
                print(f"Scanning source chunks... seen={chunks_seen}, matched_rows={matched_rows}")
            continue

        chunk = chunk.loc[quick_mask].copy()
        chunk["ITEMCODE"] = _normalize_itemcode(raw_itemcode.loc[quick_mask])
        chunk = chunk[chunk["ITEMCODE"].notna()]
        chunk = chunk[chunk["ITEMCODE"].isin(itemcode_filter)]
        if chunk.empty:
            if chunks_seen % 20 == 0:
                print(f"Scanning source chunks... seen={chunks_seen}, matched_rows={matched_rows}")
            continue

        matched_rows += len(chunk)

        chunk["ORDERID"] = chunk["ORDERID"].astype("string").str.strip()
        chunk["ITEMID"] = chunk["ITEMID"].astype("string").str.strip()
        needed_order_ids.update(chunk["ORDERID"].dropna().astype(str).tolist())
        filtered_detail_parts.append(chunk)

        if chunks_seen % 20 == 0:
            print(f"Scanning source chunks... seen={chunks_seen}, matched_rows={matched_rows}")

    if not filtered_detail_parts:
        return

    details = pd.concat(filtered_detail_parts, ignore_index=True)
    orders = _read_orders_subset(files["orders"], needed_order_ids)
    merged = details.merge(orders, on="ORDERID", how="left")
    merged = merged.merge(categories, on="ITEMID", how="left")
    yield _normalize_selected_chunk(merged, selected_cols)


def load_elasticity_source_data(
    workspace_root: Path,
    *,
    usecols: list[str] | None = None,
    itemcodes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Load full elasticity source data by concatenating normalized chunks."""
    itemcode_filter = _normalize_itemcodes_filter(itemcodes)
    if itemcode_filter:
        cached = _load_cached_source_data(workspace_root, usecols, itemcode_filter)
        if cached is not None:
            return cached

    chunks = iter_elasticity_source_chunks(
        workspace_root,
        chunksize=DEFAULT_CHUNKSIZE,
        usecols=usecols,
        itemcodes=itemcode_filter,
    )
    first_chunk = next(chunks, None)
    if first_chunk is None:
        return pd.DataFrame(columns=usecols or VALID_OUTPUT_COLS)
    data = pd.concat(chain([first_chunk], chunks), ignore_index=True)

    if itemcode_filter:
        _save_cached_source_data(workspace_root, usecols, itemcode_filter, data)

    return data
