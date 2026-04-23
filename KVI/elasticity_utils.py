"""Utilities for loading and standardizing elasticity sources.

LST files now contain item code and one elasticity value per item.
This helper normalizes the source into an ITEMID-based representation
without date-based processing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _normalize_column_name(name: Any) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return normalized.strip("_")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [_normalize_column_name(column) for column in renamed.columns]
    return renamed


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.where(series.notna(), pd.NA).astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
    return cleaned


def discover_elasticity_source(data_dir: Path) -> Path | None:
    lst_candidates = sorted(
        [path for path in data_dir.glob("LSTCSV*") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if lst_candidates:
        return lst_candidates[0]

    legacy = data_dir / "elasticity.csv"
    if legacy.exists():
        return legacy

    return None


def _read_raw_elasticity(source_path: Path) -> pd.DataFrame:
    sql_path = str(source_path).replace("'", "''")
    con = duckdb.connect()
    return con.execute(
        f"SELECT * FROM read_csv_auto('{sql_path}', header=true, all_varchar=true)"
    ).df()


def standardize_elasticity_source(
    data_dir: Path,
    categories_path: Path | None,
    output_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load the raw elasticity source and return an ITEMID-level frame.

    Returns:
        item_level_df: Frame with one row per ITEMID and item-level elasticity.
        metadata: Diagnostics about the source and merge quality.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = discover_elasticity_source(data_dir)
    metadata: dict[str, Any] = {
        "source_path": str(source_path) if source_path else None,
        "source_kind": None,
        "raw_rows": 0,
        "valid_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "distinct_itemids": 0,
        "distinct_itemcodes": 0,
        "duplicate_itemids": 0,
        "mapping_method": "direct_item_value",
        "item_level_path": None,
    }

    if source_path is None:
        return None, metadata

    raw = _normalize_columns(_read_raw_elasticity(source_path))
    metadata["raw_rows"] = int(len(raw))

    itemid_col = _pick_column(list(raw.columns), ["itemid"])
    itemcode_col = _pick_column(list(raw.columns), ["itemcode", "item_code", "item code"])
    elasticity_col = _pick_column(list(raw.columns), ["elasticity", "elasticity_value", "elas", "elastic"])

    if elasticity_col is None:
        raise ValueError(
            f"Elasticity source {source_path.name} does not contain an elasticity column."
        )

    if itemid_col is None and itemcode_col is None:
        raise ValueError(
            f"Elasticity source {source_path.name} must contain ITEMID or item code columns."
        )

    series = raw.copy()
    elasticity_values = (
        series[elasticity_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    series["elasticity"] = pd.to_numeric(elasticity_values, errors="coerce")
    series = series[series["elasticity"].notna()].copy()
    metadata["valid_rows"] = int(len(series))

    if itemid_col is not None:
        series["ITEMID"] = _clean_string_series(series[itemid_col])
        metadata["source_kind"] = "itemid"
    else:
        if categories_path is None:
            raise ValueError("categories_path is required when the elasticity source contains item codes only.")

        categories = _normalize_columns(pd.read_csv(categories_path, sep=";", dtype=str, low_memory=False))
        if "itemid" not in categories.columns or "itemcode" not in categories.columns:
            raise ValueError("Categories file must contain ITEMID and ITEMCODE columns for the merge.")

        categories = categories[["itemid", "itemcode"]].drop_duplicates().copy()
        categories["itemid"] = _clean_string_series(categories["itemid"])
        categories["itemcode"] = _clean_string_series(categories["itemcode"])

        series["ITEMCODE"] = _clean_string_series(series[itemcode_col])
        metadata["source_kind"] = "itemcode"
        metadata["distinct_itemcodes"] = int(series["ITEMCODE"].nunique())

        series = series.merge(categories, left_on="ITEMCODE", right_on="itemcode", how="left")
        series = series.rename(columns={"itemid": "ITEMID"})

    series["ITEMID"] = _clean_string_series(series["ITEMID"])
    series = series[series["ITEMID"].notna()].copy()
    metadata["matched_rows"] = int(len(series))
    metadata["unmatched_rows"] = int(metadata["valid_rows"] - metadata["matched_rows"])
    metadata["distinct_itemids"] = int(series["ITEMID"].nunique())

    n_rows = series.groupby("ITEMID", dropna=False).size().rename("n_obs").reset_index()
    item_level = series.drop_duplicates(subset=["ITEMID"], keep="first")[["ITEMID", "elasticity"]].copy()
    item_level = item_level.merge(n_rows, on="ITEMID", how="left")

    metadata["duplicate_itemids"] = int((n_rows["n_obs"] > 1).sum())

    item_level["elasticity_mean"] = item_level["elasticity"]
    item_level["elasticity_std"] = pd.NA
    item_level["elasticity_min"] = item_level["elasticity"]
    item_level["elasticity_max"] = item_level["elasticity"]
    item_level["elasticity_latest"] = item_level["elasticity"]
    item_level["elasticity_method"] = "direct"
    item_level = item_level[
        [
            "ITEMID",
            "elasticity",
            "elasticity_mean",
            "elasticity_std",
            "elasticity_min",
            "elasticity_max",
            "elasticity_latest",
            "n_obs",
            "elasticity_method",
        ]
    ]

    item_level_path = output_dir / "elasticity_by_itemid.csv"
    item_level.to_csv(item_level_path, index=False)
    metadata["item_level_path"] = str(item_level_path)

    return item_level, metadata