from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

KVI_SCORES_PATH = OUTPUT_DIR / "kvi_scores_full.csv"
KVI_CANDIDATES_PATH = OUTPUT_DIR / "kvi_candidates.csv"
SUBSTITUTE_MAP_PATH = OUTPUT_DIR / "substitute_map.json"
BEHAVIORAL_PATH = OUTPUT_DIR / "behavioral_features.csv"
ELASTICITY_BASE_PATH = OUTPUT_DIR / "elasticity_by_itemid.csv"
COST_PATH = DATA_DIR / "cost.csv"

log = logging.getLogger("kvi_context")


def _normalize_itemcode(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.lower() in ("itemcode", "itemid", "item_code", "item_id"):
            if c != "ITEMCODE":
                df = df.rename(columns={c: "ITEMCODE"})
            break
    return df


def _get_col(row: pd.Series, candidates: list[str], default: Any = None) -> Any:
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


def load_kvi_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {}

    if KVI_SCORES_PATH.exists():
        df = _normalize_itemcode(pd.read_csv(KVI_SCORES_PATH))
        ctx["kvi_scores"] = df
    else:
        ctx["kvi_scores"] = pd.DataFrame()

    if KVI_CANDIDATES_PATH.exists():
        df_kvi = _normalize_itemcode(pd.read_csv(KVI_CANDIDATES_PATH))
        ctx["kvi_set"] = set(df_kvi["ITEMCODE"].dropna().astype(int))
    else:
        ctx["kvi_set"] = set()

    if SUBSTITUTE_MAP_PATH.exists():
        with open(SUBSTITUTE_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Normalize keys once so consumers can always use int keys.
        ctx["sub_map"] = {int(k): v for k, v in raw.items()}
    else:
        ctx["sub_map"] = {}

    if BEHAVIORAL_PATH.exists():
        ctx["behavioral"] = _normalize_itemcode(pd.read_csv(BEHAVIORAL_PATH))
    else:
        ctx["behavioral"] = pd.DataFrame()

    if COST_PATH.exists():
        df_c = _normalize_itemcode(pd.read_csv(COST_PATH))
        ctx["cost_map"] = dict(zip(df_c["ITEMCODE"].astype(int), df_c["cost"]))
    else:
        ctx["cost_map"] = {}

    if ELASTICITY_BASE_PATH.exists():
        df_e = _normalize_itemcode(pd.read_csv(ELASTICITY_BASE_PATH))
        elast_base = {}
        for _, row in df_e.iterrows():
            ic = int(row["ITEMCODE"])
            bp = _get_col(row, ["BASEPRICE", "baseprice", "base_price"], default=None)
            el = _get_col(row, ["elasticity", "elast", "Elasticity", "elasticity_value"], default=None)
            if bp is not None and el is not None:
                elast_base[ic] = (float(bp), float(el))
        ctx["elast_base"] = elast_base
    else:
        ctx["elast_base"] = {}

    return ctx


def get_item_family(item_code: int, sub_map: dict[int, Any]) -> dict[str, Any]:
    entry = sub_map.get(item_code, {})

    substitutes = [int(x) for x in entry.get("substitutes", [])]
    complements = [int(x) for x in entry.get("complements", [])]
    cannibals = [int(x) for x in entry.get("cannibals", [])]
    all_items = list(dict.fromkeys([item_code] + substitutes + complements + cannibals))

    family = {
        "target": item_code,
        "substitutes": substitutes,
        "complements": complements,
        "cannibals": cannibals,
        "all_items": all_items,
    }
    log.info("Resolved family for ITEMCODE=%s: %s", item_code, all_items)
    return family
