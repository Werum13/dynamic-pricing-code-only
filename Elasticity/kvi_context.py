from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
KVI_OUTPUT_DIR = OUTPUT_DIR / "kvi"

COST_PATH = DATA_DIR / "cost.csv"

log = logging.getLogger("kvi_context")


def _resolve_optional_artifact(filename: str) -> Path | None:
    # Prefer output/kvi (current KVI pipeline), fallback to legacy output/ root.
    for base_dir in (KVI_OUTPUT_DIR, OUTPUT_DIR):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return None


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

    kvi_scores_path = _resolve_optional_artifact("kvi_scores_full.csv")
    kvi_candidates_path = _resolve_optional_artifact("kvi_candidates.csv")
    substitute_map_path = _resolve_optional_artifact("substitute_map.json")
    behavioral_path = _resolve_optional_artifact("behavioral_features.csv")
    elasticity_base_path = _resolve_optional_artifact("elasticity_by_itemid.csv")

    if kvi_scores_path is not None:
        df = _normalize_itemcode(pd.read_csv(kvi_scores_path))
        ctx["kvi_scores"] = df
    else:
        ctx["kvi_scores"] = pd.DataFrame()

    if kvi_candidates_path is not None:
        df_kvi = _normalize_itemcode(pd.read_csv(kvi_candidates_path))
        ctx["kvi_set"] = set(df_kvi["ITEMCODE"].dropna().astype(int))
    else:
        ctx["kvi_set"] = set()

    if substitute_map_path is not None:
        with open(substitute_map_path, encoding="utf-8") as f:
            raw = json.load(f)
        # Normalize keys once so consumers can always use int keys.
        ctx["sub_map"] = {int(k): v for k, v in raw.items()}
    else:
        ctx["sub_map"] = {}

    if behavioral_path is not None:
        ctx["behavioral"] = _normalize_itemcode(pd.read_csv(behavioral_path))
    else:
        ctx["behavioral"] = pd.DataFrame()

    if COST_PATH.exists():
        df_c = _normalize_itemcode(pd.read_csv(COST_PATH))
        ctx["cost_map"] = dict(zip(df_c["ITEMCODE"].astype(int), df_c["cost"]))
    else:
        ctx["cost_map"] = {}

    if elasticity_base_path is not None:
        df_e = _normalize_itemcode(pd.read_csv(elasticity_base_path))
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
