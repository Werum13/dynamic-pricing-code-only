from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series, fill_value: float = 0.0) -> pd.Series:
    numerator = num.to_numpy(dtype=float)
    denominator = den.to_numpy(dtype=float)
    out = np.full(len(numerator), fill_value, dtype=float)
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    out[valid] = numerator[valid] / denominator[valid]
    return pd.Series(out, index=num.index)


def build_scenarios(config: dict[str, Any], logger: logging.Logger) -> dict[str, str]:
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    manifests_dir = output_dir / "manifests"
    relation_path = tables_dir / "relation_features.csv"
    output_path = tables_dir / "scenario_grid.csv"
    manifest_path = manifests_dir / "scenario_manifest.json"

    df = pd.read_csv(relation_path, parse_dates=["DATE_"])
    df = df[df["current_price"] > 0].copy()
    if df.empty:
        raise RuntimeError("Relation feature table does not contain valid priced rows")

    deltas = np.array(config["scenarios"]["candidate_deltas"], dtype=float)
    repeated = df.loc[df.index.repeat(len(deltas))].copy().reset_index(drop=True)
    repeated["scenario_delta"] = np.tile(deltas, len(df))
    repeated["candidate_price"] = repeated["current_price"] * (1.0 + repeated["scenario_delta"])
    repeated["price_delta_current_pct"] = repeated["scenario_delta"]
    repeated["price_delta_base_pct"] = _safe_div(
        repeated["candidate_price"] - repeated["base_price_56d"],
        repeated["base_price_56d"],
        fill_value=0.0,
    )
    repeated["candidate_margin_pct"] = _safe_div(
        repeated["candidate_price"] - repeated["cost"],
        repeated["candidate_price"],
        fill_value=-1.0,
    )
    repeated["candidate_price_to_cost"] = _safe_div(
        repeated["candidate_price"],
        repeated["cost"].replace(0.0, np.nan),
        fill_value=0.0,
    )
    repeated["candidate_rel_price_gap"] = _safe_div(
        repeated["candidate_price"] - repeated["price_mean_28d"],
        repeated["price_mean_28d"],
        fill_value=0.0,
    )

    max_abs_change_pct = float(config["scenarios"]["max_abs_change_pct"])
    kvi_change_cap = float(config["scenarios"]["kvi_max_abs_change_pct"])
    anchor_change_cap = float(config["scenarios"]["anchor_max_abs_change_pct"])
    min_margin_pct = float(config["scenarios"]["min_margin_pct"])

    repeated["allowed_abs_change_pct"] = max_abs_change_pct
    repeated.loc[repeated["is_kvi"] == 1, "allowed_abs_change_pct"] = kvi_change_cap
    repeated.loc[repeated["is_anchor"] == 1, "allowed_abs_change_pct"] = anchor_change_cap

    repeated["passes_change_guardrail"] = (
        repeated["price_delta_current_pct"].abs() <= repeated["allowed_abs_change_pct"] + 1e-12
    ).astype(int)
    repeated["passes_margin_guardrail"] = (repeated["candidate_margin_pct"] >= min_margin_pct).astype(int)
    repeated["passes_price_floor_guardrail"] = (
        repeated["candidate_price"] >= repeated["cost"] * (1.0 + min_margin_pct)
    ).astype(int)
    repeated["scenario_is_admissible"] = (
        repeated["passes_change_guardrail"]
        * repeated["passes_margin_guardrail"]
        * repeated["passes_price_floor_guardrail"]
    ).astype(int)
    repeated["is_current_price_scenario"] = (repeated["scenario_delta"].abs() < 1e-12).astype(int)
    repeated["scenario_id"] = (
        repeated["ITEMCODE"].astype(str)
        + "_"
        + repeated["DATE_"].dt.strftime("%Y-%m-%d")
        + "_"
        + repeated["scenario_delta"].map(lambda value: f"{value:+.2f}")
    )

    repeated.to_csv(output_path, index=False)
    manifest = {
        "rows": int(len(repeated)),
        "scenario_count_per_row": int(len(deltas)),
        "input_path": str(relation_path),
        "output_path": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved scenario grid with %s rows", len(repeated))

    return {
        "scenario_grid": str(output_path),
        "scenario_manifest": str(manifest_path),
    }
