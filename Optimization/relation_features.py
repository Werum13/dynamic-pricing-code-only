from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_div(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full(len(num), fill_value, dtype=float)
    valid = np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = num[valid] / den[valid]
    return pd.Series(out, index=numerator.index)


def build_relation_features(config: dict[str, Any], logger: logging.Logger) -> dict[str, str]:
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    manifests_dir = output_dir / "manifests"
    input_path = tables_dir / "item_day_dataset.csv"
    output_path = tables_dir / "relation_features.csv"
    manifest_path = manifests_dir / "relation_features_manifest.json"

    df = pd.read_csv(input_path, parse_dates=["DATE_"])

    category_day = (
        df.groupby(["DATE_", "CATEGORY2"], as_index=False)
        .agg(
            category_price_mean=("current_price", "mean"),
            category_price_std=("current_price", "std"),
            category_active_items=("ITEMCODE", "nunique"),
            category_revenue_share_total=("revenue_share_28d", "sum"),
        )
    )
    category_day["category_price_std"] = category_day["category_price_std"].fillna(0.0)

    df = df.merge(category_day, on=["DATE_", "CATEGORY2"], how="left")
    df["relative_price_gap"] = _safe_div(
        df["current_price"] - df["category_price_mean"],
        df["category_price_mean"],
        fill_value=0.0,
    )
    df["substitute_pressure"] = (
        df["relative_price_gap"].clip(lower=0.0)
        * np.log1p(df["category_active_items"].fillna(0.0))
        * (1.0 + df["price_change_share_28d"].fillna(0.0))
    )
    df["cannibal_risk"] = (
        df["revenue_share_28d"].fillna(0.0)
        * (-df["relative_price_gap"]).clip(lower=0.0)
        * (1.0 + df["is_anchor"].fillna(0.0))
    )
    df["complement_support"] = 0.0
    df["category_competition_proxy"] = _safe_div(
        df["category_price_std"],
        df["category_price_mean"],
        fill_value=0.0,
    ) * np.log1p(df["category_active_items"].fillna(0.0))

    output_cols = [
        "DATE_",
        "ITEMCODE",
        "ITEMID",
        "CATEGORY1",
        "CATEGORY2",
        "CATEGORY1_CODE",
        "CATEGORY2_CODE",
        "ITEMID_CODE",
        "daily_amount",
        "daily_revenue",
        "current_price",
        "base_price_raw",
        "base_price_56d",
        "cost",
        "elasticity",
        "units_7d",
        "units_28d",
        "revenue_7d",
        "revenue_28d",
        "price_mean_28d",
        "price_std_28d",
        "price_cv_28d",
        "demand_mean_28d",
        "demand_std_28d",
        "active_days_28d",
        "price_change_share_28d",
        "discount_to_base",
        "gross_margin_pct",
        "future_amount_7d",
        "future_revenue_7d",
        "category_future_revenue_7d",
        "revenue_share_28d",
        "units_share_28d",
        "future_basket_share",
        "kvi_score",
        "is_kvi",
        "is_anchor",
        "dow",
        "month",
        "weekofyear",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "substitute_pressure",
        "cannibal_risk",
        "complement_support",
        "category_competition_proxy",
        "relative_price_gap",
    ]
    relation_df = df[output_cols].copy()
    relation_df.to_csv(output_path, index=False)

    manifest = {
        "rows": int(len(relation_df)),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "complement_strategy": "zero_filled",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved relation features with %s rows", len(relation_df))

    return {
        "relation_features": str(output_path),
        "relation_features_manifest": str(manifest_path),
    }
