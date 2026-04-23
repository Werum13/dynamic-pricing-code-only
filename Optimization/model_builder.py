from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("LightGBM is required. Install it with: python -m pip install lightgbm") from exc


FEATURE_COLUMNS = [
    "ITEMCODE",
    "ITEMID_CODE",
    "CATEGORY1_CODE",
    "CATEGORY2_CODE",
    "current_price",
    "candidate_price",
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
    "price_delta_current_pct",
    "price_delta_base_pct",
    "candidate_margin_pct",
    "candidate_price_to_cost",
    "candidate_rel_price_gap",
    "passes_change_guardrail",
    "passes_margin_guardrail",
    "passes_price_floor_guardrail",
    "scenario_is_admissible",
    "is_current_price_scenario",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full(len(num), fill_value, dtype=float)
    valid = np.isfinite(den) & (np.abs(den) > 1e-12)
    out[valid] = num[valid] / den[valid]
    return pd.Series(out, index=numerator.index)


def _add_targets(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    weights = config["model"]["utility_weights"]
    min_margin_pct = float(config["scenarios"]["min_margin_pct"])

    scored = df.copy()
    scored["elasticity_for_target"] = scored["elasticity"].clip(-3.5, -0.05)
    current_price = scored["current_price"].clip(lower=0.01)
    price_ratio = (scored["candidate_price"] / current_price).clip(lower=0.5, upper=1.5)
    elasticity_effect = np.exp(scored["elasticity_for_target"] * np.log(price_ratio))
    relation_effect = np.exp(
        -0.15 * scored["substitute_pressure"].clip(lower=0.0) * scored["price_delta_current_pct"].clip(lower=0.0)
        -0.10 * scored["cannibal_risk"].clip(lower=0.0) * (-scored["price_delta_current_pct"]).clip(lower=0.0)
        + 0.02 * scored["complement_support"].fillna(0.0)
    )
    scored["pseudo_qty_7d"] = (scored["future_amount_7d"].clip(lower=0.0) * elasticity_effect * relation_effect).clip(lower=0.0)
    scored["pseudo_revenue_7d"] = scored["candidate_price"] * scored["pseudo_qty_7d"]
    scored["pseudo_margin_7d"] = (scored["candidate_price"] - scored["cost"]) * scored["pseudo_qty_7d"]
    scored["pseudo_penetration_drift"] = _safe_div(
        scored["pseudo_qty_7d"] - scored["future_amount_7d"],
        scored["future_amount_7d"].replace(0.0, np.nan),
        fill_value=0.0,
    )
    scored["pseudo_basket_share"] = _safe_div(
        scored["pseudo_revenue_7d"],
        scored["category_future_revenue_7d"].replace(0.0, np.nan),
        fill_value=0.0,
    )
    scored["pseudo_basket_share_drift"] = scored["pseudo_basket_share"] - scored["future_basket_share"].fillna(0.0)

    revenue_scale = scored["future_revenue_7d"].clip(lower=1.0)
    qty_scale = scored["future_amount_7d"].clip(lower=1.0)
    guardrail_penalty = (1 - scored["scenario_is_admissible"]) * 10.0
    margin_penalty = (min_margin_pct - scored["candidate_margin_pct"]).clip(lower=0.0) * 5.0
    kvi_penalty = (
        scored["price_delta_current_pct"].abs()
        * (1.0 + scored["kvi_score"].fillna(0.0) * weights["kvi_penalty"])
    )
    cannibal_penalty = weights["cannibal_penalty"] * scored["cannibal_risk"].fillna(0.0) * (-scored["price_delta_current_pct"]).clip(lower=0.0)

    scored["target_utility"] = (
        weights["margin"] * (scored["pseudo_margin_7d"] / revenue_scale)
        + weights["revenue"] * (scored["pseudo_revenue_7d"] / revenue_scale)
        + weights["volume"] * (scored["pseudo_qty_7d"] / qty_scale)
        - weights["penetration_penalty"] * scored["pseudo_penetration_drift"].abs()
        - weights["basket_penalty"] * scored["pseudo_basket_share_drift"].abs()
        - kvi_penalty
        - margin_penalty
        - cannibal_penalty
        - guardrail_penalty
    )

    base_qty = np.maximum(scored["units_7d"], scored["demand_mean_28d"] * 7.0).clip(lower=1.0)
    fallback_qty = (base_qty * elasticity_effect).clip(lower=0.0)
    fallback_revenue = scored["candidate_price"] * fallback_qty
    fallback_margin = (scored["candidate_price"] - scored["cost"]) * fallback_qty
    scored["fallback_utility"] = (
        weights["margin"] * _safe_div(fallback_margin, revenue_scale, fill_value=0.0)
        + weights["revenue"] * _safe_div(fallback_revenue, revenue_scale, fill_value=0.0)
        + weights["volume"] * _safe_div(fallback_qty, qty_scale, fill_value=0.0)
        - 0.5 * kvi_penalty
        - margin_penalty
        - guardrail_penalty
    )
    return scored


def _build_rolling_splits(df: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, pd.Timestamp]]:
    eval_cfg = config["evaluation"]
    unique_dates = pd.Series(pd.to_datetime(df["DATE_"]).dt.normalize().unique()).sort_values()
    if unique_dates.empty:
        return []

    train_days = int(eval_cfg["train_days"])
    test_days = int(eval_cfg["test_days"])
    step_days = int(eval_cfg["step_days"])

    min_date = unique_dates.min()
    max_date = unique_dates.max()
    splits: list[dict[str, pd.Timestamp]] = []

    train_start = min_date
    while True:
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)
        if test_end > max_date:
            break
        splits.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_start = train_start + pd.Timedelta(days=step_days)
    return splits


def build_model(config: dict[str, Any], logger: logging.Logger) -> dict[str, str]:
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    manifests_dir = output_dir / "manifests"
    scenario_path = tables_dir / "scenario_grid.csv"

    scenario_df = pd.read_csv(scenario_path, parse_dates=["DATE_"])
    scored = _add_targets(scenario_df, config)
    scored_path = tables_dir / "scenario_scored.csv"
    scored.to_csv(scored_path, index=False)

    model_params = dict(config["model"]["params"])
    model_params["random_state"] = int(config["model"]["random_state"])
    feature_cols = FEATURE_COLUMNS.copy()
    scored[feature_cols] = scored[feature_cols].fillna(0.0)

    candidate_rows = scored[(scored["scenario_is_admissible"] == 1) & scored["target_utility"].notna()].copy()
    splits = _build_rolling_splits(candidate_rows, config)
    min_train_rows = int(config["evaluation"]["min_train_rows"])

    prediction_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    feature_importance_parts: list[pd.DataFrame] = []

    for fold_idx, split in enumerate(splits, start=1):
        train_mask = (candidate_rows["DATE_"] >= split["train_start"]) & (candidate_rows["DATE_"] <= split["train_end"])
        test_mask = (candidate_rows["DATE_"] >= split["test_start"]) & (candidate_rows["DATE_"] <= split["test_end"])
        train_df = candidate_rows.loc[train_mask].copy()
        test_df = candidate_rows.loc[test_mask].copy()
        if len(train_df) < min_train_rows or test_df.empty:
            continue

        model = LGBMRegressor(**model_params)
        model.fit(train_df[feature_cols], train_df["target_utility"])
        test_df["ml_pred_utility"] = model.predict(test_df[feature_cols])
        prediction_parts.append(
            test_df[
                [
                    "scenario_id",
                    "ITEMCODE",
                    "DATE_",
                    "candidate_price",
                    "scenario_delta",
                    "scenario_is_admissible",
                    "target_utility",
                    "fallback_utility",
                    "ml_pred_utility",
                    "pseudo_margin_7d",
                    "pseudo_revenue_7d",
                    "pseudo_qty_7d",
                    "pseudo_penetration_drift",
                    "pseudo_basket_share_drift",
                    "is_kvi",
                    "is_anchor",
                ]
            ].assign(fold=fold_idx)
        )

        mae = float((test_df["ml_pred_utility"] - test_df["target_utility"]).abs().mean())
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_start": str(split["train_start"].date()),
                "train_end": str(split["train_end"].date()),
                "test_start": str(split["test_start"].date()),
                "test_end": str(split["test_end"].date()),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "mae": mae,
            }
        )
        feature_importance_parts.append(
            pd.DataFrame(
                {
                    "fold": fold_idx,
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            )
        )

    final_model = LGBMRegressor(**model_params)
    final_model.fit(candidate_rows[feature_cols], candidate_rows["target_utility"])

    model_path = models_dir / "policy_model.pkl"
    schema_path = models_dir / "feature_schema.json"
    pred_path = reports_dir / "rolling_model_predictions.csv"
    metrics_path = reports_dir / "rolling_model_metrics.csv"
    importance_path = reports_dir / "feature_importance.csv"
    manifest_path = manifests_dir / "model_manifest.json"

    with open(model_path, "wb") as handle:
        pickle.dump(final_model, handle)

    schema = {
        "model_name": config["model"]["name"],
        "feature_columns": feature_cols,
        "random_state": config["model"]["random_state"],
        "scenario_scored_path": str(scored_path),
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    predictions.to_csv(pred_path, index=False)
    pd.DataFrame(fold_metrics).to_csv(metrics_path, index=False)

    if feature_importance_parts:
        importance = pd.concat(feature_importance_parts, ignore_index=True)
        importance = (
            importance.groupby("feature", as_index=False)["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "importance_mean", "std": "importance_std"})
            .sort_values("importance_mean", ascending=False)
        )
        importance.to_csv(importance_path, index=False)
    else:
        pd.DataFrame(columns=["feature", "importance_mean", "importance_std"]).to_csv(importance_path, index=False)

    manifest = {
        "candidate_rows": int(len(candidate_rows)),
        "folds_built": int(len(fold_metrics)),
        "model_path": str(model_path),
        "feature_schema": str(schema_path),
        "predictions_path": str(pred_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved ML model and %s rolling folds", len(fold_metrics))

    return {
        "scenario_scored": str(scored_path),
        "policy_model": str(model_path),
        "feature_schema": str(schema_path),
        "rolling_predictions": str(pred_path),
        "rolling_metrics": str(metrics_path),
        "feature_importance": str(importance_path),
        "model_manifest": str(manifest_path),
    }
