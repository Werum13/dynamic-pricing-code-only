from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _policy_summary(df: pd.DataFrame, policy_name: str) -> dict[str, Any]:
    if df.empty:
        return {
            "policy": policy_name,
            "rows": 0,
            "total_margin": 0.0,
            "total_revenue": 0.0,
            "total_qty": 0.0,
            "avg_penetration_drift": 0.0,
            "avg_basket_share_drift": 0.0,
            "avg_abs_price_change": 0.0,
            "kvi_avg_abs_price_change": 0.0,
            "anchor_avg_abs_price_change": 0.0,
            "guardrail_violation_rate": 0.0,
        }

    return {
        "policy": policy_name,
        "rows": int(len(df)),
        "total_margin": float(df["pseudo_margin_7d"].sum()),
        "total_revenue": float(df["pseudo_revenue_7d"].sum()),
        "total_qty": float(df["pseudo_qty_7d"].sum()),
        "avg_penetration_drift": float(df["pseudo_penetration_drift"].mean()),
        "avg_basket_share_drift": float(df["pseudo_basket_share_drift"].mean()),
        "avg_abs_price_change": float(df["price_delta_current_pct"].abs().mean()),
        "kvi_avg_abs_price_change": float(df.loc[df["is_kvi"] == 1, "price_delta_current_pct"].abs().mean() if (df["is_kvi"] == 1).any() else 0.0),
        "anchor_avg_abs_price_change": float(df.loc[df["is_anchor"] == 1, "price_delta_current_pct"].abs().mean() if (df["is_anchor"] == 1).any() else 0.0),
        "guardrail_violation_rate": float((df["scenario_is_admissible"] == 0).mean()),
    }


def evaluate_policy(config: dict[str, Any], logger: logging.Logger) -> dict[str, str]:
    output_dir = Path(config["paths"]["output_dir"])
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    manifests_dir = output_dir / "manifests"

    scored = pd.read_csv(tables_dir / "scenario_scored.csv", parse_dates=["DATE_"])
    predictions = pd.read_csv(reports_dir / "rolling_model_predictions.csv", parse_dates=["DATE_"])
    if predictions.empty:
        raise RuntimeError("Rolling model predictions are empty; cannot run policy evaluation")

    eval_df = scored.merge(
        predictions[["scenario_id", "fold", "ml_pred_utility"]],
        on="scenario_id",
        how="inner",
    )
    admissible = eval_df[eval_df["scenario_is_admissible"] == 1].copy()

    ml_choice = (
        admissible.sort_values(["ITEMCODE", "DATE_", "ml_pred_utility"], ascending=[True, True, False])
        .drop_duplicates(["ITEMCODE", "DATE_"], keep="first")
        .copy()
    )
    fallback_choice = (
        admissible.sort_values(["ITEMCODE", "DATE_", "fallback_utility"], ascending=[True, True, False])
        .drop_duplicates(["ITEMCODE", "DATE_"], keep="first")
        .copy()
    )
    current_choice = (
        admissible[admissible["is_current_price_scenario"] == 1]
        .sort_values(["ITEMCODE", "DATE_"])
        .drop_duplicates(["ITEMCODE", "DATE_"], keep="first")
        .copy()
    )

    ml_summary = _policy_summary(ml_choice, "ml_policy")
    fallback_summary = _policy_summary(fallback_choice, "elasticity_fallback")
    current_summary = _policy_summary(current_choice, "current_price")

    comparison_df = pd.DataFrame([ml_summary, fallback_summary, current_summary])
    comparison_path = reports_dir / "fallback_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    rolling_backtest = (
        ml_choice[["fold", "ITEMCODE", "DATE_", "candidate_price", "price_delta_current_pct", "pseudo_margin_7d", "pseudo_revenue_7d", "pseudo_qty_7d", "pseudo_penetration_drift", "pseudo_basket_share_drift", "is_kvi", "is_anchor"]]
        .rename(columns={"candidate_price": "ml_candidate_price"})
        .merge(
            fallback_choice[["ITEMCODE", "DATE_", "candidate_price", "pseudo_margin_7d", "pseudo_revenue_7d", "pseudo_qty_7d"]],
            on=["ITEMCODE", "DATE_"],
            how="left",
            suffixes=("", "_fallback"),
        )
        .merge(
            current_choice[["ITEMCODE", "DATE_", "candidate_price", "pseudo_margin_7d", "pseudo_revenue_7d", "pseudo_qty_7d"]],
            on=["ITEMCODE", "DATE_"],
            how="left",
            suffixes=("", "_current"),
        )
    )
    rolling_backtest_path = reports_dir / "rolling_backtest.csv"
    rolling_backtest.to_csv(rolling_backtest_path, index=False)

    guardrails = ml_choice.loc[
        (ml_choice["is_kvi"] == 1) | (ml_choice["is_anchor"] == 1),
        [
            "ITEMCODE",
            "DATE_",
            "candidate_price",
            "price_delta_current_pct",
            "is_kvi",
            "is_anchor",
            "scenario_is_admissible",
            "passes_change_guardrail",
            "passes_margin_guardrail",
            "passes_price_floor_guardrail",
        ],
    ].copy()
    guardrails["guardrail_issue"] = (
        (guardrails["passes_change_guardrail"] == 0)
        | (guardrails["passes_margin_guardrail"] == 0)
        | (guardrails["passes_price_floor_guardrail"] == 0)
    ).astype(int)
    guardrails_path = reports_dir / "guardrails_violations.csv"
    guardrails.to_csv(guardrails_path, index=False)

    ml_margin = ml_summary["total_margin"]
    fallback_margin = fallback_summary["total_margin"]
    current_margin = current_summary["total_margin"]
    margin_uplift_vs_fallback = (ml_margin - fallback_margin) / max(abs(fallback_margin), 1.0)
    margin_uplift_vs_current = (ml_margin - current_margin) / max(abs(current_margin), 1.0)
    penetration_drift = ml_summary["avg_penetration_drift"]
    basket_drift = ml_summary["avg_basket_share_drift"]
    guardrail_issue_rate = float(guardrails["guardrail_issue"].mean()) if not guardrails.empty else 0.0

    if margin_uplift_vs_fallback > 0.02 and guardrail_issue_rate <= 0.01 and abs(penetration_drift) <= 0.15 and abs(basket_drift) <= 0.05:
        verdict = "promising"
    elif margin_uplift_vs_fallback >= -0.01 and guardrail_issue_rate <= 0.02:
        verdict = "safe fallback only"
    else:
        verdict = "reject"

    summary = {
        "verdict": verdict,
        "margin_uplift_vs_fallback": margin_uplift_vs_fallback,
        "margin_uplift_vs_current": margin_uplift_vs_current,
        "avg_penetration_drift": penetration_drift,
        "avg_basket_share_drift": basket_drift,
        "guardrail_issue_rate": guardrail_issue_rate,
        "policies": [ml_summary, fallback_summary, current_summary],
    }
    summary_path = reports_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_path = manifests_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "rolling_backtest": str(rolling_backtest_path),
                "comparison": str(comparison_path),
                "guardrails": str(guardrails_path),
                "summary": str(summary_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Policy evaluation verdict: %s", verdict)

    return {
        "rolling_backtest": str(rolling_backtest_path),
        "fallback_comparison": str(comparison_path),
        "guardrails_violations": str(guardrails_path),
        "summary": str(summary_path),
        "evaluation_manifest": str(manifest_path),
    }
