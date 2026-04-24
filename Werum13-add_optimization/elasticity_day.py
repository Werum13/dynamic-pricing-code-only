from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELASTICITY_DIR = PROJECT_ROOT / "Elasticity"
for p in [PROJECT_ROOT, ELASTICITY_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from DataPreprocessor import preprocessor  # noqa: E402
from ETL import etl_with_demand_target  # noqa: E402
from DemandModel import demand_model  # noqa: E402
from data_sources import load_elasticity_source_data  # noqa: E402

MIN_TOTAL_ROWS = 20
MIN_TRAIN_ROWS = 10
DEFAULT_PRICE_BINS = 10

TARGET_AND_FUTURE_COLS = [
    "DATE_",
    "CATEGORY1",
    "CATEGORY2",
    "GMV_1D",
    "GMV_7D",
    "GMV_15D",
    "GMV_30D",
    "AMOUNT_0D_target",
    "AMOUNT_1D_target",
    "AMOUNT_7D_target",
    "AMOUNT_15D_target",
    "AMOUNT_30D_target",
    "AMOUNT_1D",
    "AMOUNT_7D",
    "AMOUNT_15D",
    "AMOUNT_30D",
]


def _make_future_row(day: pd.Timestamp, itemcode: int, df_hist: pd.DataFrame) -> pd.DataFrame:
    base = df_hist.sort_values("DATE_").tail(1).iloc[0].to_dict()
    base["DATE_"] = pd.to_datetime(day)
    base["ITEMCODE"] = itemcode
    for col in ["GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D"]:
        base[col] = np.nan
    for col in ["AMOUNT_1D", "TOTALPRICE_1D"]:
        base[col] = 0
    return pd.DataFrame([base])


def _prepare_binned_daily_history(
    hist: pd.DataFrame,
    n_bins: int = DEFAULT_PRICE_BINS,
    min_unique_prices: int = 2,
) -> pd.DataFrame:
    daily_input = hist.copy()
    if daily_input.empty:
        return daily_input

    daily_input["UNITPRICE"] = pd.to_numeric(daily_input["UNITPRICE"], errors="coerce")
    daily_input["AMOUNT"] = pd.to_numeric(daily_input["AMOUNT"], errors="coerce")
    daily_input = daily_input.dropna(subset=["DATE_", "UNITPRICE", "AMOUNT"])
    daily_input = daily_input[daily_input["AMOUNT"] > 0].copy()
    if daily_input.empty:
        return daily_input

    unique_prices = daily_input["UNITPRICE"].nunique(dropna=True)
    if unique_prices >= min_unique_prices:
        bins = np.linspace(daily_input["UNITPRICE"].min(), daily_input["UNITPRICE"].max(), n_bins + 1)
        bins = np.unique(bins)
        if len(bins) > 2:
            labels = (bins[:-1] + bins[1:]) / 2
            binned = pd.cut(
                daily_input["UNITPRICE"],
                bins=bins,
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            )
            daily_input["UNITPRICE"] = pd.to_numeric(binned, errors="coerce").fillna(daily_input["UNITPRICE"])

    daily_input["TOTALPRICE"] = daily_input["UNITPRICE"] * daily_input["AMOUNT"]

    def _weighted_price(group: pd.DataFrame) -> float:
        weights = pd.to_numeric(group["AMOUNT"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        prices = pd.to_numeric(group["UNITPRICE"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.all(weights <= 0):
            return float(np.nanmean(prices))
        return float(np.average(prices, weights=weights))

    agg_map: dict[str, str | Any] = {
        "UNITPRICE": _weighted_price,
        "AMOUNT": "sum",
        "TOTALPRICE": "sum",
    }
    for col in ["CATEGORY1", "CATEGORY2", "BASEPRICE", "cost"]:
        if col in daily_input.columns:
            agg_map[col] = "first"

    daily = (
        daily_input.sort_values("DATE_")
        .groupby(["ITEMCODE", "DATE_"], as_index=False)
        .agg(agg_map)
        .sort_values("DATE_")
        .reset_index(drop=True)
    )
    return daily


def _normalize_prices(price: float | None, prices: list[float] | None) -> list[float]:
    out: list[float] = []
    if prices:
        out.extend(prices)
    if price is not None:
        out.append(float(price))
    normalized: list[float] = []
    for value in out:
        value_num = float(value)
        if np.isfinite(value_num) and value_num > 0:
            normalized.append(value_num)
    return list(dict.fromkeys(normalized))


def _load_item_source(item_code: int) -> pd.DataFrame:
    df = load_elasticity_source_data(
        PROJECT_ROOT,
        usecols=["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"],
    )
    df["DATE_"] = pd.to_datetime(df["DATE_"])
    df = df[df["ITEMCODE"].astype("int64") == int(item_code)].copy()
    if df.empty:
        return df
    cost_path = PROJECT_ROOT / "data" / "cost.csv"
    if cost_path.exists():
        cost_df = pd.read_csv(cost_path)[["ITEMCODE", "cost"]]
        df = df.merge(cost_df, how="left", on="ITEMCODE")
    else:
        df["cost"] = np.nan
    return df


def estimate_item_elasticity_for_day(
    item_code: int,
    day: pd.Timestamp,
    price: float | None = None,
    prices: list[float] | None = None,
    window_days: int = 30,
) -> dict[str, Any]:
    day = pd.Timestamp(day).normalize()
    requested_prices = _normalize_prices(price, prices)
    df_item = _load_item_source(item_code)
    if df_item.empty:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": 0.0,
            "n_obs": 0,
            "method": "no_item_data",
            "elasticity_by_price": [],
        }

    df_pre = preprocessor(df_item.copy())
    if len(df_pre) < MIN_TOTAL_ROWS:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": float(df_pre["AMOUNT"].mean()) if len(df_pre) else 0.0,
            "n_obs": int(len(df_pre)),
            "method": "insufficient_preprocessed_data",
            "elasticity_by_price": [],
        }

    hist = df_pre[df_pre["DATE_"] < day].copy()
    if hist.empty:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": 0.0,
            "n_obs": 0,
            "method": "no_history_before_day",
            "elasticity_by_price": [],
        }

    hist_daily = _prepare_binned_daily_history(hist)
    if hist_daily.empty:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": 0.0,
            "n_obs": 0,
            "method": "empty_binned_history",
            "elasticity_by_price": [],
        }

    frame_for_etl = pd.concat([hist_daily, _make_future_row(day, int(item_code), hist_daily)], ignore_index=True)
    frame_for_etl = frame_for_etl.sort_values("DATE_").reset_index(drop=True)
    df_etl = etl_with_demand_target(frame_for_etl)
    if df_etl.empty:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": float(hist_daily["AMOUNT"].mean()),
            "n_obs": int(len(hist_daily)),
            "method": "etl_empty",
            "elasticity_by_price": [],
        }

    train_full = df_etl[df_etl["DATE_"] < day].copy()
    target = train_full["AMOUNT_0D_target"].copy()
    train_df = train_full.drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore")
    valid = train_df.notna().all(axis=1) & target.notna()
    train_df = train_df[valid]
    target = target[valid]

    if len(train_df) < MIN_TRAIN_ROWS:
        return {
            "elasticity": None,
            "baseprice": price or 0.0,
            "avg_qty": float(hist_daily["AMOUNT"].tail(window_days).mean()),
            "n_obs": int(len(train_df)),
            "method": "insufficient_train_rows",
            "elasticity_by_price": [],
        }

    model = demand_model(train=train_df, target=target)

    template = df_etl[df_etl["DATE_"] == day].tail(1).drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore").copy()
    if template.empty:
        template = train_full.tail(1).drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore").copy()

    ref_price = float(price) if price is not None else float(hist_daily.tail(window_days)["UNITPRICE"].median())
    if not np.isfinite(ref_price) or ref_price <= 0:
        ref_price = float(hist_daily["UNITPRICE"].median())
    if not np.isfinite(ref_price) or ref_price <= 0:
        ref_price = 1.0

    template["UNITPRICE"] = ref_price
    template["BASEPRICE"] = ref_price
    template["Id"] = 0
    if "cost" not in template.columns:
        template["cost"] = float(hist_daily["cost"].dropna().median()) if "cost" in hist_daily.columns and hist_daily["cost"].notna().any() else ref_price * 0.6

    for col in model.columns_:
        if col not in template.columns:
            template[col] = 0
    template = template.fillna(0)

    price_points = requested_prices or [ref_price]
    elasticity_by_price: list[dict[str, float | None]] = []
    fallback_qty = float(hist_daily.tail(window_days)["AMOUNT"].mean()) if len(hist_daily) > 0 else 0.0
    for eval_price in price_points:
        point_df = template.copy()
        point_df["UNITPRICE"] = float(eval_price)
        point_df["BASEPRICE"] = float(eval_price)
        point_df = point_df.fillna(0)
        eps_value = float(model.elasticity(point_df)[0])
        if not np.isfinite(eps_value):
            eps_value = None
        pred_qty = float(model.predict(point_df)["pred_Q"].iloc[0])
        if not np.isfinite(pred_qty):
            pred_qty = fallback_qty
        elasticity_by_price.append(
            {
                "price": float(eval_price),
                "elasticity": eps_value,
                "predicted_qty": max(pred_qty, 0.0),
            }
        )

    primary_point = next((row for row in elasticity_by_price if abs(row["price"] - ref_price) < 1e-9), elasticity_by_price[0])
    eps = primary_point["elasticity"]
    pred_qty = float(primary_point["predicted_qty"]) if primary_point["predicted_qty"] is not None else fallback_qty
    avg_qty = float(hist_daily.tail(window_days)["AMOUNT"].mean()) if len(hist_daily) > 0 else max(pred_qty, 0.0)

    if eps is not None and not np.isfinite(eps):
        eps = None

    return {
        "elasticity": eps,
        "baseprice": ref_price,
        "avg_qty": max(avg_qty, 0.0),
        "n_obs": int(len(train_df)),
        "method": "demand_model_point",
        "elasticity_by_price": elasticity_by_price,
    }


def estimate_family_elasticity_for_day(
    family: dict[str, Any],
    date: pd.Timestamp,
    hypothetical_prices: dict[int, float],
    ctx: dict[str, Any],
    window_days: int = 30,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item_code in family["all_items"]:
        info = estimate_item_elasticity_for_day(
            item_code=item_code,
            day=date,
            price=hypothetical_prices.get(item_code),
            window_days=window_days,
        )
        if info["elasticity"] is None and item_code in ctx["elast_base"]:
            base_price, base_elast = ctx["elast_base"][item_code]
            info["elasticity"] = float(base_elast)
            info["baseprice"] = float(base_price)
            info["method"] = "fallback_baseprice"
        if info["elasticity"] is None:
            info["elasticity"] = -1.2
            info["method"] = "fallback_default"
            if info.get("baseprice", 0.0) <= 0:
                info["baseprice"] = float(hypothetical_prices.get(item_code, 1.0))
        result[item_code] = info
    return result
