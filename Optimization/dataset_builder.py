from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_USECOLS = [
    "ITEMCODE",
    "ITEMID",
    "DATE_",
    "AMOUNT",
    "UNITPRICE",
    "TOTALPRICE",
    "CATEGORY1",
    "CATEGORY2",
]


def _safe_div(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, fill_value: float = 0.0):
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_arr, fill_value, dtype=float)
    valid = np.isfinite(denominator_arr) & (np.abs(denominator_arr) > 1e-12)
    result[valid] = numerator_arr[valid] / denominator_arr[valid]
    return result


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    df = chunk.copy()
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce").dt.normalize()
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0)
    df["BASEPRICE_RAW"] = _to_float(df["UNITPRICE"])
    df["TOTALPRICE"] = _to_float(df["TOTALPRICE"])
    df["UNITPRICE"] = np.where(
        df["AMOUNT"] > 0,
        df["TOTALPRICE"] / df["AMOUNT"],
        np.nan,
    )
    df["UNITPRICE"] = pd.to_numeric(df["UNITPRICE"], errors="coerce")
    df["ITEMCODE"] = pd.to_numeric(df["ITEMCODE"], errors="coerce")
    df["ITEMID"] = pd.to_numeric(df["ITEMID"], errors="coerce")
    df = df[df["ITEMCODE"].notna() & df["DATE_"].notna()].copy()
    df["ITEMCODE"] = df["ITEMCODE"].astype(int)
    df["ITEMID"] = df["ITEMID"].fillna(df["ITEMCODE"]).astype(int)
    df["CATEGORY1"] = df["CATEGORY1"].fillna("unknown").astype(str)
    df["CATEGORY2"] = df["CATEGORY2"].fillna("unknown").astype(str)
    df["UNITPRICE_X_AMOUNT"] = df["UNITPRICE"].fillna(0.0) * df["AMOUNT"].clip(lower=0.0)
    df["UNITPRICE_VALID"] = df["UNITPRICE"].notna().astype(int)
    df["BASEPRICE_X_AMOUNT"] = df["BASEPRICE_RAW"].fillna(0.0) * df["AMOUNT"].clip(lower=0.0)
    df["BASEPRICE_VALID"] = df["BASEPRICE_RAW"].notna().astype(int)
    return df


def _future_window_sum(series: pd.Series, horizon: int) -> pd.Series:
    values = series.fillna(0.0).to_numpy(dtype=float)
    csum = np.cumsum(np.r_[0.0, values])
    idx = np.arange(len(values))
    start = idx + 1
    end = np.minimum(len(values), idx + horizon + 1)
    return pd.Series(csum[end] - csum[start], index=series.index)


def _read_cost_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [col for col in df.columns if col in {"ITEMCODE", "cost"}]
    out = df[cols].copy()
    out["ITEMCODE"] = pd.to_numeric(out["ITEMCODE"], errors="coerce").astype("Int64")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out = out.dropna(subset=["ITEMCODE", "cost"]).copy()
    out["ITEMCODE"] = out["ITEMCODE"].astype(int)
    return out.drop_duplicates("ITEMCODE", keep="last")


def _scan_item_totals(
    data_path: Path,
    chunk_size: int,
    max_rows: int | None,
    min_item_amount: float,
    max_items: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    item_stats: dict[int, dict[str, Any]] = {}
    total_rows = 0
    global_min_date: pd.Timestamp | None = None
    global_max_date: pd.Timestamp | None = None

    for chunk in pd.read_csv(data_path, usecols=RAW_USECOLS, chunksize=chunk_size):
        if max_rows is not None and total_rows >= max_rows:
            break
        if max_rows is not None and total_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - total_rows].copy()
        total_rows += len(chunk)

        df = _prepare_chunk(chunk)
        if df.empty:
            continue

        current_min = df["DATE_"].min()
        current_max = df["DATE_"].max()
        global_min_date = current_min if global_min_date is None else min(global_min_date, current_min)
        global_max_date = current_max if global_max_date is None else max(global_max_date, current_max)

        grouped = (
            df.groupby("ITEMCODE", as_index=False)
            .agg(
                total_amount=("AMOUNT", "sum"),
                total_revenue=("TOTALPRICE", "sum"),
                first_date=("DATE_", "min"),
                last_date=("DATE_", "max"),
                itemid=("ITEMID", "last"),
                category1=("CATEGORY1", "last"),
                category2=("CATEGORY2", "last"),
            )
        )

        for row in grouped.itertuples(index=False):
            state = item_stats.setdefault(
                int(row.ITEMCODE),
                {
                    "ITEMCODE": int(row.ITEMCODE),
                    "ITEMID": int(row.itemid),
                    "CATEGORY1": str(row.category1),
                    "CATEGORY2": str(row.category2),
                    "total_amount": 0.0,
                    "total_revenue": 0.0,
                    "first_date": row.first_date,
                    "last_date": row.last_date,
                },
            )
            state["ITEMID"] = int(row.itemid)
            state["CATEGORY1"] = str(row.category1)
            state["CATEGORY2"] = str(row.category2)
            state["total_amount"] += float(row.total_amount)
            state["total_revenue"] += float(row.total_revenue)
            state["first_date"] = min(state["first_date"], row.first_date)
            state["last_date"] = max(state["last_date"], row.last_date)

    if not item_stats or global_min_date is None or global_max_date is None:
        raise RuntimeError("No valid rows were found in full_data.csv")

    metadata = pd.DataFrame(item_stats.values())
    metadata = metadata[metadata["total_amount"] >= float(min_item_amount)].copy()
    metadata = metadata.sort_values("total_revenue", ascending=False).head(max_items).reset_index(drop=True)
    metadata["selection_rank"] = np.arange(1, len(metadata) + 1)
    logger.info("Selected %s items for ML branch", len(metadata))
    return metadata, global_min_date.normalize(), global_max_date.normalize()


def _aggregate_item_day(
    data_path: Path,
    selected_items: set[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_size: int,
    max_rows: int | None,
) -> pd.DataFrame:
    grouped_parts: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(data_path, usecols=RAW_USECOLS, chunksize=chunk_size):
        if max_rows is not None and total_rows >= max_rows:
            break
        if max_rows is not None and total_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - total_rows].copy()
        total_rows += len(chunk)

        df = _prepare_chunk(chunk)
        if df.empty:
            continue

        mask = (
            df["ITEMCODE"].isin(selected_items)
            & (df["DATE_"] >= start_date)
            & (df["DATE_"] <= end_date)
        )
        df = df.loc[mask].copy()
        if df.empty:
            continue

        grouped = (
            df.groupby(["ITEMCODE", "DATE_"], as_index=False)
            .agg(
                ITEMID=("ITEMID", "last"),
                CATEGORY1=("CATEGORY1", "last"),
                CATEGORY2=("CATEGORY2", "last"),
                daily_amount=("AMOUNT", "sum"),
                daily_revenue=("TOTALPRICE", "sum"),
                unitprice_x_amount=("UNITPRICE_X_AMOUNT", "sum"),
                unitprice_obs=("UNITPRICE_VALID", "sum"),
                baseprice_x_amount=("BASEPRICE_X_AMOUNT", "sum"),
                baseprice_obs=("BASEPRICE_VALID", "sum"),
                txn_count=("AMOUNT", "size"),
            )
        )
        grouped_parts.append(grouped)

    if not grouped_parts:
        raise RuntimeError("No rows matched the selected item set and date window")

    item_day = pd.concat(grouped_parts, ignore_index=True)
    item_day = (
        item_day.groupby(["ITEMCODE", "DATE_"], as_index=False)
        .agg(
            ITEMID=("ITEMID", "last"),
            CATEGORY1=("CATEGORY1", "last"),
            CATEGORY2=("CATEGORY2", "last"),
            daily_amount=("daily_amount", "sum"),
            daily_revenue=("daily_revenue", "sum"),
            unitprice_x_amount=("unitprice_x_amount", "sum"),
            unitprice_obs=("unitprice_obs", "sum"),
            baseprice_x_amount=("baseprice_x_amount", "sum"),
            baseprice_obs=("baseprice_obs", "sum"),
            txn_count=("txn_count", "sum"),
        )
    )
    return item_day


def _read_elasticity_subset(
    elasticity_path: Path,
    selected_items: set[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_size: int,
) -> pd.DataFrame:
    if not elasticity_path.exists():
        return pd.DataFrame(columns=["DATE_", "ITEMCODE", "elasticity_raw"])

    header = pd.read_csv(elasticity_path, nrows=0)
    normalized_columns = {str(col).strip().lower(): col for col in header.columns}

    date_col = normalized_columns.get("date")
    item_col = normalized_columns.get("itemcode") or normalized_columns.get("item_code")
    elastic_col = normalized_columns.get("elastic") or normalized_columns.get("elasticity")

    if item_col is None or elastic_col is None:
        return pd.DataFrame(columns=["DATE_", "ITEMCODE", "elasticity_raw"])

    parts: list[pd.DataFrame] = []

    # Dated format: Date, ItemCode, Elastic
    if date_col is not None:
        usecols = [date_col, item_col, elastic_col]
        for chunk in pd.read_csv(elasticity_path, usecols=usecols, chunksize=chunk_size):
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce").dt.normalize()
            chunk[item_col] = pd.to_numeric(chunk[item_col], errors="coerce")
            chunk[elastic_col] = pd.to_numeric(chunk[elastic_col], errors="coerce")
            mask = (
                chunk[item_col].isin(selected_items)
                & (chunk[date_col] >= start_date)
                & (chunk[date_col] <= end_date)
            )
            subset = chunk.loc[mask, [date_col, item_col, elastic_col]].copy()
            if not subset.empty:
                subset[item_col] = subset[item_col].astype(int)
                parts.append(
                    subset.rename(
                        columns={
                            date_col: "DATE_",
                            item_col: "ITEMCODE",
                            elastic_col: "elasticity_raw",
                        }
                    )
                )

        if not parts:
            return pd.DataFrame(columns=["DATE_", "ITEMCODE", "elasticity_raw"])
        return pd.concat(parts, ignore_index=True)

    # Item-level format: ItemCode, Elasticity
    usecols = [item_col, elastic_col]
    for chunk in pd.read_csv(elasticity_path, usecols=usecols, chunksize=chunk_size):
        chunk[item_col] = pd.to_numeric(chunk[item_col], errors="coerce")
        chunk[elastic_col] = pd.to_numeric(chunk[elastic_col], errors="coerce")
        subset = chunk.loc[
            chunk[item_col].isin(selected_items),
            [item_col, elastic_col],
        ].copy()
        if not subset.empty:
            subset[item_col] = subset[item_col].astype(int)
            parts.append(
                subset.rename(
                    columns={
                        item_col: "ITEMCODE",
                        elastic_col: "elasticity_raw",
                    }
                )
            )

    if not parts:
        return pd.DataFrame(columns=["ITEMCODE", "elasticity_raw"])

    item_level = pd.concat(parts, ignore_index=True)
    item_level = item_level.dropna(subset=["ITEMCODE", "elasticity_raw"])
    item_level = item_level.drop_duplicates(subset=["ITEMCODE"], keep="last")
    return item_level.reset_index(drop=True)


def _build_item_panels(
    item_day: pd.DataFrame,
    metadata: pd.DataFrame,
    cost_map: pd.DataFrame,
    elasticity: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_history_days: int,
) -> pd.DataFrame:
    meta_lookup = metadata.set_index("ITEMCODE").to_dict(orient="index")
    cost_lookup = cost_map.set_index("ITEMCODE")["cost"].to_dict()

    has_dated_elasticity = "DATE_" in elasticity.columns
    elasticity_lookup = {}
    if has_dated_elasticity:
        elasticity_lookup = {
            int(itemcode): group[["DATE_", "elasticity_raw"]].drop_duplicates("DATE_").set_index("DATE_")["elasticity_raw"]
            for itemcode, group in elasticity.groupby("ITEMCODE")
        }

    item_level_elasticity: dict[int, float] = {}
    if {"ITEMCODE", "elasticity_raw"}.issubset(elasticity.columns):
        item_level_frame = elasticity[["ITEMCODE", "elasticity_raw"]].copy()
        item_level_frame["ITEMCODE"] = pd.to_numeric(item_level_frame["ITEMCODE"], errors="coerce")
        item_level_frame["elasticity_raw"] = pd.to_numeric(item_level_frame["elasticity_raw"], errors="coerce")
        item_level_frame = item_level_frame.dropna(subset=["ITEMCODE", "elasticity_raw"])
        item_level_frame["ITEMCODE"] = item_level_frame["ITEMCODE"].astype(int)
        item_level_frame = item_level_frame.drop_duplicates(subset=["ITEMCODE"], keep="last")
        item_level_elasticity = dict(item_level_frame[["ITEMCODE", "elasticity_raw"]].values)

    full_dates = pd.date_range(start_date, end_date, freq="D")
    panels: list[pd.DataFrame] = []

    for itemcode, group in item_day.groupby("ITEMCODE", sort=True):
        meta = meta_lookup[int(itemcode)]
        base = pd.DataFrame({"DATE_": full_dates})
        merged = base.merge(group, on="DATE_", how="left")
        merged["ITEMCODE"] = int(itemcode)
        merged["ITEMID"] = int(meta["ITEMID"])
        merged["CATEGORY1"] = str(meta["CATEGORY1"])
        merged["CATEGORY2"] = str(meta["CATEGORY2"])
        merged["daily_amount"] = merged["daily_amount"].fillna(0.0)
        merged["daily_revenue"] = merged["daily_revenue"].fillna(0.0)
        merged["unitprice_x_amount"] = merged["unitprice_x_amount"].fillna(0.0)
        merged["unitprice_obs"] = merged["unitprice_obs"].fillna(0)
        merged["baseprice_x_amount"] = merged["baseprice_x_amount"].fillna(0.0)
        merged["baseprice_obs"] = merged["baseprice_obs"].fillna(0)
        merged["txn_count"] = merged["txn_count"].fillna(0)
        merged["cost"] = float(cost_lookup.get(int(itemcode), np.nan))

        observed_price = _safe_div(merged["unitprice_x_amount"], merged["daily_amount"].replace(0.0, np.nan), fill_value=np.nan)
        mean_price = _safe_div(merged["unitprice_x_amount"], merged["unitprice_obs"].replace(0.0, np.nan), fill_value=np.nan)
        merged["current_price"] = pd.Series(observed_price, index=merged.index).fillna(pd.Series(mean_price, index=merged.index))
        merged["current_price"] = merged["current_price"].replace([np.inf, -np.inf], np.nan)
        merged["current_price"] = merged["current_price"].ffill().bfill()
        if merged["current_price"].isna().all():
            fallback_price = merged["cost"].dropna().iloc[0] * 1.15 if merged["cost"].notna().any() else 1.0
            merged["current_price"] = fallback_price
        else:
            merged["current_price"] = merged["current_price"].fillna(merged["current_price"].median())

        observed_base_price = _safe_div(merged["baseprice_x_amount"], merged["daily_amount"].replace(0.0, np.nan), fill_value=np.nan)
        mean_base_price = _safe_div(merged["baseprice_x_amount"], merged["baseprice_obs"].replace(0.0, np.nan), fill_value=np.nan)
        merged["base_price_raw"] = pd.Series(observed_base_price, index=merged.index).fillna(pd.Series(mean_base_price, index=merged.index))
        merged["base_price_raw"] = merged["base_price_raw"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        merged["base_price_raw"] = merged["base_price_raw"].fillna(merged["current_price"])

        merged["is_sale_day"] = (merged["daily_amount"] > 0).astype(int)
        merged["history_days_available"] = np.arange(len(merged))
        merged["units_7d"] = merged["daily_amount"].rolling(7, min_periods=1).sum().shift(1).fillna(0.0)
        merged["units_28d"] = merged["daily_amount"].rolling(28, min_periods=1).sum().shift(1).fillna(0.0)
        merged["revenue_7d"] = merged["daily_revenue"].rolling(7, min_periods=1).sum().shift(1).fillna(0.0)
        merged["revenue_28d"] = merged["daily_revenue"].rolling(28, min_periods=1).sum().shift(1).fillna(0.0)
        merged["price_mean_28d"] = merged["current_price"].rolling(28, min_periods=3).mean().shift(1)
        merged["price_std_28d"] = merged["current_price"].rolling(28, min_periods=3).std(ddof=0).shift(1)
        merged["demand_mean_28d"] = merged["daily_amount"].rolling(28, min_periods=3).mean().shift(1)
        merged["demand_std_28d"] = merged["daily_amount"].rolling(28, min_periods=3).std(ddof=0).shift(1)
        merged["active_days_28d"] = merged["is_sale_day"].rolling(28, min_periods=3).mean().shift(1)
        price_delta = merged["current_price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged["price_change_share_28d"] = price_delta.abs().gt(0.001).rolling(28, min_periods=3).mean().shift(1).fillna(0.0)
        merged["price_cv_28d"] = _safe_div(merged["price_std_28d"], merged["price_mean_28d"], fill_value=0.0)
        merged["base_price_56d"] = merged["base_price_raw"].rolling(56, min_periods=7).max().shift(1)
        merged["base_price_56d"] = merged["base_price_56d"].fillna(merged["base_price_raw"]).fillna(merged["current_price"])
        merged["discount_to_base"] = 1.0 - _safe_div(merged["current_price"], merged["base_price_56d"], fill_value=1.0)
        merged["gross_margin_pct"] = _safe_div(
            merged["current_price"] - merged["cost"],
            merged["current_price"].replace(0.0, np.nan),
            fill_value=0.0,
        )
        merged["future_amount_7d"] = _future_window_sum(merged["daily_amount"], 7)
        merged["future_revenue_7d"] = _future_window_sum(merged["daily_revenue"], 7)
        merged["dow"] = merged["DATE_"].dt.dayofweek.astype(int)
        merged["month"] = merged["DATE_"].dt.month.astype(int)
        merged["weekofyear"] = merged["DATE_"].dt.isocalendar().week.astype(int)
        merged["dow_sin"] = np.sin(2.0 * np.pi * merged["dow"] / 7.0)
        merged["dow_cos"] = np.cos(2.0 * np.pi * merged["dow"] / 7.0)
        merged["month_sin"] = np.sin(2.0 * np.pi * merged["month"] / 12.0)
        merged["month_cos"] = np.cos(2.0 * np.pi * merged["month"] / 12.0)

        elasticity_series = elasticity_lookup.get(int(itemcode)) if has_dated_elasticity else None
        if elasticity_series is not None:
            merged = merged.merge(
                elasticity_series.rename("elasticity_raw"),
                left_on="DATE_",
                right_index=True,
                how="left",
            )
        else:
            merged["elasticity_raw"] = item_level_elasticity.get(int(itemcode), np.nan)
        merged["elasticity"] = (
            pd.to_numeric(merged["elasticity_raw"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )
        merged["elasticity"] = merged["elasticity"].fillna(-1.0).clip(-4.0, 0.25)

        merged = merged[merged["history_days_available"] >= int(min_history_days)].copy()
        panels.append(merged)

    if not panels:
        raise RuntimeError("No item panels survived the minimum history threshold")

    dataset = pd.concat(panels, ignore_index=True)
    dataset["price_mean_28d"] = dataset["price_mean_28d"].fillna(dataset["current_price"])
    dataset["price_std_28d"] = dataset["price_std_28d"].fillna(0.0)
    dataset["demand_mean_28d"] = dataset["demand_mean_28d"].fillna(0.0)
    dataset["demand_std_28d"] = dataset["demand_std_28d"].fillna(0.0)
    dataset["active_days_28d"] = dataset["active_days_28d"].fillna(0.0)
    dataset["cost"] = dataset["cost"].fillna(dataset["current_price"] * 0.7)

    dataset["category_future_revenue_7d"] = dataset.groupby(["DATE_", "CATEGORY2"])["future_revenue_7d"].transform("sum")
    dataset["category_revenue_28d"] = dataset.groupby(["DATE_", "CATEGORY2"])["revenue_28d"].transform("sum")
    dataset["category_units_28d"] = dataset.groupby(["DATE_", "CATEGORY2"])["units_28d"].transform("sum")
    dataset["revenue_share_28d"] = _safe_div(dataset["revenue_28d"], dataset["category_revenue_28d"], fill_value=0.0)
    dataset["units_share_28d"] = _safe_div(dataset["units_28d"], dataset["category_units_28d"], fill_value=0.0)
    dataset["future_basket_share"] = _safe_div(
        dataset["future_revenue_7d"],
        dataset["category_future_revenue_7d"],
        fill_value=0.0,
    )

    item_summary = (
        dataset.groupby("ITEMCODE", as_index=False)
        .agg(
            total_revenue_window=("daily_revenue", "sum"),
            total_amount_window=("daily_amount", "sum"),
            activity_rate=("is_sale_day", "mean"),
            avg_revenue_share=("revenue_share_28d", "mean"),
        )
    )
    item_summary["rev_rank"] = item_summary["total_revenue_window"].rank(pct=True)
    item_summary["qty_rank"] = item_summary["total_amount_window"].rank(pct=True)
    item_summary["activity_rank"] = item_summary["activity_rate"].rank(pct=True)
    item_summary["kvi_score"] = (
        0.5 * item_summary["rev_rank"]
        + 0.3 * item_summary["qty_rank"]
        + 0.2 * item_summary["activity_rank"]
    ).clip(0.0, 1.0)

    return dataset.merge(item_summary[["ITEMCODE", "kvi_score"]], on="ITEMCODE", how="left")


def _build_elasticity_context_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    context = dataset[
        [
            "DATE_",
            "ITEMCODE",
            "ITEMID",
            "CATEGORY1",
            "CATEGORY2",
            "current_price",
            "cost",
            "units_7d",
            "future_amount_7d",
            "revenue_7d",
            "future_revenue_7d",
            "price_mean_28d",
            "price_std_28d",
            "demand_mean_28d",
            "discount_to_base",
            "gross_margin_pct",
            "dow",
            "month",
            "kvi_score",
            "is_kvi",
        ]
    ].copy()
    context = context.rename(
        columns={
            "units_7d": "qty_prev_7d",
            "future_amount_7d": "qty_next_7d",
            "revenue_7d": "revenue_prev_7d",
            "future_revenue_7d": "revenue_next_7d",
        }
    )
    return context.sort_values(["ITEMCODE", "DATE_"]).reset_index(drop=True)


def build_dataset(config: dict[str, Any], logger: logging.Logger) -> dict[str, str]:
    output_tables = Path(config["paths"]["output_dir"]) / "tables"
    output_manifests = Path(config["paths"]["output_dir"]) / "manifests"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_manifests.mkdir(parents=True, exist_ok=True)

    data_path = Path(config["paths"]["full_data"])
    cost_path = Path(config["paths"]["cost"])
    elasticity_path = Path(config["paths"]["elasticity"])

    dataset_cfg = config["dataset"]
    metadata, global_min_date, global_max_date = _scan_item_totals(
        data_path=data_path,
        chunk_size=int(dataset_cfg["chunk_size"]),
        max_rows=dataset_cfg.get("max_rows"),
        min_item_amount=float(dataset_cfg["min_item_amount"]),
        max_items=int(dataset_cfg["max_items"]),
        logger=logger,
    )
    start_date = global_max_date - pd.Timedelta(days=int(dataset_cfg["history_days"]) - 1)

    selected_items = set(metadata["ITEMCODE"].astype(int).tolist())
    item_day = _aggregate_item_day(
        data_path=data_path,
        selected_items=selected_items,
        start_date=start_date,
        end_date=global_max_date,
        chunk_size=int(dataset_cfg["chunk_size"]),
        max_rows=dataset_cfg.get("max_rows"),
    )
    elasticity = _read_elasticity_subset(
        elasticity_path=elasticity_path,
        selected_items=selected_items,
        start_date=start_date,
        end_date=global_max_date,
        chunk_size=int(dataset_cfg["chunk_size"]),
    )
    cost_map = _read_cost_map(cost_path)

    dataset = _build_item_panels(
        item_day=item_day,
        metadata=metadata,
        cost_map=cost_map,
        elasticity=elasticity,
        start_date=start_date,
        end_date=global_max_date,
        min_history_days=int(dataset_cfg["min_history_days"]),
    )

    kvi_cutoff = float(dataset["kvi_score"].quantile(1.0 - float(dataset_cfg["kvi_top_pct"])))
    dataset["is_kvi"] = (dataset["kvi_score"] >= kvi_cutoff).astype(int)
    dataset["is_anchor"] = (
        (dataset["is_kvi"] == 1)
        & (dataset["revenue_share_28d"] >= dataset["revenue_share_28d"].quantile(0.75))
        & (dataset["active_days_28d"] >= dataset["active_days_28d"].median())
    ).astype(int)

    elasticity_context = _build_elasticity_context_dataset(dataset)
    elasticity_context_path = output_tables / "elasticity_context_dataset.csv"
    elasticity_context.to_csv(elasticity_context_path, index=False)

    dataset["CATEGORY1_CODE"] = pd.factorize(dataset["CATEGORY1"])[0].astype(int)
    dataset["CATEGORY2_CODE"] = pd.factorize(dataset["CATEGORY2"])[0].astype(int)
    dataset["ITEMID_CODE"] = pd.factorize(dataset["ITEMID"])[0].astype(int)

    metadata_out = (
        dataset.groupby("ITEMCODE", as_index=False)
        .agg(
            ITEMID=("ITEMID", "first"),
            CATEGORY1=("CATEGORY1", "first"),
            CATEGORY2=("CATEGORY2", "first"),
            kvi_score=("kvi_score", "max"),
            is_kvi=("is_kvi", "max"),
            is_anchor=("is_anchor", "max"),
            avg_price=("current_price", "mean"),
            total_revenue_window=("daily_revenue", "sum"),
        )
        .sort_values("total_revenue_window", ascending=False)
        .reset_index(drop=True)
    )

    dataset_path = output_tables / "item_day_dataset.csv"
    metadata_path = output_tables / "item_metadata.csv"
    manifest_path = output_manifests / "dataset_manifest.json"

    dataset.to_csv(dataset_path, index=False)
    metadata_out.to_csv(metadata_path, index=False)
    manifest = {
        "dataset_rows": int(len(dataset)),
        "selected_items": int(metadata["ITEMCODE"].nunique()),
        "global_min_date": str(global_min_date.date()),
        "global_max_date": str(global_max_date.date()),
        "start_date": str(start_date.date()),
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "elasticity_context_path": str(elasticity_context_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved item-day dataset with %s rows", len(dataset))

    return {
        "item_day_dataset": str(dataset_path),
        "item_metadata": str(metadata_path),
        "dataset_manifest": str(manifest_path),
        "elasticity_context_dataset": str(elasticity_context_path),
    }
