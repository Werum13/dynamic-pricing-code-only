"""
Price Optimizer — Web Interface (Streamlit)
============================================
Запуск:
    cd <project_root>
    streamlit run app.py

Версия 3:
  - динамический unit_price = TOTALPRICE/AMOUNT (меняется по дням)
  - base price (статический) = UNITPRICE из CSV (прайс-листовая цена)
  - объясняющий отчёт по связанным товарам вместо/дополнение к матрице
  - 3 новых графика взаимодействия товаров по спросу
  - обоснование решения оптимизатора
"""

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Elasticity.Baseline import MLOptimizerBaseline, SimResult
from Elasticity.DemandModel import demand_model
from Elasticity.ETL import etl_with_demand_target
from Elasticity.kvi_context import load_kvi_context
from Elasticity.data_sources import load_elasticity_source_data
from Elasticity.pipeline import Pipeline as LSTPipeline

# ─── Пути к данным ──────────────────────────────────────────────────────────
COST_PATH = BASE_DIR / "data" / "cost.csv"
ORDER_DETAILS_PATH = BASE_DIR / "data" / "Order_Details.csv"
ORDERS_PATH = BASE_DIR / "data" / "Orders.csv"

DROP_COLS = [
    "DATE_", "CATEGORY1", "CATEGORY2",
    "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
    "AMOUNT_0D_target", "AMOUNT_1D_target", "AMOUNT_7D_target",
    "AMOUNT_15D_target", "AMOUNT_30D_target",
    "AMOUNT_1D", "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D",
]

SIGNAL_DIAGNOSTIC_COLS = [
    "price_cv_30d", "price_range_pct_30d", "price_change_share_30d",
    "selling_days_30d", "demand_cv_30d", "AMOUNT_mean_30d",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Загрузка данных
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_raw_data():
    data = load_elasticity_source_data(
        BASE_DIR,
        usecols=["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"],
    )
    cost = pd.read_csv(COST_PATH)
    merged = pd.merge(data, cost[["ITEMCODE", "cost"]], how="left", on="ITEMCODE")
    merged["DATE_"] = pd.to_datetime(merged["DATE_"])
    return merged


@st.cache_data(ttl=3600)
def get_itemcodes():
    if not ORDER_DETAILS_PATH.exists():
        raw = load_raw_data()
        return sorted(pd.to_numeric(raw["ITEMCODE"], errors="coerce").dropna().astype("int64").unique().tolist())

    unique_codes: set[int] = set()
    for chunk in pd.read_csv(
        ORDER_DETAILS_PATH, usecols=["ITEMCODE"], dtype=str, chunksize=500_000, low_memory=False,
    ):
        codes = pd.to_numeric(
            chunk["ITEMCODE"].astype("string").str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if codes.notna().any():
            unique_codes.update(codes.dropna().astype("int64").tolist())

    return sorted(unique_codes) if unique_codes else []


# ═════════════════════════════════════════════════════════════════════════════
#  Цены: утилиты
# ═════════════════════════════════════════════════════════════════════════════

def _compute_daily_unit_price(group: pd.DataFrame) -> float:
    """
    Динамический unit price = TOTALPRICE / AMOUNT за день.
    Отражает фактическую цену за единицу в чеке (меняется изо дня в день).
    В отличие от UNITPRICE (статической), учитывает скидки и наценки в конкретном чеке.
    """
    total = pd.to_numeric(
        group["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    ).sum()
    amount = pd.to_numeric(group["AMOUNT"], errors="coerce").sum()
    if amount > 0 and np.isfinite(total) and np.isfinite(amount):
        return float(total / amount)
    return np.nan


def find_unitary_elasticity_price(prices, elasticity_values):
    eps = np.array(elasticity_values, dtype=float)
    P = np.array(prices, dtype=float)
    mask = (eps[:-1] + 1) * (eps[1:] + 1) < 0
    if mask.any():
        idx = np.where(mask)[0][0]
        P1, E1 = P[idx], eps[idx]
        P2, E2 = P[idx + 1], eps[idx + 1]
        return float(P1 + (P2 - P1) * (-1 - E1) / (E2 - E1))
    return None


def _create_future_row(day, itemcode, df_hist):
    last_date = df_hist["DATE_"].max()
    last_row = df_hist[df_hist["DATE_"] == last_date].iloc[0].copy()
    new_row = last_row.to_dict()
    new_row["DATE_"] = pd.to_datetime(day)
    new_row["ITEMCODE"] = itemcode
    for c in ["GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D"]:
        new_row[c] = np.nan
    new_row["AMOUNT_1D"] = 0
    new_row["TOTALPRICE_1D"] = 0
    return pd.DataFrame([new_row])


def _build_actual_daily_summary(data, first_day, last_day):
    """
    Агрегирует исторический факт на дневной уровень.
    actual_price = TOTALPRICE/AMOUNT (динамический unit price, меняется по дням).
    """
    data = data.copy()
    data["AMOUNT"] = pd.to_numeric(data["AMOUNT"], errors="coerce")
    data["TOTALPRICE"] = pd.to_numeric(
        data["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )
    data["UNITPRICE"] = pd.to_numeric(
        data["UNITPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )

    period = data[(data["DATE_"] >= first_day) & (data["DATE_"] < last_day)].copy()
    if len(period) == 0:
        return pd.DataFrame(columns=["DATE_", "actual_amount", "actual_gmv", "actual_price"])

    period["DATE_"] = pd.to_datetime(period["DATE_"]).dt.normalize()
    sold = period[period["AMOUNT"] > 0].copy()

    daily = period.groupby("DATE_", as_index=False).agg(
        actual_amount=("AMOUNT", "sum"),
        actual_gmv=("TOTALPRICE", "sum"),
    )

    if len(sold) > 0:
        # Динамический unit price: TOTALPRICE/AMOUNT per day
        daily_price = (
            sold.groupby("DATE_", as_index=False)
            .apply(lambda g: pd.Series({"actual_price": _compute_daily_unit_price(g)}))
            .reset_index(drop=True)
        )
        # Если apply вернул multi-index
        if "DATE_" not in daily_price.columns:
            daily_price = (
                sold.groupby("DATE_")
                .apply(_compute_daily_unit_price)
                .rename("actual_price")
                .reset_index()
            )
        daily = daily.merge(daily_price, on="DATE_", how="left")
    else:
        daily["actual_price"] = np.nan

    return daily.sort_values("DATE_").reset_index(drop=True)


def _lookup_historical_price(data, day):
    """
    Возвращает динамический unit price (TOTALPRICE/AMOUNT) за конкретный день.
    Это фактическая цена продажи — меняется изо дня в день в зависимости от чеков.
    """
    data_c = data.copy()
    data_c["DATE_"] = pd.to_datetime(data_c["DATE_"], errors="coerce")
    data_c["AMOUNT"] = pd.to_numeric(data_c["AMOUNT"], errors="coerce")
    data_c["TOTALPRICE"] = pd.to_numeric(
        data_c["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )

    day_norm = pd.Timestamp(day).normalize()
    mask = (data_c["DATE_"].dt.normalize() == day_norm) & (data_c["AMOUNT"] > 0)
    same_day = data_c.loc[mask]

    if len(same_day) == 0:
        return None

    return _compute_daily_unit_price(same_day)


def _summarize_elasticity_by_band(prices, elasticity_values, n_bands=4):
    prices = pd.Series(prices, dtype=float)
    elasticity = pd.Series(elasticity_values, dtype=float)
    valid = prices.notna() & elasticity.notna() & np.isfinite(elasticity)
    if valid.sum() == 0:
        return []

    n_unique = int(prices[valid].nunique())
    n_bins = max(1, min(n_bands, n_unique))
    labels = [f"Q{i+1}" for i in range(n_bins)]

    if n_bins == 1:
        return [{"price_band": labels[0], "price_min": float(prices[valid].min()),
                 "price_max": float(prices[valid].max()), "mean_elasticity": float(elasticity[valid].mean()),
                 "n_points": int(valid.sum())}]

    bands = pd.qcut(prices[valid], q=n_bins, labels=labels, duplicates="drop")
    out = []
    for label in bands.cat.categories:
        idx = bands == label
        band_prices = prices.loc[bands.index[idx]]
        band_eps = elasticity.loc[bands.index[idx]]
        out.append({"price_band": str(label), "price_min": float(band_prices.min()),
                    "price_max": float(band_prices.max()), "mean_elasticity": float(band_eps.mean()),
                    "n_points": int(idx.sum())})
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Связи товаров
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _load_relations_map():
    ctx = load_kvi_context()
    raw_map = ctx.get("sub_map", {})
    out: dict[int, dict[str, list[int]]] = {}
    for k, v in raw_map.items():
        kk = int(k)
        out[kk] = {
            "substitutes": [int(x) for x in v.get("substitutes", [])],
            "complements": [int(x) for x in v.get("complements", [])],
            "cannibals": [int(x) for x in v.get("cannibals", [])],
        }
    return out


def _build_relation_matrix(itemcode: int, rel_map: dict[int, dict[str, list[int]]], max_items: int = 12):
    direct = rel_map.get(int(itemcode), {})
    direct_subs = [int(x) for x in direct.get("substitutes", [])]
    direct_comps = [int(x) for x in direct.get("complements", [])]
    direct_canns = [int(x) for x in direct.get("cannibals", [])]

    incoming = []
    for src, entry in rel_map.items():
        neighbors = set(entry.get("substitutes", []) + entry.get("complements", []) + entry.get("cannibals", []))
        if int(itemcode) in neighbors:
            incoming.append(int(src))

    ordered_ids = [int(itemcode)] + direct_subs + direct_comps + direct_canns + incoming
    seen = set()
    item_ids = []
    for x in ordered_ids:
        if x not in seen:
            seen.add(x)
            item_ids.append(x)

    if len(item_ids) > max_items:
        item_ids = item_ids[:max_items]

    matrix = pd.DataFrame("", index=item_ids, columns=item_ids)
    for src in item_ids:
        entry = rel_map.get(int(src), {})
        subs = set(int(x) for x in entry.get("substitutes", []))
        comps = set(int(x) for x in entry.get("complements", []))
        canns = set(int(x) for x in entry.get("cannibals", []))
        for dst in item_ids:
            if src == dst:
                matrix.loc[src, dst] = "self"
                continue
            tags = []
            if dst in comps:
                tags.append("companion")
            if dst in subs:
                tags.append("competitor/substitute")
            if dst in canns:
                tags.append("competitor/cannibal")
            matrix.loc[src, dst] = " | ".join(tags)

    role_summary = {"substitutes": direct_subs, "complements": direct_comps, "cannibals": direct_canns}
    return matrix, item_ids, role_summary


def _build_relation_report(itemcode: int, role_summary: dict, rel_map: dict) -> list[dict]:
    """
    Строит объясняющий отчёт по связанным товарам.
    Для каждой связи возвращает тип, экономическую интерпретацию и рекомендацию.
    """
    ROLE_META = {
        "substitutes": {
            "emoji": "🔄",
            "label": "Субститут (заменитель)",
            "color": "🟡",
            "economic_meaning": (
                "Товары-заменители конкурируют за один и тот же спрос. "
                "При повышении цены целевого товара часть покупателей переключается на субститут "
                "→ спрос на субститут растёт (положительная перекрёстная эластичность)."
            ),
            "price_advice": (
                "Если повышаешь цену целевого товара, субститут может позволить "
                "скорректировать цену вверх (часть спроса перетечёт к нему автоматически). "
                "Следи, чтобы разрыв в ценах не стал слишком большим."
            ),
        },
        "cannibals": {
            "emoji": "🍖",
            "label": "Каннибал (внутренний конкурент)",
            "color": "🔴",
            "economic_meaning": (
                "Каннибал — это более дешёвая или более привлекательная альтернатива из того же портфеля. "
                "Снижение цены целевого товара ниже уровня каннибала может перетянуть продажи "
                "от него к целевому товару и снизить суммарную маржу портфеля."
            ),
            "price_advice": (
                "При оптимизации цены учитывай ценовой коридор каннибала. "
                "Оптимизатор минимизирует каннибализацию, но важно мониторить долю продаж обоих товаров."
            ),
        },
        "complements": {
            "emoji": "🤝",
            "label": "Комплемент (дополняющий товар)",
            "color": "🟢",
            "economic_meaning": (
                "Товары-комплементы покупают вместе. Рост продаж одного влечёт рост продаж другого "
                "(отрицательная перекрёстная эластичность по доходу от портфеля). "
                "Снижение цены на целевой товар может стимулировать продажи комплемента."
            ),
            "price_advice": (
                "Рассматривай ценообразование как пакет: можно искусственно занизить цену одного "
                "и отыграть маржу на другом. Оптимизатор учитывает комплементы при совместной оптимизации."
            ),
        },
    }

    rows = []
    for role_key in ["cannibals", "substitutes", "complements"]:
        items = role_summary.get(role_key, [])
        meta = ROLE_META[role_key]
        for related_ic in items:
            # Обратная связь: как related видит наш товар
            related_entry = rel_map.get(related_ic, {})
            reverse_roles = []
            if itemcode in related_entry.get("substitutes", []):
                reverse_roles.append("субститут")
            if itemcode in related_entry.get("cannibals", []):
                reverse_roles.append("каннибал")
            if itemcode in related_entry.get("complements", []):
                reverse_roles.append("комплемент")
            reverse_str = " / ".join(reverse_roles) if reverse_roles else "—"

            rows.append({
                "related_itemcode": related_ic,
                "role_emoji": meta["emoji"],
                "role_label": meta["label"],
                "color": meta["color"],
                "economic_meaning": meta["economic_meaning"],
                "price_advice": meta["price_advice"],
                "reverse_role": reverse_str,
            })
    return rows


def _pick_secondary_item(itemcode: int, role_summary: dict[str, list[int]]):
    for role in ["substitutes", "cannibals", "complements"]:
        vals = role_summary.get(role, [])
        if vals:
            return int(vals[0]), role
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  Взаимодействие по спросу (3 графика)
# ═════════════════════════════════════════════════════════════════════════════

def _build_demand_interaction_charts(
    target_ic: int,
    second_ic: int,
    second_role: str,
    group_all_ics: list[int],
    two_items_df: pd.DataFrame,
    group_df: pd.DataFrame,
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
):
    """
    Строит 3 графика взаимодействия двух товаров:
      1. Динамика цен (dynamic unit_price = TOTALPRICE/AMOUNT) по дням
      2. Динамика спроса (AMOUNT) по дням + кросс-эффект
      3. Доля продаж целевого товара внутри группы (stacked bar)
    """
    if two_items_df.empty:
        return None

    two = two_items_df.copy()
    two["DATE_"] = pd.to_datetime(two["DATE_"]).dt.normalize()
    two = two[(two["DATE_"] >= first_day) & (two["DATE_"] < last_day)]
    two["AMOUNT"] = pd.to_numeric(two["AMOUNT"], errors="coerce")
    two["TOTALPRICE"] = pd.to_numeric(
        two["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )

    if two.empty:
        return None

    # Агрегируем до дня: unit_price = TOTALPRICE/AMOUNT, amount = sum
    def _daily_agg(g):
        total = g["TOTALPRICE"].sum()
        amt = g["AMOUNT"].sum()
        up = total / amt if amt > 0 else np.nan
        return pd.Series({"unit_price_dynamic": up, "daily_amount": amt})

    daily = (
        two[two["AMOUNT"] > 0]
        .groupby(["DATE_", "ITEMCODE"], as_index=False)
        .apply(_daily_agg)
        .reset_index(drop=True)
    )

    target_daily = daily[daily["ITEMCODE"].astype("int64") == int(target_ic)].sort_values("DATE_")
    second_daily = daily[daily["ITEMCODE"].astype("int64") == int(second_ic)].sort_values("DATE_")

    role_labels = {
        "substitutes": "Субститут",
        "cannibals": "Каннибал",
        "complements": "Комплемент",
    }
    role_label = role_labels.get(second_role, second_role)

    # ── График 1: Динамика цен ──────────────────────────────────────────
    fig1 = go.Figure()
    if len(target_daily) > 0:
        fig1.add_trace(go.Scatter(
            x=target_daily["DATE_"], y=target_daily["unit_price_dynamic"],
            mode="lines+markers", name=f"ITEMCODE {target_ic} (целевой)",
            line=dict(color="steelblue", width=2.5),
        ))
    if len(second_daily) > 0:
        fig1.add_trace(go.Scatter(
            x=second_daily["DATE_"], y=second_daily["unit_price_dynamic"],
            mode="lines+markers", name=f"ITEMCODE {second_ic} ({role_label})",
            line=dict(color="coral", width=2, dash="dot"),
        ))
    fig1.update_layout(
        title="📈 Динамика дневного Unit Price (TOTALPRICE/AMOUNT)",
        xaxis_title="Дата", yaxis_title="Unit Price (факт)",
        height=320, legend=dict(orientation="h", y=-0.2),
    )

    # ── График 2: Спрос по дням + кросс-эффект ──────────────────────────
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    if len(target_daily) > 0:
        fig2.add_trace(go.Scatter(
            x=target_daily["DATE_"], y=target_daily["daily_amount"],
            mode="lines+markers", name=f"Спрос {target_ic}",
            line=dict(color="steelblue", width=2.5),
        ), secondary_y=False)
    if len(second_daily) > 0:
        fig2.add_trace(go.Scatter(
            x=second_daily["DATE_"], y=second_daily["daily_amount"],
            mode="lines+markers", name=f"Спрос {second_ic} ({role_label})",
            line=dict(color="coral", width=2, dash="dot"),
        ), secondary_y=True)
        # Цена целевого товара на фоне (прозрачная) — для кросс-эффекта
        if len(target_daily) > 0:
            fig2.add_trace(go.Scatter(
                x=target_daily["DATE_"], y=target_daily["unit_price_dynamic"],
                mode="lines", name=f"Цена {target_ic} (ось 2)",
                line=dict(color="steelblue", width=1, dash="longdash"),
                opacity=0.4,
            ), secondary_y=True)

    _cross_hint = {
        "substitutes": "При росте цены целевого товара спрос на субститут должен расти",
        "cannibals": "При снижении цены целевого товара он каннибализирует продажи другого",
        "complements": "Продажи комплементов двигаются в одну сторону (совместная покупка)",
    }.get(second_role, "")

    fig2.update_layout(
        title=f"📊 Взаимодействие по спросу  —  {_cross_hint}",
        xaxis_title="Дата", height=320,
        legend=dict(orientation="h", y=-0.25),
    )
    fig2.update_yaxes(title_text=f"Спрос ITEMCODE {target_ic}", secondary_y=False)
    fig2.update_yaxes(title_text=f"Спрос / Цена {second_ic}", secondary_y=True)

    # ── График 3: Доля продаж внутри группы (stacked bar) ───────────────
    grp = group_df.copy()
    grp["DATE_"] = pd.to_datetime(grp["DATE_"]).dt.normalize()
    grp = grp[(grp["DATE_"] >= first_day) & (grp["DATE_"] < last_day)]
    grp["AMOUNT"] = pd.to_numeric(grp["AMOUNT"], errors="coerce")

    if not grp.empty:
        share_df = (
            grp[grp["AMOUNT"] > 0]
            .groupby(["DATE_", "ITEMCODE"], as_index=False)
            .agg(daily_amount=("AMOUNT", "sum"))
        )
        # Общий дневной объём по группе
        total_day = share_df.groupby("DATE_")["daily_amount"].sum().rename("total_amount").reset_index()
        share_df = share_df.merge(total_day, on="DATE_")
        share_df["share_pct"] = share_df["daily_amount"] / share_df["total_amount"].clip(lower=1e-8) * 100

        fig3 = go.Figure()
        colors = ["steelblue", "coral", "seagreen", "goldenrod", "mediumpurple", "tomato"]
        all_group_ics = share_df["ITEMCODE"].astype("int64").unique().tolist()
        for i, ic in enumerate(all_group_ics):
            part = share_df[share_df["ITEMCODE"].astype("int64") == ic].sort_values("DATE_")
            label = f"{ic} (целевой)" if ic == int(target_ic) else (
                f"{ic} ({role_label})" if ic == int(second_ic) else f"{ic}"
            )
            fig3.add_trace(go.Bar(
                x=part["DATE_"], y=part["share_pct"],
                name=label,
                marker_color=colors[i % len(colors)],
            ))
        fig3.update_layout(
            barmode="stack",
            title="📊 Доля продаж товаров внутри группы по дням (%)",
            xaxis_title="Дата", yaxis_title="Доля от группы (%)",
            height=320, legend=dict(orientation="h", y=-0.3),
            yaxis=dict(range=[0, 100]),
        )
    else:
        fig3 = None

    return fig1, fig2, fig3


# ═════════════════════════════════════════════════════════════════════════════
#  Групповая маржа
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Загружаем историю группы товаров...")
def _load_group_data(itemcodes: tuple[int, ...]):
    if not itemcodes:
        return pd.DataFrame()

    data = load_elasticity_source_data(
        BASE_DIR,
        usecols=["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT"],
        itemcodes=[int(x) for x in itemcodes],
    )
    if data.empty:
        return data

    cost = pd.read_csv(COST_PATH, usecols=["ITEMCODE", "cost"])
    out = data.merge(cost, how="left", on="ITEMCODE")
    out["DATE_"] = pd.to_datetime(out["DATE_"], errors="coerce").dt.normalize()
    out["AMOUNT"] = pd.to_numeric(out["AMOUNT"], errors="coerce")
    out["UNITPRICE"] = pd.to_numeric(
        out["UNITPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )
    out["TOTALPRICE"] = pd.to_numeric(
        out["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    return out


def _build_group_margin_series(group_data, history_pred, selected_itemcode, first_day, last_day):
    if group_data.empty:
        return pd.DataFrame()

    df = group_data.copy()
    df["margin_actual"] = df["TOTALPRICE"].fillna(0.0) - df["cost"].fillna(0.0) * df["AMOUNT"].fillna(0.0)

    window = df[(df["DATE_"] >= first_day) & (df["DATE_"] < last_day)].copy()
    if window.empty:
        return pd.DataFrame()

    hist_group = window.groupby("DATE_", as_index=False).agg(group_margin_history=("margin_actual", "sum"))

    other_items = window[window["ITEMCODE"].astype("int64") != int(selected_itemcode)]
    other_margin = other_items.groupby("DATE_", as_index=False).agg(other_items_margin=("margin_actual", "sum"))

    opt_sel = history_pred.copy()
    if len(opt_sel) == 0 or "DATE_" not in opt_sel.columns or "margin" not in opt_sel.columns:
        return hist_group
    opt_sel["DATE_"] = pd.to_datetime(opt_sel["DATE_"], errors="coerce").dt.normalize()
    opt_sel = opt_sel.groupby("DATE_", as_index=False).agg(selected_margin_optimized=("margin", "sum"))

    merged = hist_group.merge(other_margin, on="DATE_", how="left").merge(opt_sel, on="DATE_", how="left")
    merged["other_items_margin"] = merged["other_items_margin"].fillna(0.0)
    merged["selected_margin_optimized"] = merged["selected_margin_optimized"].fillna(0.0)
    merged["group_margin_optimized"] = merged["other_items_margin"] + merged["selected_margin_optimized"]
    return merged


# ═════════════════════════════════════════════════════════════════════════════
#  Кривая эластичности
# ═════════════════════════════════════════════════════════════════════════════

def _build_elasticity_data_from_model(raw_item: pd.DataFrame, first_day: pd.Timestamp, n_bins: int = 10):
    if raw_item.empty:
        return None

    hist = raw_item[raw_item["DATE_"] < first_day].copy()
    if len(hist) < 30:
        return None

    etl_df = etl_with_demand_target(hist)
    if etl_df.empty or "AMOUNT_0D_target" not in etl_df.columns:
        return None

    train_full = etl_df.copy()
    target_col = "AMOUNT_7D_target" if "AMOUNT_7D_target" in train_full.columns else "AMOUNT_0D_target"
    target = pd.to_numeric(train_full[target_col], errors="coerce")
    train_X = train_full.drop(columns=DROP_COLS, errors="ignore")
    valid = train_X.notna().all(axis=1) & target.notna()
    train_X = train_X[valid]
    target = target[valid]
    if len(train_X) < 20:
        return None

    model = demand_model(train=train_X, target=target)

    price_series = (
        pd.to_numeric(hist["UNITPRICE"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    price_series = price_series[price_series > 0]
    if len(price_series) < 2:
        return None

    n_eff = max(2, min(int(n_bins), int(price_series.nunique())))
    try:
        bins = pd.qcut(price_series, q=n_eff, duplicates="drop")
        price_grid = price_series.groupby(bins).median().astype(float).sort_values().to_numpy()
    except Exception:
        price_grid = np.linspace(float(price_series.min()), float(price_series.max()), n_eff)

    if len(price_grid) < 2:
        center = float(price_series.median())
        if not np.isfinite(center) or center <= 0:
            return None
        price_grid = np.linspace(center * 0.80, center * 1.20, int(n_bins))

    # ── Баг 4 фикс: расширяем сетку вправо на 150% от исторического максимума.
    # Если оптимизатор рекомендует цену выше исторической (при неэластичном
    # спросе — почти всегда), оптимальная точка P* должна попадать в кадр.
    p_max_hist = float(price_grid[-1])
    p_extended_max = p_max_hist * 1.5  # покрываем зону вплоть до +50% от истории
    # Добавляем 5 дополнительных точек в расширенной зоне
    extra_grid = np.linspace(p_max_hist * 1.05, p_extended_max, 5)
    price_grid = np.unique(np.concatenate([price_grid, extra_grid]))

    template = train_full.sort_values("DATE_").tail(1).copy()
    rows = []
    for p in price_grid:
        r = template.copy()
        r["UNITPRICE"] = float(p)
        if "BASEPRICE" in r.columns:
            base = r["BASEPRICE"].replace(0, np.nan)
            r["discount"] = 1.0 - r["UNITPRICE"] / base
        if "cost" in r.columns:
            cost = r["cost"].replace(0, np.nan)
            r["margin_%"] = (r["UNITPRICE"] - r["cost"]) / cost
        rows.append(r)

    grid_df = pd.concat(rows, ignore_index=True)
    for col in model.columns_:
        if col not in grid_df.columns:
            grid_df[col] = 0
    grid_df = grid_df.fillna(0)

    pred = model.predict(grid_df.copy())
    q = pred["pred_Q"].to_numpy(dtype=float)
    q_low = pred["pred_Q_lower"].to_numpy(dtype=float)
    q_up = pred["pred_Q_upper"].to_numpy(dtype=float)
    p_sorted, q_mono, q_low_mono, q_up_mono = model.monotonicize_curve(
        price_grid, q, lower=q_low, upper=q_up,
    )

    q_low_final = q_low_mono if q_low_mono is not None else q_mono
    q_up_final = q_up_mono if q_up_mono is not None else q_mono

    # ── Баг 2 фикс: ε считаем через model.elasticity(), а не np.gradient ──
    # np.gradient от монотонизированной (кусочно-постоянной) кривой = 0.
    # model.elasticity() использует центральную разность по исходной модели
    # и возвращает правильные значения (~−0.5, согласно таблице по дням).
    if len(price_grid) >= 2:
        eps_rows = []
        for p_val in p_sorted:
            row_eps = template.copy()
            row_eps["UNITPRICE"] = float(p_val)
            if "BASEPRICE" in row_eps.columns:
                base = row_eps["BASEPRICE"].replace(0, np.nan)
                row_eps["discount"] = 1.0 - row_eps["UNITPRICE"] / base
            if "cost" in row_eps.columns:
                cost_val = row_eps["cost"].replace(0, np.nan)
                row_eps["margin_%"] = (row_eps["UNITPRICE"] - row_eps["cost"]) / cost_val
            for col in model.columns_:
                if col not in row_eps.columns:
                    row_eps[col] = 0
            row_eps = row_eps.fillna(0)
            eps_rows.append(row_eps)

        eps_grid_df = pd.concat(eps_rows, ignore_index=True)
        eps_sorted = model.elasticity(eps_grid_df)
        # Сглаживаем шум медианным фильтром (3 точки)
        from scipy.ndimage import uniform_filter1d
        if len(eps_sorted) >= 3:
            eps_sorted = uniform_filter1d(eps_sorted.astype(float), size=3, mode="nearest")
        eps_sorted = np.asarray(eps_sorted, dtype=float)
    else:
        eps_sorted = np.full_like(p_sorted, np.nan, dtype=float)

    interval_level = None
    try:
        interval_level = float(model.get_uncertainty_summary().get("interval_level", np.nan))
        if not np.isfinite(interval_level):
            interval_level = None
    except Exception:
        interval_level = None

    unitary_price = find_unitary_elasticity_price(p_sorted, eps_sorted)

    return {
        "prices": p_sorted.tolist(), "qty": q_mono.tolist(),
        "qty_lower": np.asarray(q_low_final, dtype=float).tolist(),
        "qty_upper": np.asarray(q_up_final, dtype=float).tolist(),
        "gmv": (p_sorted * q_mono).tolist(),
        "gmv_lower": (p_sorted * np.asarray(q_low_final, dtype=float)).tolist(),
        "gmv_upper": (p_sorted * np.asarray(q_up_final, dtype=float)).tolist(),
        "elasticity": eps_sorted.tolist(), "optimal_prices": [],
        "unitary_price": unitary_price,
        "elasticity_by_band": _summarize_elasticity_by_band(p_sorted, eps_sorted),
        "interval_level": interval_level,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Обоснование решения оптимизатора
# ═════════════════════════════════════════════════════════════════════════════

def _build_optimizer_rationale(history_pred: pd.DataFrame, itemcode: int) -> str:
    """
    Генерирует текстовое обоснование, почему оптимизатор выбрал именно эти цены.
    Анализирует паттерны в history_pred и формулирует объяснение.
    """
    if history_pred.empty:
        return "Нет данных для обоснования."

    lines = []

    # Средняя оптимальная цена
    avg_opt = history_pred["unitprice"].mean() if "unitprice" in history_pred.columns else None

    # Средняя историческая (динамический unit_price)
    hist_col = "unit_price_dynamic" if "unit_price_dynamic" in history_pred.columns else "historical_price"
    avg_hist = history_pred[hist_col].mean() if hist_col in history_pred.columns else None

    # Средняя эластичность
    avg_eps = history_pred["elasticity"].mean() if "elasticity" in history_pred.columns else None

    # Среднее изменение цены
    pct_col = "price_change_pct" if "price_change_pct" in history_pred.columns else "price_delta_pct"
    avg_pct = history_pred[pct_col].mean() if pct_col in history_pred.columns else None

    # Stock binding
    n_stock_bind = int(history_pred["stock_binding"].sum()) if "stock_binding" in history_pred.columns else 0
    n_days = len(history_pred)

    lines.append(f"**ITEMCODE {itemcode} — {n_days} дней симуляции**\n")

    # 1. Направление изменения цены
    if avg_opt is not None and avg_hist is not None and np.isfinite(avg_hist) and avg_hist > 0:
        direction = "выше" if avg_opt > avg_hist else "ниже"
        pct_diff = (avg_opt - avg_hist) / avg_hist * 100
        lines.append(
            f"🎯 **Оптимизатор рекомендовал цену {direction} исторической** в среднем на {abs(pct_diff):.1f}%. "
            f"Средняя рекомендованная: {avg_opt:.2f}, средняя историческая: {avg_hist:.2f}."
        )

    # 2. Объяснение через эластичность
    if avg_eps is not None and np.isfinite(avg_eps):
        if abs(avg_eps) > 1:
            lines.append(
                f"📐 **Эластичный спрос** (|ε| = {abs(avg_eps):.2f} > 1): спрос чувствителен к цене. "
                "Оптимизатор находит баланс между ценой и объёмом продаж — слишком высокая цена "
                "сильно срезает спрос и снижает маржу."
            )
        elif abs(avg_eps) < 0.5:
            lines.append(
                f"📐 **Неэластичный спрос** (|ε| = {abs(avg_eps):.2f} < 0.5): покупатели мало реагируют "
                "на изменение цены. Оптимизатор склонен повышать цену — потери в объёме минимальны, "
                "а маржа на единицу растёт."
            )
        else:
            lines.append(
                f"📐 **Умеренная эластичность** (|ε| = {abs(avg_eps):.2f}): оптимизатор балансирует "
                "между ценовой надбавкой и риском снижения спроса."
            )

    # 3. Складское ограничение
    if n_stock_bind > 0:
        lines.append(
            f"📦 **Складское ограничение активировалось в {n_stock_bind} из {n_days} дней**: "
            "в эти дни доступных запасов не хватало для реализации прогнозного спроса — "
            "цена была поднята выше аналитического оптимума, чтобы не допустить дефицита."
        )

    # 4. Метод эластичности
    if "elasticity_method" in history_pred.columns:
        methods = history_pred["elasticity_method"].value_counts()
        primary = methods.index[0] if len(methods) > 0 else "unknown"
        method_labels = {
            "demand_model_point": "ML demand_model на локальном окне",
            "fallback_baseprice": "базовая эластичность из elasticity_by_itemid.csv",
            "fallback_default": "дефолтное значение (-1.2) из-за нехватки данных",
        }
        label = method_labels.get(primary, primary)
        lines.append(f"🔬 **Источник эластичности**: {label} (основной метод для {methods.iloc[0]} дней из {n_days}).")
        if "fallback_default" in methods.index:
            n_fb = methods.get("fallback_default", 0)
            lines.append(
                f"⚠️ В {n_fb} дне(й) использовалось дефолтное значение ε=-1.2 из-за недостатка "
                "исторических данных в окне. Рекомендации за эти дни менее точны."
            )

    # 5. Итог
    if avg_opt is not None:
        ramsey_approx = None
        if avg_eps is not None and avg_eps < -1 and "cost" in history_pred.columns:
            avg_cost = history_pred["cost"].mean()
            if np.isfinite(avg_cost) and avg_cost > 0:
                ramsey_approx = round(avg_eps / (avg_eps + 1) * avg_cost, 2)

        if ramsey_approx is not None:
            lines.append(
                f"📊 Аналитический оптимум Рамсея (без ограничений): ~{ramsey_approx:.2f}. "
                f"Фактическая рекомендация {avg_opt:.2f} отличается из-за ограничений "
                "по складу, KVI-корректировки и сглаживания."
            )

    return "\n\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  Ядро симуляции
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(itemcode, first_day, n_days, progress_bar, status_text):
    first_day = pd.Timestamp(first_day).normalize()
    n_days = max(int(n_days), 1)
    default_last_day = first_day + pd.Timedelta(days=n_days)

    status_text.text("📦 Инициализация LST и загрузка данных SKU...")
    progress_bar.progress(5)
    pipe = LSTPipeline(itemcode=int(itemcode))

    # raw_item захватываем ДО препроцессора: UNITPRICE = статический (из CSV)
    # Для динамического unit_price = TOTALPRICE/AMOUNT используем _compute_daily_unit_price
    raw_item = pipe.data.copy() if isinstance(pipe.data, pd.DataFrame) else pd.DataFrame()
    if raw_item.empty:
        raise ValueError(f"По ITEMCODE={itemcode} нет данных в источнике")

    raw_item["DATE_"] = pd.to_datetime(raw_item["DATE_"], errors="coerce")
    raw_item["AMOUNT"] = pd.to_numeric(raw_item["AMOUNT"], errors="coerce")
    raw_item["UNITPRICE"] = pd.to_numeric(
        raw_item["UNITPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )
    raw_item["TOTALPRICE"] = pd.to_numeric(
        raw_item["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
    )

    status_text.text("🔄 Выполнение симуляции LST...")
    progress_bar.progress(35)
    sim_result = pipe.simulation(first_day=first_day, n_days=n_days, window_days=30, run_evaluation=False)

    progress_bar.progress(75)
    status_text.text("📊 Подготовка результатов UI...")

    raw_history = sim_result.get("history_pred")
    history_pred = raw_history.copy() if isinstance(raw_history, pd.DataFrame) else pd.DataFrame()

    baseline_table = sim_result.get("baseline", pd.DataFrame())
    if not isinstance(baseline_table, pd.DataFrame):
        baseline_table = pd.DataFrame()

    raw_last_day = sim_result.get("last_day")
    last_day = pd.Timestamp(cast(Any, raw_last_day)).normalize() if raw_last_day else default_last_day

    if len(history_pred) > 0:
        history_pred["DATE_"] = pd.to_datetime(history_pred["DATE_"], errors="coerce").dt.normalize()
        history_pred = history_pred.sort_values("DATE_").reset_index(drop=True)

        # Добавляем bounds-колонки если отсутствуют
        for low_col, src_col in [
            ("quantity_lower", "quantity"), ("quantity_upper", "quantity"),
            ("gmv_lower", "gmv"), ("gmv_upper", "gmv"),
            ("margin_lower", "margin"), ("margin_upper", "margin"),
        ]:
            if low_col not in history_pred.columns and src_col in history_pred.columns:
                history_pred[low_col] = history_pred[src_col]

        if "penalty_value" not in history_pred.columns:
            history_pred["penalty_value"] = 0.0

        # ----------------------------------------------------------------
        # historical_price = динамический unit_price из сырых данных
        # = TOTALPRICE / AMOUNT за каждый день
        # Это фактическая цена за единицу в чеке, меняется изо дня в день.
        # ----------------------------------------------------------------
        history_pred["historical_price"] = history_pred["DATE_"].apply(
            lambda d: _lookup_historical_price(raw_item, pd.Timestamp(d))
        )

        # ----------------------------------------------------------------
        # baseprice = статическая прайс-листовая цена (UNITPRICE из CSV)
        # Если в pipeline уже пришёл per-day baseprice — оставляем его.
        # Если нет — берём медиану UNITPRICE из сырых данных.
        # НЕ перезаписываем одним значением на весь период!
        # ----------------------------------------------------------------
        if "baseprice" not in history_pred.columns or history_pred["baseprice"].isna().all():
            # Fallback: статическая медиана по всему периоду из сырых данных
            static_bp = pd.to_numeric(raw_item["UNITPRICE"], errors="coerce").median()
            history_pred["baseprice"] = static_bp

        # price_delta_pct = оптимальная цена vs динамический unit_price
        if "price_delta_pct" not in history_pred.columns:
            ref = history_pred["historical_price"].replace(0, np.nan)
            history_pred["price_delta_pct"] = (history_pred["unitprice"] - ref) / ref * 100.0

    for col in SIGNAL_DIAGNOSTIC_COLS:
        if col not in history_pred.columns:
            history_pred[col] = np.nan

    actual_daily = _build_actual_daily_summary(raw_item, first_day, last_day)
    if len(actual_daily) > 0 and len(history_pred) > 0:
        actual_daily["DATE_"] = pd.to_datetime(actual_daily["DATE_"], errors="coerce").dt.normalize()
        # actual_price уже = динамический unit_price, поэтому НЕ дублируем в historical_price
        actual_daily_merge = actual_daily[["DATE_", "actual_amount", "actual_gmv"]].copy()
        history_pred = history_pred.merge(actual_daily_merge, on="DATE_", how="left")

    if pipe.data is not None and len(pipe.data) > 0:
        df_sim = pipe.data[(pipe.data["DATE_"] >= first_day) & (pipe.data["DATE_"] < last_day)].copy()
    else:
        df_sim = raw_item[(raw_item["DATE_"] >= first_day) & (raw_item["DATE_"] < last_day)].copy()

    if baseline_table.empty and len(history_pred) > 0:
        ml_res = SimResult(
            strategy_name="ML_Optimizer",
            total_gmv=float(history_pred["gmv"].sum()),
            total_margin=float(history_pred["margin"].sum()),
            avg_margin_pct=float(history_pred["margin_percent"].mean()) / 100,
        )
        baseline_table = ml_res.summary().to_frame().T.set_index("strategy")

    if "ML_Optimizer" not in baseline_table.index and len(history_pred) > 0:
        try:
            ml_strat = MLOptimizerBaseline(history_pred)
            ml_res = ml_strat.simulate(
                df=df_sim, demand_col="AMOUNT", price_col="UNITPRICE",
                cost_col="cost", date_col="DATE_", initial_stock=0.0,
            )
            baseline_table = pd.concat([
                baseline_table,
                ml_res.summary().to_frame().T.set_index("strategy"),
            ])
        except Exception as e:
            st.warning(f"⚠️ Не удалось добавить ML_Optimizer в baseline: {e}")

    status_text.text("📉 Построение кривой эластичности...")
    progress_bar.progress(88)
    elasticity_data = _build_elasticity_data_from_model(raw_item, first_day, n_bins=10)
    if elasticity_data is None:
        elasticity_data = {
            "prices": [], "qty": [], "qty_lower": [], "qty_upper": [],
            "gmv": [], "gmv_lower": [], "gmv_upper": [], "elasticity": [],
            "optimal_prices": history_pred["unitprice"].dropna().astype(float).tolist()
            if "unitprice" in history_pred.columns else [],
            "unitary_price": None, "elasticity_by_band": [], "interval_level": None,
        }
    else:
        elasticity_data["optimal_prices"] = (
            history_pred["unitprice"].dropna().astype(float).tolist()
            if "unitprice" in history_pred.columns else []
        )

    progress_bar.progress(100)
    status_text.text("✅ Готово!")
    return history_pred, baseline_table, elasticity_data, raw_item


# ═════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Price Optimizer", layout="wide")
st.title("🎯 Price Optimizer")
st.markdown("Выберите товар, задайте период и запустите прогноз.")

with st.sidebar:
    st.header("⚙️ Параметры")

    itemcodes = get_itemcodes()
    if not itemcodes:
        st.error("В источнике не найдено ITEMCODE. Проверьте data/Order_Details.csv")
        st.stop()

    selected_itemcode_raw = st.selectbox(
        "📦 ITEMCODE", options=itemcodes,
        index=itemcodes.index(107) if 107 in itemcodes else 0,
    )
    if selected_itemcode_raw is None:
        st.error("Не удалось выбрать ITEMCODE")
        st.stop()
    selected_itemcode = int(selected_itemcode_raw)

    @st.cache_data(ttl=3600, show_spinner="Загружаем диапазон дат...")
    def _get_global_date_range():
        if not ORDERS_PATH.exists():
            raw = load_raw_data()
            dt = pd.to_datetime(raw["DATE_"], errors="coerce").dropna()
        else:
            dt = pd.read_csv(ORDERS_PATH, usecols=["DATE_"], parse_dates=["DATE_"], low_memory=False)["DATE_"].dropna()
        if len(dt) == 0:
            today = pd.Timestamp.today().normalize().date()
            return today, today
        return dt.min().date(), dt.max().date()

    min_date, max_date = _get_global_date_range()

    start_date = st.date_input("📅 As-of date", value=max_date, min_value=min_date, max_value=max_date)
    n_days = st.slider("📆 Горизонт прогноза (дней)", min_value=1, max_value=30, value=10)

    st.divider()
    run_btn = st.button("🚀 Спрогнозировать", type="primary", use_container_width=True)

# ─── Основная область ────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    progress_bar = st.progress(0, text="Инициализация…")
    status_text = st.empty()

    try:
        first_day = pd.Timestamp(start_date)
        hist_pred, baseline_tbl, elast_data, raw_item_cache = run_simulation(
            itemcode=selected_itemcode, first_day=first_day, n_days=n_days,
            progress_bar=progress_bar, status_text=status_text,
        )
        st.session_state.results = {
            "history_pred": hist_pred, "baseline": baseline_tbl,
            "elasticity_data": elast_data, "itemcode": selected_itemcode,
            "period": f"{start_date} → {start_date + pd.Timedelta(days=n_days)}",
            "first_day": first_day.normalize(),
            "last_day": (first_day + pd.Timedelta(days=n_days)).normalize(),
            "raw_item": raw_item_cache,
        }
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        import traceback
        st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

# ─── Отображение результатов ─────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results

    st.divider()
    st.subheader(f"📊 Результаты: ITEMCODE {res['itemcode']}  |  {res['period']}")

    # 1. Baseline comparison
    st.markdown("### Сравнение стратегий")
    st.caption("Считается на дневном уровне по всему периоду и использует исторический спрос/склад.")
    st.dataframe(res["baseline"], use_container_width=True)

    # 2. ML vs лучший baseline
    st.markdown("### 📈 ML vs лучший baseline")

    if "ML_Optimizer" not in res["baseline"].index:
        st.error("❌ ML_Optimizer отсутствует в таблице baseline.")
        st.dataframe(res["baseline"], use_container_width=True)
    else:
        ml_row = res["baseline"].loc["ML_Optimizer"]
        other_strategies = res["baseline"].drop(index="ML_Optimizer")
        if len(other_strategies) == 0:
            st.warning("Нет baseline-стратегий для сравнения.")
            best_baseline_name, best_baseline = "N/A", None
        else:
            best_baseline_name = other_strategies["total_gmv"].idxmax()
            best_baseline = other_strategies.loc[best_baseline_name]

        def _delta(ml, base):
            if abs(base) < 1e-8:
                return 0.0
            return (ml - base) / abs(base) * 100

        if best_baseline is not None:
            delta_cols = st.columns(4)
            delta_cols[0].metric("Δ GMV", f"{_delta(ml_row['total_gmv'], best_baseline['total_gmv']):+.1f}%",
                help=f"ML: {ml_row['total_gmv']:,.0f} vs {best_baseline_name}: {best_baseline['total_gmv']:,.0f}")
            delta_cols[1].metric("Δ Margin", f"{_delta(ml_row['total_margin'], best_baseline['total_margin']):+.1f}%")
            delta_cols[2].metric("Δ Margin %", f"{_delta(ml_row['avg_margin_pct'], best_baseline['avg_margin_pct']):+.1f} п.п.")
            delta_cols[3].metric("Δ Fill Rate", f"{_delta(ml_row['fill_rate_%'], best_baseline['fill_rate_%']):+.1f}%")
            st.caption(f"Лучший baseline: **{best_baseline_name}**")

        # 3. Ключевые метрики ML
        st.markdown("### Ключевые метрики ML-оптимизатора")
        elast = res.get("elasticity_data", {})
        opt_prices = elast.get("optimal_prices", [])
        avg_optimal_price = float(np.mean(opt_prices)) if opt_prices else None
        unitary_price = elast.get("unitary_price")
        interval_level = elast.get("interval_level")

        hist_pred_df = res.get("history_pred")
        avg_price_delta_pct = None
        if hist_pred_df is not None and "price_delta_pct" in hist_pred_df.columns:
            valid_deltas = hist_pred_df["price_delta_pct"].dropna()
            if len(valid_deltas) > 0:
                avg_price_delta_pct = float(valid_deltas.mean())

        cols = st.columns(7)
        cols[0].metric("GMV", f"{ml_row['total_gmv']:,.0f}")
        cols[1].metric("Margin", f"{ml_row['total_margin']:,.0f}")
        cols[2].metric("Margin %", f"{ml_row['avg_margin_pct']:.1f}%")
        cols[3].metric("Fill Rate", f"{ml_row['fill_rate_%']:.1f}%")
        cols[4].metric("Avg Optimal Price", f"{avg_optimal_price:,.2f}" if avg_optimal_price else "N/A")
        cols[5].metric("Avg Δ vs Hist", f"{avg_price_delta_pct:+.1f}%" if avg_price_delta_pct is not None else "N/A",
            help="Разница ML цены vs дневного unit_price (TOTALPRICE/AMOUNT)")
        cols[6].metric("Unitary ε Price", f"{unitary_price:,.2f}" if unitary_price else "N/A",
            help="Цена где ε(P) = -1 (максимум выручки)")

        # 3b. Диагностика сигнала
        st.markdown("### 🔎 Диагностика сигнала SKU")
        st.caption("Показатели ~30 дней до каждого дня оптимизации.")
        if hist_pred_df is not None and len(hist_pred_df) > 0:
            latest_diag_df = hist_pred_df[SIGNAL_DIAGNOSTIC_COLS].dropna(how="all").tail(1)
            latest_diag = latest_diag_df.iloc[0] if len(latest_diag_df) else pd.Series(dtype=float)
        else:
            latest_diag = pd.Series(dtype=float)

        diag_cols = st.columns(5)
        diag_cols[0].metric("Price CV 30d",
            f"{latest_diag.get('price_cv_30d', float('nan')):.2f}" if len(latest_diag) else "N/A")
        diag_cols[1].metric("Price range 30d",
            f"{latest_diag.get('price_range_pct_30d', float('nan')) * 100:.1f}%" if len(latest_diag) else "N/A")
        diag_cols[2].metric("Price changes 30d",
            f"{latest_diag.get('price_change_share_30d', float('nan')) * 100:.1f}%" if len(latest_diag) else "N/A")
        diag_cols[3].metric("Selling days 30d",
            f"{latest_diag.get('selling_days_30d', float('nan')):.0f}" if len(latest_diag) else "N/A")
        diag_cols[4].metric("Demand CV 30d",
            f"{latest_diag.get('demand_cv_30d', float('nan')):.2f}" if len(latest_diag) else "N/A")

        if interval_level is not None:
            st.caption(f"Интервалы — residual-based bands уровня ~{interval_level:.0%}.")

    # 4. Графики эластичности
    st.markdown("### 📈 Графики эластичности")

    if elast.get("prices") and len(elast["prices"]) > 0:
        prices = elast["prices"]
        qty = elast["qty"]
        qty_lower = elast.get("qty_lower", qty)
        qty_upper = elast.get("qty_upper", qty)
        gmv = elast["gmv"]
        gmv_lower = elast.get("gmv_lower", gmv)
        gmv_upper = elast.get("gmv_upper", gmv)
        eps_vals = elast["elasticity"]
        opt_price = opt_prices[-1] if opt_prices else None
        unitary_p = elast.get("unitary_price")

        fig = make_subplots(rows=1, cols=3,
            subplot_titles=("Цена → Количество Q(P)", "Цена → GMV P·Q(P)", "Цена → Эластичность ε(P)"))

        fig.add_trace(go.Scatter(x=prices, y=qty, mode="lines", name="Количество",
            line=dict(color="steelblue", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=prices + prices[::-1], y=qty_upper + qty_lower[::-1],
            fill="toself", fillcolor="rgba(70,130,180,0.18)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        if opt_price:
            fig.add_vline(x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}", annotation_position="top right", row=1, col=1)
        if unitary_p:
            fig.add_vline(x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"|ε|=1: {unitary_p:.2f}", annotation_position="bottom left", row=1, col=1)
        fig.update_xaxes(title_text="Цена", row=1, col=1)
        fig.update_yaxes(title_text="Количество Q", row=1, col=1)

        fig.add_trace(go.Scatter(x=prices, y=gmv, mode="lines", name="GMV",
            line=dict(color="crimson", width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=prices + prices[::-1], y=gmv_upper + gmv_lower[::-1],
            fill="toself", fillcolor="rgba(220,20,60,0.14)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False), row=1, col=2)
        if opt_price:
            fig.add_vline(x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}", annotation_position="top right", row=1, col=2)
        if unitary_p:
            fig.add_vline(x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"|ε|=1: {unitary_p:.2f}", annotation_position="bottom left", row=1, col=2)
        fig.update_xaxes(title_text="Цена", row=1, col=2)
        fig.update_yaxes(title_text="GMV (P·Q)", row=1, col=2)

        fig.add_trace(go.Scatter(x=prices, y=eps_vals, mode="lines", name="ε(P)",
            line=dict(color="darkviolet", width=2.5)), row=1, col=3)
        fig.add_hline(y=-1, line_dash="dash", line_color="red", line_width=1.5,
            annotation_text="ε = −1", annotation_position="right", row=1, col=3)
        if opt_price:
            fig.add_vline(x=opt_price, line_dash="dash", line_color="green", line_width=2,
                annotation_text=f"P* = {opt_price:.2f}", annotation_position="top right", row=1, col=3)
        if unitary_p:
            fig.add_vline(x=unitary_p, line_dash="dot", line_color="orange", line_width=2,
                annotation_text=f"P_unitary = {unitary_p:.2f}", annotation_position="bottom left", row=1, col=3)
        fig.update_xaxes(title_text="Цена", row=1, col=3)
        fig.update_yaxes(title_text="Эластичность ε(P)", row=1, col=3)

        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🟢 **P\\*** — оптимальная цена  |  🟠 **|ε|=1** — максимум выручки  |  "
            "🔴 **ε = −1** — граница эластичного/неэластичного спроса"
        )

        band_table = pd.DataFrame(elast.get("elasticity_by_band", []))
        if len(band_table) > 0:
            st.markdown("### 📐 Эластичность по ценовым диапазонам")
            st.caption("Q1 — самые низкие цены в сетке.")
            st.dataframe(band_table, use_container_width=True, hide_index=True, column_config={
                "price_band": "Диапазон", "price_min": st.column_config.NumberColumn("Цена min", format="%.2f"),
                "price_max": st.column_config.NumberColumn("Цена max", format="%.2f"),
                "mean_elasticity": st.column_config.NumberColumn("Средняя ε", format="%.3f"),
                "n_points": st.column_config.NumberColumn("Точек", format="%d"),
            })
    else:
        st.caption("Ценовая сетка недоступна; показываю дневные графики.")
        daily_plot = res["history_pred"].copy()
        if len(daily_plot) > 0:
            daily_plot["DATE_"] = pd.to_datetime(daily_plot["DATE_"])
            fig_daily = make_subplots(rows=1, cols=3,
                subplot_titles=("Динамика цены", "Динамика количества", "Динамика GMV"))
            fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot.get("unitprice"),
                mode="lines+markers", name="ML price", line=dict(color="steelblue", width=2)), row=1, col=1)
            if "historical_price" in daily_plot.columns:
                fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot["historical_price"],
                    mode="lines+markers", name="Unit price (факт)", line=dict(color="gray", width=1.5, dash="dot")),
                    row=1, col=1)
            fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot.get("quantity"),
                mode="lines+markers", name="Qty", line=dict(color="darkgreen", width=2)), row=1, col=2)
            if "actual_amount" in daily_plot.columns:
                fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot["actual_amount"],
                    mode="lines+markers", name="Actual qty", line=dict(color="orange", width=1.5, dash="dot")),
                    row=1, col=2)
            fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot.get("gmv"),
                mode="lines+markers", name="GMV", line=dict(color="crimson", width=2)), row=1, col=3)
            if "actual_gmv" in daily_plot.columns:
                fig_daily.add_trace(go.Scatter(x=daily_plot["DATE_"], y=daily_plot["actual_gmv"],
                    mode="lines+markers", name="Actual GMV", line=dict(color="purple", width=1.5, dash="dot")),
                    row=1, col=3)
            fig_daily.update_layout(height=420, showlegend=True)
            for c, yt in [(1, "Цена"), (2, "Количество"), (3, "GMV")]:
                fig_daily.update_xaxes(title_text="Дата", row=1, col=c)
                fig_daily.update_yaxes(title_text=yt, row=1, col=c)
            st.plotly_chart(fig_daily, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════
    # 5. Связи товаров — объясняющий отчёт + матрица
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("### 🧩 Связи товаров: аналитический отчёт")

    rel_map = _load_relations_map()
    rel_matrix, family_items, role_summary = _build_relation_matrix(int(res["itemcode"]), rel_map)
    relation_rows = _build_relation_report(int(res["itemcode"]), role_summary, rel_map)

    if relation_rows:
        for rr in relation_rows:
            with st.expander(
                f"{rr['color']} {rr['role_emoji']} ITEMCODE {rr['related_itemcode']} — {rr['role_label']}",
                expanded=True,
            ):
                st.markdown(f"**Экономический смысл связи:**\n{rr['economic_meaning']}")
                st.markdown(f"**Рекомендация по ценообразованию:**\n{rr['price_advice']}")
                rev = rr["reverse_role"]
                if rev and rev != "—":
                    st.caption(f"Обратная связь: ITEMCODE {rr['related_itemcode']} видит текущий товар как: {rev}")
    else:
        st.caption("У данного товара нет зарегистрированных связей в substitute_map.json.")

    with st.expander("📋 Показать матрицу связей (сырой вид)"):
        st.caption("Строки — источник, столбцы — цель. Типы: companion, competitor/substitute, competitor/cannibal.")
        st.dataframe(rel_matrix, use_container_width=True)

    first_day_res = pd.Timestamp(res.get("first_day", pd.Timestamp.today())).normalize()
    last_day_res = pd.Timestamp(res.get("last_day", first_day_res + pd.Timedelta(days=10))).normalize()
    second_item, second_role = _pick_secondary_item(int(res["itemcode"]), role_summary)

    # ═════════════════════════════════════════════════════════════════════
    # 6. Взаимодействие товаров по ценам и спросу
    # ═════════════════════════════════════════════════════════════════════
    if second_item is not None:
        role_label_map = {"substitutes": "Субститут", "cannibals": "Каннибал", "complements": "Комплемент"}
        role_label_str = role_label_map.get(second_role, second_role)

        st.markdown(f"### 🔁 Взаимодействие: ITEMCODE {res['itemcode']} ↔ {second_item} ({role_label_str})")

        two_items_df = _load_group_data(tuple(sorted({int(res["itemcode"]), int(second_item)})))
        group_all_df = _load_group_data(tuple(sorted(set(int(x) for x in family_items))))

        charts = _build_demand_interaction_charts(
            target_ic=int(res["itemcode"]),
            second_ic=int(second_item),
            second_role=second_role,
            group_all_ics=[int(x) for x in family_items],
            two_items_df=two_items_df,
            group_df=group_all_df,
            first_day=first_day_res,
            last_day=last_day_res,
        )

        if charts is not None:
            fig1, fig2, fig3 = charts
            # График 1: Динамика цен
            st.markdown("#### 💹 Дневной Unit Price (TOTALPRICE/AMOUNT) по обоим товарам")
            st.caption(
                "Динамический unit_price отражает фактическую цену за единицу в чеке, "
                "вычисляемую как TOTALPRICE/AMOUNT за день. Позволяет увидеть ценовое взаимодействие."
            )
            st.plotly_chart(fig1, use_container_width=True)

            # График 2: Спрос + кросс-эффект
            st.markdown("#### 📊 Взаимодействие по спросу (кросс-эффект)")
            st.caption(
                "Синяя линия — спрос целевого товара (левая ось). "
                "Красная пунктирная — спрос связанного товара (правая ось). "
                "Серая пунктирная — цена целевого товара для наглядности кросс-эффекта."
            )
            st.plotly_chart(fig2, use_container_width=True)

            # График 3: Доля продаж
            if fig3 is not None:
                st.markdown("#### 📊 Доля продаж внутри группы по дням")
                st.caption(
                    "Stacked bar: сумма долей всех товаров в группе = 100% за каждый день. "
                    "Если цена целевого товара выросла и его спрос упал — его доля в баре уменьшится, "
                    "а доля каннибала/субститута вырастет."
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("В окне симуляции нет данных для построения графиков взаимодействия.")

    # ═════════════════════════════════════════════════════════════════════
    # 7. Групповая маржа
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("### 💰 Групповая маржа: история vs оптимизация")

    group_df_full = _load_group_data(tuple(sorted(set(int(x) for x in family_items))))
    group_margin_df = _build_group_margin_series(
        group_data=group_df_full, history_pred=res["history_pred"],
        selected_itemcode=int(res["itemcode"]),
        first_day=first_day_res, last_day=last_day_res,
    )

    if len(group_margin_df) > 0 and "group_margin_optimized" in group_margin_df.columns:
        total_hist = float(group_margin_df["group_margin_history"].sum())
        total_opt = float(group_margin_df["group_margin_optimized"].sum())
        delta_pct = 0.0 if abs(total_hist) < 1e-8 else (total_opt - total_hist) / abs(total_hist) * 100.0

        gm_cols = st.columns(3)
        gm_cols[0].metric("Историческая маржа группы", f"{total_hist:,.0f}")
        gm_cols[1].metric("Маржа группы (с оптимизацией)", f"{total_opt:,.0f}")
        gm_cols[2].metric("Δ маржи группы", f"{delta_pct:+.1f}%")

        # ── Баг 1 фикс: объясняем разрыв ────────────────────────────────
        # Разрыв между baseline_table ("маржа при исторической цене") и
        # group_margin_history — разные базы измерения:
        #   baseline: исторический спрос без складского ограничения
        #   group_margin: реализованные продажи, ограниченные складом
        if "ML_Optimizer" in res["baseline"].index:
            ml_margin = float(res["baseline"].loc["ML_Optimizer", "total_margin"])
            # Разница из-за складских ограничений: склад режет realized_qty
            stock_bound_days = int(res["history_pred"]["stock_binding"].sum()) \
                if "stock_binding" in res["history_pred"].columns else 0
            if stock_bound_days > 0 or abs(total_opt - ml_margin) > 1:
                st.info(
                    f"ℹ️ **Почему маржа в этом графике ({total_opt:,.0f}) "
                    f"расходится с baseline-таблицей ({ml_margin:,.0f})?**\n\n"
                    f"Baseline считает маржу по всей группе, используя **исторический спрос** "
                    f"(фактические продажи без ограничений). "
                    f"График выше показывает **реализованную** маржу ML-оптимизатора, "
                    f"где объём продаж ограничен доступными складскими остатками.\n\n"
                    + (f"🔒 Складское ограничение активировалось в **{stock_bound_days} из "
                       f"{len(res['history_pred'])} дней** — это главная причина разрыва. "
                       "При большем запасе оптимизатор реализовал бы больший объём."
                       if stock_bound_days > 0 else "")
                )

        fig_gm = go.Figure()
        fig_gm.add_trace(go.Scatter(x=group_margin_df["DATE_"], y=group_margin_df["group_margin_history"],
            mode="lines+markers", name="История (группа)"))
        fig_gm.add_trace(go.Scatter(x=group_margin_df["DATE_"], y=group_margin_df["group_margin_optimized"],
            mode="lines+markers", name="Оптимизация (группа)"))
        fig_gm.update_layout(height=340, xaxis_title="Дата", yaxis_title="Маржа")
        st.plotly_chart(fig_gm, use_container_width=True)
        st.caption("Для группы: выбранный SKU берётся из оптимизации, остальные — по историческому факту.")

    # ═════════════════════════════════════════════════════════════════════
    # 8. Прогноз модели по дням + колонки по связанным товарам
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("### 📅 Прогноз модели по дням")
    st.caption(
        "**Дневной Unit Price** = TOTALPRICE/AMOUNT из истории (динамическая цена за единицу).  "
        "**Base Price** = статическая прайс-листовая цена из CSV.  "
        "**ML Цена** = рекомендованная оптимизатором."
    )

    daily = res["history_pred"].copy()
    daily["DATE_"] = pd.to_datetime(daily["DATE_"]).dt.date

    # Добавляем данные по связанному товару в таблицу
    if second_item is not None and not two_items_df.empty:
        two = two_items_df.copy()
        two["DATE_norm"] = pd.to_datetime(two["DATE_"]).dt.normalize()
        two_sel = two[two["ITEMCODE"].astype("int64") == int(second_item)].copy()
        two_sel["AMOUNT"] = pd.to_numeric(two_sel["AMOUNT"], errors="coerce")
        two_sel["TOTALPRICE"] = pd.to_numeric(
            two_sel["TOTALPRICE"].astype(str).str.replace(",", ".", regex=False), errors="coerce",
        )

        if len(two_sel) > 0:
            rel_daily = (
                two_sel[two_sel["AMOUNT"] > 0]
                .groupby("DATE_norm", as_index=False)
                .apply(lambda g: pd.Series({
                    "related_amount": g["AMOUNT"].sum(),
                    "related_unit_price": _compute_daily_unit_price(g),
                }))
                .reset_index(drop=True)
            )
            if "DATE_norm" not in rel_daily.columns:
                rel_daily = (
                    two_sel[two_sel["AMOUNT"] > 0]
                    .groupby("DATE_norm")
                    .apply(lambda g: pd.Series({
                        "related_amount": g["AMOUNT"].sum(),
                        "related_unit_price": _compute_daily_unit_price(g),
                    }))
                    .reset_index()
                )
            # Мержим по дате
            daily["DATE_norm"] = pd.to_datetime(daily["DATE_"]).dt.normalize() if "DATE_" in daily.columns else pd.NaT
            daily = daily.merge(rel_daily, on="DATE_norm", how="left")
            daily.drop(columns=["DATE_norm"], inplace=True, errors="ignore")

    # Определяем доступные колонки
    display_cols = [
        "DATE_",
        "unitprice",               # ML рекомендованная цена
        "historical_price",        # Дневной unit_price (TOTALPRICE/AMOUNT) из истории
        "baseprice",               # Статическая base price из прайс-листа
        "price_delta_pct",         # Δ ML vs исторической
        "cost",
        "quantity", "quantity_lower", "quantity_upper",
        "gmv", "gmv_lower", "gmv_upper",
        "margin", "margin_lower", "margin_upper",
        "actual_amount", "actual_gmv",
        "margin_percent", "elasticity", "penalty_value",
        "related_unit_price",      # Дневной unit_price связанного товара
        "related_amount",          # Дневной объём продаж связанного товара
        "stock_binding",           # Был ли активен складской constraint
        "elasticity_method",       # Метод расчёта эластичности
    ] + SIGNAL_DIAGNOSTIC_COLS

    available_cols = [c for c in display_cols if c in daily.columns]

    col_config = {
        "DATE_": "Дата",
        "unitprice": st.column_config.NumberColumn("ML Цена", format="%.2f",
            help="Оптимальная цена от ML-оптимизатора"),
        "historical_price": st.column_config.NumberColumn("Дневной Unit Price", format="%.2f",
            help="TOTALPRICE/AMOUNT за день из истории — фактическая цена за единицу в чеке. Меняется изо дня в день."),
        "baseprice": st.column_config.NumberColumn("Base Price (прайс-лист)", format="%.2f",
            help="Статическая прайс-листовая цена из CSV (UNITPRICE). Не меняется при скидках."),
        "price_delta_pct": st.column_config.NumberColumn("Δ ML vs Hist", format="%+.1f%%",
            help="Разница ML цены vs дневного unit_price: +завышаем, -занижаем"),
        "cost": st.column_config.NumberColumn("Себестоимость", format="%.2f"),
        "quantity": st.column_config.NumberColumn("Прогноз кол-ва", format="%d"),
        "quantity_lower": st.column_config.NumberColumn("Кол-во low", format="%.2f"),
        "quantity_upper": st.column_config.NumberColumn("Кол-во high", format="%.2f"),
        "gmv": st.column_config.NumberColumn("Прогноз GMV", format="%.2f"),
        "gmv_lower": st.column_config.NumberColumn("GMV low", format="%.2f"),
        "gmv_upper": st.column_config.NumberColumn("GMV high", format="%.2f"),
        "margin": st.column_config.NumberColumn("Маржа", format="%.2f"),
        "margin_lower": st.column_config.NumberColumn("Маржа low", format="%.2f"),
        "margin_upper": st.column_config.NumberColumn("Маржа high", format="%.2f"),
        "actual_amount": st.column_config.NumberColumn("Факт. кол-во", format="%.2f"),
        "actual_gmv": st.column_config.NumberColumn("Факт. GMV", format="%.2f"),
        "margin_percent": st.column_config.NumberColumn("Маржа %", format="%.1f"),
        "elasticity": st.column_config.NumberColumn("ε(P*)", format="%.3f"),
        "penalty_value": st.column_config.NumberColumn("Soft penalty", format="%.2f"),
        "related_unit_price": st.column_config.NumberColumn(
            f"Unit Price {second_item or '—'} ({role_label_map.get(second_role,'') if second_role else ''})",
            format="%.2f",
            help="Дневной unit_price связанного товара (TOTALPRICE/AMOUNT)"
        ),
        "related_amount": st.column_config.NumberColumn(
            f"Продажи {second_item or '—'}",
            format="%.0f",
            help="Дневной объём продаж связанного товара (каннибал/субститут/комплемент)"
        ),
        "stock_binding": "Склад ограничивал",
        "elasticity_method": "Метод ε",
        "price_cv_30d": st.column_config.NumberColumn("Price CV 30d", format="%.2f"),
        "price_range_pct_30d": st.column_config.NumberColumn("Price range 30d", format="%.2f"),
        "price_change_share_30d": st.column_config.NumberColumn("Price changes 30d", format="%.2f"),
        "selling_days_30d": st.column_config.NumberColumn("Selling days 30d", format="%.0f"),
        "demand_cv_30d": st.column_config.NumberColumn("Demand CV 30d", format="%.2f"),
        "AMOUNT_mean_30d": st.column_config.NumberColumn("Avg demand 30d", format="%.2f"),
    }

    st.dataframe(
        daily[available_cols],
        use_container_width=True, hide_index=True,
        column_config={k: v for k, v in col_config.items() if k in available_cols},
    )

    # ═════════════════════════════════════════════════════════════════════
    # 9. Обоснование решения оптимизатора
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("### 🧠 Обоснование решения оптимизатора")
    st.caption("Автоматический анализ паттернов в результатах симуляции.")

    rationale = _build_optimizer_rationale(res["history_pred"], int(res["itemcode"]))
    st.markdown(rationale)

    # 10. Кнопка нового прогноза
    st.divider()
    if st.button("🔄 Новый прогноз", use_container_width=True):
        st.session_state.results = None
        st.rerun()
