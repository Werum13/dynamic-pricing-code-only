from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).parent
WORKSPACE_ROOT = BASE_DIR.parent
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "optimization"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"


BACKTEST_USECOLS = [
    "fold",
    "ITEMCODE",
    "DATE_",
    "ml_candidate_price",
    "price_delta_current_pct",
    "pseudo_margin_7d",
    "pseudo_revenue_7d",
    "pseudo_qty_7d",
    "pseudo_penetration_drift",
    "pseudo_basket_share_drift",
    "is_kvi",
    "is_anchor",
    "candidate_price",
    "pseudo_margin_7d_fallback",
    "pseudo_revenue_7d_fallback",
    "pseudo_qty_7d_fallback",
    "candidate_price_current",
    "pseudo_margin_7d_current",
    "pseudo_revenue_7d_current",
    "pseudo_qty_7d_current",
]

SCENARIO_USECOLS = [
    "DATE_",
    "ITEMCODE",
    "CATEGORY1",
    "CATEGORY2",
    "current_price",
    "candidate_price",
    "scenario_delta",
    "price_delta_current_pct",
    "candidate_margin_pct",
    "scenario_is_admissible",
    "is_current_price_scenario",
    "target_utility",
    "fallback_utility",
    "pseudo_margin_7d",
    "pseudo_revenue_7d",
    "pseudo_qty_7d",
    "substitute_pressure",
    "cannibal_risk",
    "category_competition_proxy",
    "kvi_score",
    "is_kvi",
    "is_anchor",
]


st.set_page_config(
    page_title="ML Price Policy v1",
    page_icon="📈",
    layout="wide",
)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return _read_json(BASE_DIR / "config.json")


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    return _read_json(REPORTS_DIR / "summary.json")


@st.cache_data(show_spinner=False)
def load_feature_schema() -> dict:
    return _read_json(MODELS_DIR / "feature_schema.json")


@st.cache_data(show_spinner=False)
def load_run_manifest() -> dict:
    return _read_json(MANIFESTS_DIR / "run_manifest.json")


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "fallback_comparison.csv")


@st.cache_data(show_spinner=False)
def load_rolling_metrics() -> pd.DataFrame:
    path = REPORTS_DIR / "rolling_model_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "rolling_backtest.csv", usecols=BACKTEST_USECOLS, parse_dates=["DATE_"])


@st.cache_data(show_spinner=False)
def load_scenarios() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "scenario_scored.csv", usecols=SCENARIO_USECOLS, parse_dates=["DATE_"])


@st.cache_data(show_spinner=False)
def load_item_metadata() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "item_metadata.csv")


def _require_outputs() -> None:
    needed = [
        REPORTS_DIR / "summary.json",
        REPORTS_DIR / "fallback_comparison.csv",
        REPORTS_DIR / "rolling_backtest.csv",
        TABLES_DIR / "scenario_scored.csv",
        TABLES_DIR / "item_metadata.csv",
        MODELS_DIR / "feature_schema.json",
        MANIFESTS_DIR / "run_manifest.json",
    ]
    missing = [str(path.relative_to(WORKSPACE_ROOT)) for path in needed if not path.exists()]
    if missing:
        st.error("Не хватает артефактов ML-ветки.")
        st.code("\n".join(missing))
        st.caption("Сначала запустите: `python Optimization/orchestrator.py`")
        st.stop()


def _metric_delta_text(value: float | None, scale: float = 100.0, suffix: str = "%") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * scale:+.1f}{suffix}"


def _policy_label_map() -> dict[str, str]:
    return {
        "ml_policy": "ML policy",
        "elasticity_fallback": "Elasticity fallback",
        "current_price": "Current price",
    }


def _format_policy_table(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = _policy_label_map()
    table = df.copy()
    table["policy"] = table["policy"].map(rename_map).fillna(table["policy"])
    return table


def _build_daily_strategy_frame(backtest: pd.DataFrame) -> pd.DataFrame:
    ml = (
        backtest.groupby("DATE_", as_index=False)
        .agg(
            price=("ml_candidate_price", "mean"),
            margin=("pseudo_margin_7d", "sum"),
            revenue=("pseudo_revenue_7d", "sum"),
            qty=("pseudo_qty_7d", "sum"),
        )
        .assign(policy="ML policy")
    )
    fallback = (
        backtest.groupby("DATE_", as_index=False)
        .agg(
            price=("candidate_price", "mean"),
            margin=("pseudo_margin_7d_fallback", "sum"),
            revenue=("pseudo_revenue_7d_fallback", "sum"),
            qty=("pseudo_qty_7d_fallback", "sum"),
        )
        .assign(policy="Elasticity fallback")
    )
    current = (
        backtest.groupby("DATE_", as_index=False)
        .agg(
            price=("candidate_price_current", "mean"),
            margin=("pseudo_margin_7d_current", "sum"),
            revenue=("pseudo_revenue_7d_current", "sum"),
            qty=("pseudo_qty_7d_current", "sum"),
        )
        .assign(policy="Current price")
    )
    return pd.concat([ml, fallback, current], ignore_index=True)


def _get_item_dates(backtest: pd.DataFrame, itemcode: int) -> list:
    return sorted(backtest.loc[backtest["ITEMCODE"] == itemcode, "DATE_"].dt.date.unique().tolist())


def _ensure_sidebar_state(backtest: pd.DataFrame) -> tuple[list[int], int, list]:
    itemcodes = sorted(backtest["ITEMCODE"].dropna().astype(int).unique().tolist())
    if not itemcodes:
        st.error("В rolling_backtest.csv нет доступных ITEMCODE.")
        st.stop()

    if "selected_item" not in st.session_state or st.session_state["selected_item"] not in itemcodes:
        st.session_state["selected_item"] = itemcodes[0]

    item_dates = _get_item_dates(backtest, int(st.session_state["selected_item"]))
    if not item_dates:
        st.error("Для выбранного ITEMCODE нет доступных дат.")
        st.stop()

    if "selected_date" not in st.session_state or st.session_state["selected_date"] not in item_dates:
        st.session_state["selected_date"] = item_dates[-1]

    if "forecast_days" not in st.session_state:
        st.session_state["forecast_days"] = min(7, max(1, len(item_dates)))

    return itemcodes, int(st.session_state["selected_item"]), item_dates


def render_overview(summary: dict, comparison: pd.DataFrame, daily_strategies: pd.DataFrame) -> None:
    st.subheader("Общая сводка")
    cols = st.columns(5)
    cols[0].metric("Verdict", str(summary.get("verdict", "n/a")).upper())
    cols[1].metric("Margin uplift vs fallback", _metric_delta_text(summary.get("margin_uplift_vs_fallback")))
    cols[2].metric("Margin uplift vs current", _metric_delta_text(summary.get("margin_uplift_vs_current")))
    cols[3].metric("Penetration drift", _metric_delta_text(summary.get("avg_penetration_drift")))
    cols[4].metric("Guardrail issue rate", _metric_delta_text(summary.get("guardrail_issue_rate")))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Сравнение стратегий")
        st.dataframe(_format_policy_table(comparison), use_container_width=True, hide_index=True)
    with right:
        fig = px.bar(
            _format_policy_table(comparison),
            x="policy",
            y=["total_margin", "total_revenue", "total_qty"],
            barmode="group",
            title="Итоги по стратегиям",
        )
        fig.update_layout(height=380, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        daily_strategies,
        x="DATE_",
        y="margin",
        color="policy",
        title="Дневная суммарная маржа по стратегиям",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        daily_strategies,
        x="DATE_",
        y="price",
        color="policy",
        title="Средняя выбранная цена по дням",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def render_model_card(config: dict, feature_schema: dict, feature_importance: pd.DataFrame, rolling_metrics: pd.DataFrame) -> None:
    st.subheader("Model card")
    cols = st.columns(4)
    cols[0].metric("Model", feature_schema.get("model_name", "n/a"))
    cols[1].metric("Random state", str(feature_schema.get("random_state", "n/a")))
    cols[2].metric("Feature count", len(feature_schema.get("feature_columns", [])))
    cols[3].metric("Scenario candidates", len(config["scenarios"]["candidate_deltas"]))

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("### Utility weights")
        st.dataframe(
            pd.DataFrame(
                [{"weight": key, "value": value} for key, value in config["model"]["utility_weights"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Model params")
        st.dataframe(
            pd.DataFrame(
                [{"param": key, "value": value} for key, value in config["model"]["params"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("### Scenario guardrails")
        guardrails = pd.DataFrame(
            [
                {"parameter": "candidate_deltas", "value": ", ".join(f"{delta:+.0%}" for delta in config["scenarios"]["candidate_deltas"])},
                {"parameter": "min_margin_pct", "value": f"{config['scenarios']['min_margin_pct']:.1%}"},
                {"parameter": "max_abs_change_pct", "value": f"{config['scenarios']['max_abs_change_pct']:.1%}"},
                {"parameter": "kvi_max_abs_change_pct", "value": f"{config['scenarios']['kvi_max_abs_change_pct']:.1%}"},
                {"parameter": "anchor_max_abs_change_pct", "value": f"{config['scenarios']['anchor_max_abs_change_pct']:.1%}"},
            ]
        )
        st.dataframe(guardrails, use_container_width=True, hide_index=True)

    st.markdown("### Rolling validation")
    if rolling_metrics.empty:
        st.info("Файл rolling_model_metrics.csv пока пуст.")
    else:
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Folds", int(len(rolling_metrics)))
        metrics_cols[1].metric("Avg MAE", f"{rolling_metrics['mae'].mean():.4f}")
        metrics_cols[2].metric("Min / Max train rows", f"{rolling_metrics['train_rows'].min()} / {rolling_metrics['train_rows'].max()}")
        fig = px.line(rolling_metrics, x="fold", y="mae", markers=True, title="MAE по rolling folds")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rolling_metrics, use_container_width=True, hide_index=True)

    st.markdown("### Feature importance")
    if feature_importance.empty:
        st.info("Feature importance пока недоступна.")
    else:
        top_importance = feature_importance.sort_values("importance_mean", ascending=False).head(20)
        fig = px.bar(
            top_importance.sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std" if "importance_std" in top_importance.columns else None,
            title="Top-20 features",
        )
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)


def _estimate_elasticity_curve(price_series: pd.Series, qty_series: pd.Series) -> pd.Series:
    price = price_series.to_numpy(dtype=float)
    qty = qty_series.to_numpy(dtype=float)
    if len(price) < 2:
        return pd.Series([float("nan")] * len(price_series), index=price_series.index)
    dq_dp = pd.Series(qty).interpolate(limit_direction="both").pipe(
        lambda s: pd.Series(
            np.gradient(s.to_numpy(dtype=float), price),
            index=s.index,
        )
    )
    elasticity = dq_dp.to_numpy(dtype=float) * price / pd.Series(qty).replace(0, pd.NA).astype(float).fillna(1e-8).to_numpy(dtype=float)
    return pd.Series(elasticity, index=price_series.index)


def render_item_explorer(
    itemcode: int,
    selected_date: pd.Timestamp,
    forecast_days: int,
    backtest: pd.DataFrame,
    scenarios: pd.DataFrame,
    item_meta: pd.DataFrame,
) -> None:
    st.subheader("Item explorer")
    item_backtest = backtest[backtest["ITEMCODE"] == itemcode].sort_values("DATE_").copy()
    item_scenarios = scenarios[(scenarios["ITEMCODE"] == itemcode)].sort_values(["DATE_", "candidate_price"]).copy()
    item_info = item_meta[item_meta["ITEMCODE"] == itemcode].head(1)

    if item_info.empty or item_backtest.empty:
        st.warning("Для выбранного товара нет данных в rolling backtest.")
        return

    meta = item_info.iloc[0]
    period_end = selected_date + pd.Timedelta(days=max(int(forecast_days) - 1, 0))
    period_backtest = item_backtest[(item_backtest["DATE_"] >= selected_date) & (item_backtest["DATE_"] <= period_end)].copy()
    info_cols = st.columns(6)
    info_cols[0].metric("ITEMCODE", int(meta["ITEMCODE"]))
    info_cols[1].metric("ITEMID", int(meta["ITEMID"]))
    info_cols[2].metric("CATEGORY1", str(meta["CATEGORY1"]))
    info_cols[3].metric("CATEGORY2", str(meta["CATEGORY2"]))
    info_cols[4].metric("KVI score", f"{meta['kvi_score']:.3f}")
    info_cols[5].metric("Flags", f"KVI={int(meta['is_kvi'])} | Anchor={int(meta['is_anchor'])}")

    date_backtest = item_backtest[item_backtest["DATE_"] == selected_date].copy()
    date_scenarios = item_scenarios[item_scenarios["DATE_"] == selected_date].copy()

    if not period_backtest.empty:
        period_cols = st.columns(6)
        period_cols[0].metric("Прогнозный горизонт", f"{forecast_days} дн.")
        period_cols[1].metric("ML margin", f"{period_backtest['pseudo_margin_7d'].sum():.0f}")
        period_cols[2].metric("Fallback margin", f"{period_backtest['pseudo_margin_7d_fallback'].sum():.0f}")
        period_cols[3].metric("Current margin", f"{period_backtest['pseudo_margin_7d_current'].sum():.0f}")
        period_cols[4].metric("ML revenue", f"{period_backtest['pseudo_revenue_7d'].sum():.0f}")
        period_cols[5].metric("ML qty", f"{period_backtest['pseudo_qty_7d'].sum():.1f}")

    if not date_backtest.empty:
        row = date_backtest.iloc[0]
        cols = st.columns(6)
        cols[0].metric("ML price", f"{row['ml_candidate_price']:.2f}")
        cols[1].metric("Fallback price", f"{row['candidate_price']:.2f}")
        cols[2].metric("Current price", f"{row['candidate_price_current']:.2f}")
        cols[3].metric("ML qty 7d", f"{row['pseudo_qty_7d']:.1f}")
        cols[4].metric("ML revenue 7d", f"{row['pseudo_revenue_7d']:.0f}")
        cols[5].metric("ML margin 7d", f"{row['pseudo_margin_7d']:.0f}")

    timeline = pd.DataFrame(
        {
            "DATE_": item_backtest["DATE_"],
            "ML price": item_backtest["ml_candidate_price"],
            "Elasticity fallback": item_backtest["candidate_price"],
            "Current price": item_backtest["candidate_price_current"],
        }
    ).melt(id_vars="DATE_", var_name="policy", value_name="price")
    fig = px.line(timeline, x="DATE_", y="price", color="policy", title=f"Цены во времени для ITEMCODE {itemcode}")
    fig.add_vrect(x0=selected_date, x1=period_end, fillcolor="rgba(30,136,229,0.12)", line_width=0)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    perf = pd.DataFrame(
        {
            "DATE_": item_backtest["DATE_"],
            "ML margin": item_backtest["pseudo_margin_7d"],
            "Fallback margin": item_backtest["pseudo_margin_7d_fallback"],
            "Current margin": item_backtest["pseudo_margin_7d_current"],
        }
    ).melt(id_vars="DATE_", var_name="policy", value_name="margin")
    fig = px.line(perf, x="DATE_", y="margin", color="policy", title="Маржа 7d по стратегиям")
    fig.add_vrect(x0=selected_date, x1=period_end, fillcolor="rgba(30,136,229,0.12)", line_width=0)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ценовые кривые на выбранную дату")
    if date_scenarios.empty:
        st.info("Для выбранной даты нет сценариев.")
        return

    selected_row = date_backtest.iloc[0] if not date_backtest.empty else None
    curve = date_scenarios.sort_values("candidate_price").copy()
    curve["elasticity_curve"] = _estimate_elasticity_curve(curve["candidate_price"], curve["pseudo_qty_7d"])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Цена → Спрос", "Цена → GMV", "Цена → Эластичность"),
    )
    fig.add_trace(
        go.Scatter(x=curve["candidate_price"], y=curve["pseudo_qty_7d"], mode="lines+markers", name="Demand Q(P)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["candidate_price"], y=curve["pseudo_revenue_7d"], mode="lines+markers", name="GMV P·Q(P)"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=curve["candidate_price"], y=curve["elasticity_curve"], mode="lines+markers", name="Elasticity ε(P)"),
        row=1,
        col=3,
    )
    fig.add_hline(y=-1, line_dash="dash", line_color="red", row=1, col=3)
    if selected_row is not None:
        fig.add_vline(x=float(selected_row["ml_candidate_price"]), line_dash="dash", line_color="green", row=1, col=1)
        fig.add_vline(x=float(selected_row["ml_candidate_price"]), line_dash="dash", line_color="green", row=1, col=2)
        fig.add_vline(x=float(selected_row["ml_candidate_price"]), line_dash="dash", line_color="green", row=1, col=3)
        fig.add_vline(x=float(selected_row["candidate_price"]), line_dash="dot", line_color="orange", row=1, col=1)
        fig.add_vline(x=float(selected_row["candidate_price"]), line_dash="dot", line_color="orange", row=1, col=2)
        fig.add_vline(x=float(selected_row["candidate_price"]), line_dash="dot", line_color="orange", row=1, col=3)
    fig.update_layout(height=420, showlegend=False)
    fig.update_xaxes(title_text="Цена", row=1, col=1)
    fig.update_xaxes(title_text="Цена", row=1, col=2)
    fig.update_xaxes(title_text="Цена", row=1, col=3)
    fig.update_yaxes(title_text="Спрос", row=1, col=1)
    fig.update_yaxes(title_text="GMV", row=1, col=2)
    fig.update_yaxes(title_text="ε(P)", row=1, col=3)
    st.plotly_chart(fig, use_container_width=True)

    utility_fig = go.Figure()
    utility_fig.add_trace(
        go.Scatter(
            x=curve["candidate_price"],
            y=curve["target_utility"],
            mode="lines+markers",
            name="Target utility",
        )
    )
    utility_fig.add_trace(
        go.Scatter(
            x=curve["candidate_price"],
            y=curve["fallback_utility"],
            mode="lines+markers",
            name="Fallback utility",
        )
    )
    if selected_row is not None:
        utility_fig.add_vline(x=float(selected_row["ml_candidate_price"]), line_dash="dash", line_color="green", annotation_text="ML chosen")
        utility_fig.add_vline(x=float(selected_row["candidate_price"]), line_dash="dot", line_color="orange", annotation_text="Fallback chosen")
    utility_fig.update_layout(title="Utility по ценовым кандидатам", height=360, xaxis_title="Candidate price", yaxis_title="Utility")
    st.plotly_chart(utility_fig, use_container_width=True)

    scenario_table = date_scenarios[
        [
            "candidate_price",
            "scenario_delta",
            "scenario_is_admissible",
            "candidate_margin_pct",
            "target_utility",
            "fallback_utility",
            "pseudo_qty_7d",
            "pseudo_revenue_7d",
            "pseudo_margin_7d",
            "substitute_pressure",
            "cannibal_risk",
        ]
    ].sort_values("candidate_price")
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)


def render_strategy_deepdive(backtest: pd.DataFrame, comparison: pd.DataFrame) -> None:
    st.subheader("Стратегии и drift-метрики")
    fig = px.scatter(
        _format_policy_table(comparison),
        x="avg_abs_price_change",
        y="total_margin",
        size="total_revenue",
        color="policy",
        hover_data=["avg_penetration_drift", "avg_basket_share_drift", "guardrail_violation_rate"],
        title="Компромисс: изменение цены vs суммарная маржа",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    kvi = backtest.groupby("DATE_", as_index=False).agg(
        kvi_price_change=("price_delta_current_pct", lambda s: s[backtest.loc[s.index, "is_kvi"] == 1].abs().mean()),
        anchor_price_change=("price_delta_current_pct", lambda s: s[backtest.loc[s.index, "is_anchor"] == 1].abs().mean()),
    )
    fig = px.line(
        kvi.melt(id_vars="DATE_", var_name="segment", value_name="avg_abs_price_change"),
        x="DATE_",
        y="avg_abs_price_change",
        color="segment",
        title="Среднее абсолютное изменение цены для KVI / anchor",
    )
    fig.update_layout(height=340)
    st.plotly_chart(fig, use_container_width=True)


def render_artifacts(run_manifest: dict) -> None:
    st.subheader("Артефакты запуска")
    artifacts = pd.DataFrame(
        [{"artifact": key, "path": value} for key, value in run_manifest.get("artifacts", {}).items()]
    )
    st.dataframe(artifacts, use_container_width=True, hide_index=True)


def main() -> None:
    _require_outputs()
    config = load_config()
    summary = load_summary()
    comparison = load_comparison()
    backtest = load_backtest()
    scenarios = load_scenarios()
    item_meta = load_item_metadata()
    feature_schema = load_feature_schema()
    run_manifest = load_run_manifest()
    feature_importance = load_feature_importance()
    rolling_metrics = load_rolling_metrics()
    daily_strategies = _build_daily_strategy_frame(backtest)

    with st.sidebar:
        st.header("⚙️ Параметры")
        itemcodes, _, item_dates = _ensure_sidebar_state(backtest)

        if st.button("🔄 Reload artifacts", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        selected_item = st.selectbox("📦 ITEMCODE", itemcodes, key="selected_item")
        item_dates = _get_item_dates(backtest, int(selected_item))
        if st.session_state.get("selected_date") not in item_dates:
            st.session_state["selected_date"] = item_dates[-1]
        selected_date = st.selectbox("📅 Дата", item_dates, key="selected_date")
        st.divider()
        st.caption("Запуск ML-ветки:")
        st.code("python Optimization/orchestrator.py", language="bash")
        st.caption("Запуск дашборда:")
        st.code("streamlit run Optimization/dashboard.py", language="bash")
        max_horizon = min(30, max(1, len(item_dates))) if item_dates else 1
        if st.session_state.get("forecast_days", 1) > max_horizon:
            st.session_state["forecast_days"] = max_horizon
        forecast_days = st.slider("📆 Длина прогноза (дней)", min_value=1, max_value=max_horizon, key="forecast_days")
        st.caption("Смена ITEMCODE и даты перестраивает экран сразу. Кнопка Reload нужна только для перечитывания файлов с диска.")

    st.title("📈 ML Price Policy v1 Dashboard")
    st.caption("Изолированный интерфейс для новой ML-ветки. Legacy pipeline не затрагивается.")

    tabs = st.tabs(
        [
            "Overview",
            "Model card",
            "Strategy deep-dive",
            "Item explorer",
            "Artifacts",
        ]
    )
    with tabs[0]:
        render_overview(summary, comparison, daily_strategies)
    with tabs[1]:
        render_model_card(config, feature_schema, feature_importance, rolling_metrics)
    with tabs[2]:
        render_strategy_deepdive(backtest, comparison)
    with tabs[3]:
        render_item_explorer(int(selected_item), pd.Timestamp(selected_date), int(forecast_days), backtest, scenarios, item_meta)
    with tabs[4]:
        render_artifacts(run_manifest)


if __name__ == "__main__":
    main()
