# Optimization

Isolated ML pricing branch. It builds a compact `item-day-scenario` dataset, scores discrete price candidates, backtests the policy on rolling windows, and compares it with the elasticity fallback without changing the legacy pipeline.

## Boundary

- Legacy files stay untouched: `app.py`, `compute_elasticity.py`, `Elasticity/`, and historical result files.
- Optimization code lives in `Optimization/`, while generated artifacts are written to the shared `output/optimization/` folder.

## Inputs used

- `../data/full_data.csv`
- `../data/cost.csv`
- `../data/elasticity.csv`

The branch derives missing concepts such as behavioral features, relation proxies, and KVI proxies internally because dedicated source files for them are not present in the repository.

## Output layout

- `../output/optimization/logs/orchestrator.log`
- `../output/optimization/tables/item_day_dataset.csv`
- `../output/optimization/tables/relation_features.csv`
- `../output/optimization/tables/scenario_grid.csv`
- `../output/optimization/tables/scenario_scored.csv`
- `../output/optimization/tables/item_metadata.csv`
- `../output/optimization/models/policy_model.pkl`
- `../output/optimization/models/feature_schema.json`
- `../output/optimization/reports/rolling_model_predictions.csv`
- `../output/optimization/reports/rolling_backtest.csv`
- `../output/optimization/reports/fallback_comparison.csv`
- `../output/optimization/reports/guardrails_violations.csv`
- `../output/optimization/reports/summary.json`
- `../output/optimization/manifests/run_manifest.json`

## Model

The default model is `LightGBMRegressor`. Install it if needed:

```bash
python -m pip install lightgbm
```

## Running

From the repository root:

```bash
python Optimization/orchestrator.py
```

Quick dry-run on a subset:

```bash
python Optimization/orchestrator.py --max-rows 200000 --max-items 20
```

Dashboard:

```bash
streamlit run Optimization/dashboard.py
```

The dashboard shows:
- overall verdict and strategy comparison
- model card with LightGBM params, utility weights, feature count, rolling metrics, and feature importance
- daily ML vs fallback vs current strategy charts
- item/date explorer with a forecast-horizon slider, chosen prices, and full scenario table for the selected day
- price curves for demand `Q(P)`, GMV `P·Q(P)`, elasticity `ε(P)`, and utility by candidate price
- artifact manifest from the isolated ML branch

## Notes

- `full_data.csv` is large, so dataset building reads it in chunks.
- Realized price is now recalculated from `TOTALPRICE / AMOUNT`; the raw `UNITPRICE` from source data is preserved as a base-price reference.
- Price scenarios are discrete, not continuous.
- KVI and anchor items get tighter price-change guardrails.
- The fallback comparison uses the same discrete scenario space, but scores candidates with elasticity-based logic instead of the ML model.
