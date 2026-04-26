import json
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELASTICITY_DIR = Path(__file__).resolve().parent
for p in [PROJECT_ROOT, ELASTICITY_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from Elasticity.DataPreprocessor import preprocessor
    from Elasticity.ETL import etl_with_demand_target
    from Elasticity.DemandModel import demand_model
    from Elasticity.Evaluation import ElasticityEvaluator
    from Elasticity.Baseline import compare_baselines
    from Elasticity.data_sources import load_elasticity_source_data
except ImportError:
    from DataPreprocessor import preprocessor
    from ETL import etl_with_demand_target
    from DemandModel import demand_model
    from Evaluation import ElasticityEvaluator
    from Baseline import compare_baselines
    from data_sources import load_elasticity_source_data

from Elasticity.item_price_analyzer import analyze_item as analyze_item_policy
from Warehouse.byer import init_warehouse, update_warehouse_day

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Кэширование моделей
# ---------------------------------------------------------------------------

def _load_model_cache(base_dir: Path) -> dict:
    path = base_dir / "model_cache.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_model_cache(base_dir: Path, cache: dict) -> None:
    with open(base_dir / "model_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _model_dir(base_dir: Path) -> Path:
    d = base_dir / "models"
    d.mkdir(exist_ok=True)
    return d


def _model_path(base_dir: Path, itemcode, date: pd.Timestamp) -> Path:
    return _model_dir(base_dir) / f"model_{itemcode}_{date.strftime('%Y-%m-%d')}.pkl"


def _need_retrain(last_trained_str, current_date: pd.Timestamp) -> bool:
    if last_trained_str is None:
        return True
    return (current_date - pd.to_datetime(last_trained_str)).days >= 7


def _load_cached_model(base_dir: Path, itemcode, cache: dict):
    key = str(itemcode)
    if key not in cache:
        return None, None
    entry = cache[key]
    path = _model_path(base_dir, itemcode, pd.to_datetime(entry["last_trained"]))
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model, entry["last_trained"]


def _save_model(base_dir: Path, itemcode, date: pd.Timestamp, model, cache: dict) -> dict:
    path = _model_path(base_dir, itemcode, date)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    cache[str(itemcode)] = {
        "last_trained": date.strftime("%Y-%m-%d"),
        "model_path": str(path),
    }
    _save_model_cache(base_dir, cache)
    return cache


# ---------------------------------------------------------------------------
# Обучение / загрузка модели
# ---------------------------------------------------------------------------

def _get_model(
    base_dir: Path,
    itemcode,
    day: pd.Timestamp,
    train_df: pd.DataFrame,
    target: pd.Series,
    cache: dict,
) -> tuple:
    model, last_trained_str = _load_cached_model(base_dir, itemcode, cache)
    is_monday = day.weekday() == 0

    if model is None or _need_retrain(last_trained_str, day) or is_monday:
        model = demand_model(train=train_df, target=target)
        cache = _save_model(base_dir, itemcode, day, model, cache)

    return model, cache


# ---------------------------------------------------------------------------
# Оценка модели (hold-out + walk-forward CV)
# ---------------------------------------------------------------------------

def _evaluate_model(df_etl: pd.DataFrame, base_dir: Path) -> None:
    if len(df_etl) < 50:
        print(f"Insufficient data for evaluation ({len(df_etl)} rows, need >= 50). Skipped.")
        return

    evaluator = ElasticityEvaluator(
        df=df_etl,
        target_col="AMOUNT_0D_target",
        date_col="DATE_",
        test_fraction=0.2,
    )

    print("\nModel evaluation — temporal hold-out 80/20")
    try:
        metrics = evaluator.temporal_split_eval()
        print(metrics)
    except Exception as e:
        print(f"Hold-out failed: {e}")

    print("\nModel evaluation — walk-forward CV (5 folds)")
    try:
        cv_table = evaluator.walk_forward_eval(n_splits=5)
        print(cv_table.to_string())
        cv_table.to_csv(base_dir / "evaluation_cv.csv")
    except Exception as e:
        print(f"Walk-forward CV failed: {e}")


# ---------------------------------------------------------------------------
# Baseline-сравнение
# ---------------------------------------------------------------------------

def _run_baselines(
    pipeline,
    df_sim_period: pd.DataFrame,
    itemcode: int,
    model_demand=None,
    df_for_elasticity=None,
) -> pd.DataFrame:
    demand_fn = None
    if model_demand is not None and df_for_elasticity is not None:
        def demand_fn(day, price, cost):
            row = df_for_elasticity[df_for_elasticity["DATE_"] <= day]
            if len(row) == 0:
                return 0.0
            template = row.iloc[[-1]].drop(
                columns=["DATE_", "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
                         "AMOUNT_0D_target", "AMOUNT_1D_target", "AMOUNT_7D_target",
                         "AMOUNT_15D_target", "AMOUNT_30D_target",
                         "AMOUNT_1D", "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D"],
                errors="ignore",
            ).copy()
            template["UNITPRICE"] = price
            template["Id"] = 0
            pred = model_demand.predict(template)
            return float(pred["pred_Q"].iloc[0])

    try:
        table = compare_baselines(
            df=df_sim_period,
            ml_results=pipeline.history_pred,
            demand_col="AMOUNT",
            price_col="UNITPRICE",
            cost_col="cost",
            date_col="DATE_",
            initial_stock=0.0,
            markup=0.30,
        )
        print(table.to_string())
        table.to_csv(pipeline.BASE_DIR / f"baseline_comparison_{itemcode}.csv")
        return table
    except Exception as e:
        print(f"Baseline comparison failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Синтетическая строка для текущего дня
# ---------------------------------------------------------------------------

def _make_future_row(day: pd.Timestamp, itemcode, df_hist: pd.DataFrame) -> pd.DataFrame:
    last_row = df_hist[df_hist["DATE_"] == df_hist["DATE_"].max()].iloc[0].to_dict()
    last_row["DATE_"] = pd.to_datetime(day)
    last_row["ITEMCODE"] = itemcode
    for col in ["GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D"]:
        last_row[col] = np.nan
    for col in ["AMOUNT_1D", "TOTALPRICE_1D"]:
        last_row[col] = 0
    return pd.DataFrame([last_row])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:

    def __init__(self, itemcode: int = 17662):
        self.data = None
        self.itemcode = int(itemcode)
        self.WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
        self.OUTPUT_DIR = self.WORKSPACE_ROOT / "output" / "elasticity"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_DIR = self.OUTPUT_DIR
        self._path_cost = self.WORKSPACE_ROOT / "data" / "cost.csv"
        self.history_pred = pd.DataFrame(
            columns=["DATE_", "ITEMCODE", "unitprice", "gmv",
                     "margin", "quantity", "margin_percent"]
        )
        self.warehouse = None
        self._read_data(itemcode=self.itemcode)
        self._load_existing_results()

    # ------------------------------------------------------------------

    def _read_data(self, itemcode: int | None = None) -> None:
        if not self._path_cost.exists():
            raise FileNotFoundError(f"File not found: {self._path_cost.resolve()}")

        if itemcode is not None:
            print(
                f"Loading source data for ITEMCODE={int(itemcode)} "
                f"(first run may take a few minutes; then it is cached)..."
            )
        else:
            print("Loading source data (first run may take a few minutes)...")

        data = load_elasticity_source_data(
            self.WORKSPACE_ROOT,
            usecols=["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"],
            itemcodes=[int(itemcode)] if itemcode is not None else None,
        )
        cost = pd.read_csv(self._path_cost)
        self.data = pd.merge(data, cost[["ITEMCODE", "cost"]], how="left", on="ITEMCODE")
        print(f"Data loaded: {self.data.shape}, columns: {list(self.data.columns)}")

    def _load_existing_results(self) -> None:
        files = list(self.BASE_DIR.glob("opt_results_*.csv"))
        if not files:
            return
        latest = max(files, key=lambda p: p.stat().st_mtime)
        try:
            self.history_pred = pd.read_csv(latest)
            self.history_pred["DATE_"] = pd.to_datetime(self.history_pred["DATE_"])
            print(f"Loaded existing results: {latest.name} ({len(self.history_pred)} rows)")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    # ------------------------------------------------------------------
    # Колонки, которые не нужны при обучении и предсказании
    # ------------------------------------------------------------------

    _TARGET_AND_FUTURE_COLS = [
        "DATE_", "CATEGORY1", "CATEGORY2",
        "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
        "AMOUNT_0D_target", "AMOUNT_1D_target", "AMOUNT_7D_target",
        "AMOUNT_15D_target", "AMOUNT_30D_target",
        "AMOUNT_1D", "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D",
    ]

    # ------------------------------------------------------------------

    def simulation(
        self,
        first_day: pd.Timestamp | None = None,
        n_days: int = 10,
        window_days: int = 30,
        run_evaluation: bool = True,
    ) -> dict[str, object]:
        itemcode = self.itemcode

        # Препроцессинг
        self.data = self.data[self.data["ITEMCODE"] == itemcode]
        self.data = preprocessor(self.data)
        print(f"Preprocessed: {self.data.shape}")

        if first_day is None:
            first_day = self.data["DATE_"].max() - pd.Timedelta("180D")
        else:
            first_day = pd.Timestamp(first_day).normalize()
        n_days = max(int(n_days), 1)
        last_day = first_day + pd.Timedelta(days=n_days)
        print(f"Simulation period: {first_day.date()} -> {last_day.date()}")

        # Оценка модели на исторических данных до симуляции
        df_hist_eval = self.data[self.data["DATE_"] < first_day].copy()
        if run_evaluation and len(df_hist_eval) >= 30:
            # Добавляем строку-заглушку, чтобы rolling-окно не обрезало хвост
            dummy = df_hist_eval.iloc[[-1]].copy()
            dummy["DATE_"] = df_hist_eval["DATE_"].max() + pd.Timedelta("8D")
            df_hist_eval = pd.concat([df_hist_eval, dummy], ignore_index=True)
            try:
                df_etl_eval = etl_with_demand_target(df_hist_eval)
                _evaluate_model(df_etl_eval, self.BASE_DIR)
            except Exception as e:
                print(f"Pre-simulation evaluation failed: {e}")

        # Инициализация склада
        warehouse_state_path = self.BASE_DIR / f"warehouse_state_{itemcode}.pkl"
        warehouse_store_csv  = self.BASE_DIR / f"warehouse_store_{itemcode}.csv"

        self.warehouse = init_warehouse(
            self.data.copy(),
            itemcode=itemcode,
            base_dir=self.BASE_DIR,
            state_path=warehouse_state_path,
            store_csv_path=warehouse_store_csv,
        )

        self.history_pred = pd.DataFrame(
            columns=["DATE_", "ITEMCODE", "unitprice", "gmv",
                     "margin", "quantity", "margin_percent"]
        )

        policy_params = {
            "LAMBDA_KVI": 10.0,
            "DELTA_KVI": 0.2,
            "MIN_MARGIN_PCT": 0.05,
            "MAX_PRICE_MULT": 2.5,   # оптимизатор может поднять цену до 2.5× от базовой
            "MIN_PRICE_MULT": 0.5,
            # ----------------------------------------------------------------
            # DIGIT_PRICE_DELTAS — дискретная сетка заменена на непрерывную
            # оптимизацию через scipy.minimize_scalar (см. analyze_item_policy).
            # Оставляем для совместимости: если digit_optimize_family_prices
            # всё ещё использует сетку — расширили до [-0.5, +1.0] с мелким шагом.
            # Корень бага: при |ε| < 1 маржа монотонно растёт → всегда выбирался
            # правый край (+20%). Теперь правый край = +100% (MAX_PRICE_MULT).
            # ----------------------------------------------------------------
            "DIGIT_PRICE_DELTAS": [
                -0.30, -0.20, -0.15, -0.10, -0.07, -0.05, -0.03,
                0.0,
                0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
            ],
        }
        hp_path = None
        for candidate in [self.WORKSPACE_ROOT / "hyperparameters.json", self.WORKSPACE_ROOT / "KVI" / "hyperparameters.json"]:
            if candidate.exists():
                hp_path = candidate
                break
        if hp_path is not None:
            with open(hp_path, "r", encoding="utf-8") as f:
                hp = json.load(f)
                policy_params.update(hp.get("agent6", hp))

        day = first_day

        # ------------------------------------------------------------------
        # Баг 5 фикс: строим ETL-таблицу один раз для всех исторических данных,
        # чтобы в цикле можно было брать правильную строку признаков
        # для конкретного дня (dayofweek, month, rolling-лаги и т.д.).
        # Без этого template всегда = tail(1) → все дни одинаковы.
        # ------------------------------------------------------------------
        df_hist_all = self.data[self.data["DATE_"] < first_day].copy()
        try:
            df_etl_all = etl_with_demand_target(df_hist_all)
        except Exception as _etl_e:
            print(f"ETL pre-build failed: {_etl_e}")
            df_etl_all = pd.DataFrame()

        _TARGET_COLS = [
            "DATE_", "CATEGORY1", "CATEGORY2",
            "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
            "AMOUNT_0D_target", "AMOUNT_1D_target", "AMOUNT_7D_target",
            "AMOUNT_15D_target", "AMOUNT_30D_target",
            "AMOUNT_1D", "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D",
        ]

        def _get_day_template(day: pd.Timestamp) -> pd.DataFrame | None:
            """
            Возвращает строку признаков из df_etl_all, соответствующую
            конкретному дню симуляции. Несёт правильные dayofweek, month,
            rolling-лаги — устраняет баг плоской оптимизации.
            """
            if df_etl_all.empty:
                return None
            day_norm = pd.Timestamp(day).normalize()
            # Ищем строку с признаками ≤ day (ближайшая предшествующая)
            candidates = df_etl_all[df_etl_all["DATE_"] <= day_norm]
            if candidates.empty:
                return None
            row = candidates.sort_values("DATE_").tail(1).copy()
            # Синтетически обновляем дату и временны́е признаки для нужного дня
            row["DATE_"] = day_norm
            row["dayofweek"] = day_norm.dayofweek
            row["month"] = day_norm.month
            row["day"] = day_norm.day
            row["weekofyear"] = int(day_norm.isocalendar()[1])
            row["is_weekend"] = int(day_norm.dayofweek >= 5)
            return row.drop(columns=[c for c in _TARGET_COLS if c in row.columns and c != "DATE_"],
                            errors="ignore")

        # ------------------------------------------------------------------
        # Основной цикл
        # ------------------------------------------------------------------

        while day < last_day:
            print(f"\n--- {day.date()} ---")

            # Обновление склада
            wh = update_warehouse_day(
                self.warehouse,
                day,
                actual_sales=None,
                state_path=warehouse_state_path,
                store_csv_path=warehouse_store_csv,
            )
            print(f"Warehouse: rest={wh['rest']}, available={wh['available']}, pred={wh['pred']:.2f}")

            same_day_prices = self.data[
                (self.data["DATE_"].dt.normalize() == pd.Timestamp(day).normalize()) &
                (self.data["AMOUNT"] > 0)
            ]["UNITPRICE"]
            if len(same_day_prices) > 0:
                item_baseprice = float(same_day_prices.median())
            else:
                past_prices = self.data[
                    (self.data["DATE_"] < day) &
                    (self.data["AMOUNT"] > 0)
                ]["UNITPRICE"]
                item_baseprice = float(past_prices.tail(30).median()) if len(past_prices) > 0 else float(self.data["UNITPRICE"].median())

            item_cost = float(self.data["cost"].iloc[0]) if "cost" in self.data.columns else 0.0
            if not np.isfinite(item_baseprice) or item_baseprice <= 0:
                item_baseprice = max(item_cost * 1.3, 1.0)

            rec_df = analyze_item_policy(
                item_code=itemcode,
                date=str(pd.Timestamp(day).date()),
                price=float(item_baseprice),
                window_days=int(window_days),
                params=policy_params,
                output_path=None,
            )
            if rec_df.empty:
                raise RuntimeError(f"Policy block returned empty recommendation for ITEMCODE={itemcode} on {day.date()}")

            target_rows = rec_df[rec_df["ITEMCODE"].astype("int64") == int(itemcode)]
            rec_row = target_rows.iloc[0] if len(target_rows) > 0 else rec_df.iloc[0]

            optimal_price = round(float(rec_row["recommended_price"]), 2)
            expected_qty = max(float(rec_row.get("demand_new", 0.0)), 0.0)
            realized_qty = int(round(min(expected_qty, float(wh["available"]))))
            realized_qty = max(realized_qty, 0)

            realized_gmv = round(optimal_price * realized_qty, 2)
            realized_margin = round((optimal_price - item_cost) * realized_qty, 2)
            realized_margin_pct = round((optimal_price - item_cost) / max(optimal_price, 1e-8) * 100, 2)
            elasticity_val = float(rec_row.get("elasticity", np.nan))
            elasticity_method = str(rec_row.get("elasticity_method", "unknown"))
            n_obs_window = int(rec_row.get("n_obs_window", 0))
            stock_binding = expected_qty > float(wh["available"])

            print(
                f"Optimal price: {optimal_price:.2f} | "
                f"Q_expected: {expected_qty:.2f} | "
                f"Q_realized: {realized_qty} | "
                f"GMV: {realized_gmv:.2f} | "
                f"Margin: {realized_margin:.2f} | "
                f"Elasticity: {elasticity_val:.3f} ({elasticity_method}) | "
                f"n_obs={n_obs_window} | "
                f"Stock binding: {stock_binding}"
            )

            # Обновление склада по факту
            last_idx = self.warehouse.store.index[-1]
            self.warehouse.store.loc[last_idx, "demand"]    = realized_qty
            self.warehouse.store.loc[last_idx, "sale_real"] = min(realized_qty, wh["available"])
            self.warehouse.store.loc[last_idx, "rest"]      = max(wh["available"] - realized_qty, 0)
            self.warehouse.rest = self.warehouse.store.loc[last_idx, "rest"]

            self.history_pred = pd.concat(
                [
                    self.history_pred,
                    pd.DataFrame([{
                        "DATE_":          day,
                        "ITEMCODE":       itemcode,
                        "unitprice":      optimal_price,
                        "gmv":            realized_gmv,
                        "margin":         realized_margin,
                        "quantity":       realized_qty,
                        "margin_percent": realized_margin_pct,
                        "cost":           item_cost,
                        "baseprice":      item_baseprice,
                        "elasticity":     elasticity_val,
                        "elasticity_method": elasticity_method,
                        "n_obs_window":   n_obs_window,
                        "stock_binding":  stock_binding,
                    }]),
                ],
                ignore_index=True,
            )

            day += pd.Timedelta("1D")

        # ------------------------------------------------------------------
        # Сохранение результатов
        # ------------------------------------------------------------------

        self.history_pred = (
            self.history_pred
            .drop_duplicates(subset=["DATE_", "ITEMCODE"], keep="last")
            .sort_values("DATE_")
            .reset_index(drop=True)
        )

        out_csv = self.BASE_DIR / f"opt_results_{itemcode}.csv"
        self.history_pred.to_csv(out_csv, index=False)
        print(f"\nResults saved: {out_csv}")

        # ------------------------------------------------------------------
        # Baseline-сравнение
        # ------------------------------------------------------------------

        df_sim_period = self.data[
            (self.data["DATE_"] >= first_day) &
            (self.data["DATE_"] < last_day)
        ].copy()

        baseline_table = _run_baselines(self, df_sim_period, itemcode)

        print("\nSimulation complete.")
        return {
            "history_pred": self.history_pred.copy(),
            "baseline": baseline_table.copy(),
            "first_day": first_day,
            "last_day": last_day,
        }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipe = Pipeline()
    pipe.simulation()
