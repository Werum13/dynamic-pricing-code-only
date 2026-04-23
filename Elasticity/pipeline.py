import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from DataPreprocessor import preprocessor
from ETL import etl_with_demand_target
from DemandModel import demand_model
from PriceOptimizer import price_optimizer
from byer import init_warehouse, update_warehouse_day
from Evaluation import ElasticityEvaluator
from Baseline import compare_baselines
from data_sources import load_elasticity_source_data

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
) -> None:
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
    except Exception as e:
        print(f"Baseline comparison failed: {e}")


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

    def __init__(self):
        self.data = None
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
        self._read_data()
        self._load_existing_results()

    # ------------------------------------------------------------------

    def _read_data(self) -> None:
        if not self._path_cost.exists():
            raise FileNotFoundError(f"File not found: {self._path_cost.resolve()}")

        data = load_elasticity_source_data(
            self.WORKSPACE_ROOT,
            usecols=["ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE", "AMOUNT", "CATEGORY1", "CATEGORY2"],
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

    def simulation(self) -> None:
        itemcode = 107

        # Препроцессинг
        self.data = self.data[self.data["ITEMCODE"] == itemcode]
        self.data = preprocessor(self.data)
        print(f"Preprocessed: {self.data.shape}")

        first_day = self.data["DATE_"].max() - pd.Timedelta("180D")
        last_day  = first_day + pd.Timedelta("10D")
        print(f"Simulation period: {first_day.date()} -> {last_day.date()}")

        # Оценка модели на исторических данных до симуляции
        df_hist_eval = self.data[self.data["DATE_"] < first_day].copy()
        if len(df_hist_eval) >= 30:
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

        model_cache  = _load_model_cache(self.BASE_DIR)
        model_demand = None

        self.history_pred = pd.DataFrame(
            columns=["DATE_", "ITEMCODE", "unitprice", "gmv",
                     "margin", "quantity", "margin_percent"]
        )

        day = first_day

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

            # Исторические данные + строка для сегодня
            df = self.data[self.data["DATE_"] < day].copy()
            future_row = _make_future_row(day, itemcode, df)
            df = pd.concat([df, future_row], ignore_index=True).sort_values("DATE_").reset_index(drop=True)

            df = etl_with_demand_target(df)
            print(f"After ETL: {df.shape}")

            # Обучающая выборка: без последнего дня (next-day target смотрит вперёд)
            cutoff = day - pd.Timedelta(days=1)
            train_full = df[(df["DATE_"] < day) & (df["DATE_"] < cutoff)].copy()

            target   = train_full["AMOUNT_0D_target"].copy()
            train_df = train_full.drop(columns=self._TARGET_AND_FUTURE_COLS, errors="ignore")

            valid = train_df.notna().all(axis=1) & target.notna()
            train_df = train_df[valid]
            target   = target[valid]

            print(f"Training rows: {len(train_df)} (data up to {cutoff.date()})")

            model_demand, model_cache = _get_model(
                base_dir=self.BASE_DIR,
                itemcode=itemcode,
                day=day,
                train_df=train_df,
                target=target,
                cache=model_cache,
            )

            # Строка-шаблон для сегодняшнего дня (одна строка признаков)
            today_template = (
                df[df["DATE_"] == day]
                .drop(columns=self._TARGET_AND_FUTURE_COLS, errors="ignore")
                .copy()
            )
            today_template["Id"] = 0

            item_cost      = float(self.data["cost"].iloc[0]) if "cost" in self.data.columns else 0.0
            hist_prices = self.data[
                (self.data["DATE_"].dt.normalize() == pd.Timestamp(day).normalize()) &
                (self.data["AMOUNT"] > 0)
            ]["UNITPRICE"]
            item_baseprice = (
                float(hist_prices.median()) if len(hist_prices) > 0
                else float(today_template["UNITPRICE"].iloc[0]) if "UNITPRICE" in today_template.columns
                else None
            )

            # Оптимизация цены
            d = price_optimizer(
                demand_model=model_demand,
                row_features=today_template,
                stock=float(wh["available"]),
                cost=item_cost,
                base_price=item_baseprice,
                objective="profit",
                n_grid=300,
            )

            optimal_price = round(float(d["unitprice"]), 2)
            realized_qty  = int(round(float(d["quantity"])))
            realized_gmv  = round(optimal_price * realized_qty, 2)
            realized_margin = round((optimal_price - item_cost) * realized_qty, 2)
            realized_margin_pct = round((optimal_price - item_cost) / max(optimal_price, 1e-8) * 100, 2)

            print(
                f"Optimal price: {optimal_price:.2f} | "
                f"Q: {realized_qty} | "
                f"GMV: {realized_gmv:.2f} | "
                f"Margin: {realized_margin:.2f} | "
                f"Elasticity: {d['elasticity']:.3f} | "
                f"Stock binding: {d['stock_binding']}"
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
                        "elasticity":     d["elasticity"],
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

        try:
            df_for_baseline = etl_with_demand_target(
                self.data[self.data["DATE_"] < last_day].copy()
            )
            _run_baselines(self, df_sim_period, itemcode, model_demand, df_for_baseline)
        except Exception as e:
            print(f"Baseline comparison failed: {e}")
            _run_baselines(self, df_sim_period, itemcode)

        print("\nSimulation complete.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipe = Pipeline()
    pipe.simulation()
