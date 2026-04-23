"""
compute_elasticity.py
=====================
Считает ценовую эластичность для каждого ITEMCODE в одной точке цены.

Этапы:
    1. Загружаем full_data.csv (чанками, только нужные колонки) + cost.csv
    2. Для каждого ITEMCODE:
             a. DataPreprocessor — приведение типов, UNITPRICE, удаление выбросов
             b. ETL (etl_with_demand_target) — бинирование цен, rolling-фичи, time-фичи
             c. DemandModel — обучение Q(P) на исторических данных
             d. Численная производная → эластичность ε(P) в одной базовой точке
    3. Сохраняем CSV: ItemCode, Elasticity
"""

import sys
import json
import pickle
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "elasticity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(MODULE_DIR))

from DataPreprocessor import preprocessor         # noqa: E402
from ETL import etl_with_demand_target            # noqa: E402
from DemandModel import demand_model              # noqa: E402

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

NEEDED_COLS = [
    "ITEMCODE", "DATE_", "UNITPRICE", "TOTALPRICE",
    "AMOUNT", "CATEGORY1", "CATEGORY2",
]

TARGET_AND_FUTURE_COLS = [
    "DATE_", "CATEGORY1", "CATEGORY2",
    "GMV_1D", "GMV_7D", "GMV_15D", "GMV_30D",
    "AMOUNT_0D_target",
    "AMOUNT_1D_target",
    "AMOUNT_7D_target",
    "AMOUNT_15D_target",
    "AMOUNT_30D_target",
    "AMOUNT_1D",
    "AMOUNT_7D", "AMOUNT_15D", "AMOUNT_30D",
]

CHUNK_SIZE     = 200_000
MIN_TOTAL_ROWS = 20
MIN_TRAIN_ROWS = 10
MODEL_DIR      = OUTPUT_DIR / "models"
MODEL_CACHE    = OUTPUT_DIR / "elasticity_model_cache.json"
CACHE_DIR      = OUTPUT_DIR / ".elasticity_cache"
RAW_DIR        = CACHE_DIR / "raw_items"
RAW_MANIFEST   = CACHE_DIR / "raw_manifest.json"


def _model_path(itemcode, date: pd.Timestamp) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"model_{itemcode}_{date.strftime('%Y-%m-%d')}.pkl"


def _load_model_cache() -> dict:
    if MODEL_CACHE.exists():
        with open(MODEL_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_model_cache(cache: dict) -> None:
    with open(MODEL_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _load_cached_model(itemcode, cache: dict):
    key = str(itemcode)
    if key not in cache:
        return None, None
    entry = cache[key]
    path = _model_path(itemcode, pd.to_datetime(entry["last_trained"]))
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        return pickle.load(f), entry["last_trained"]


def _save_model(itemcode, date: pd.Timestamp, model, cache: dict) -> None:
    path = _model_path(itemcode, date)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    cache[str(itemcode)] = {
        "last_trained": date.strftime("%Y-%m-%d"),
        "model_path": str(path),
    }
    _save_model_cache(cache)


def _load_raw_manifest() -> dict:
    if RAW_MANIFEST.exists():
        with open(RAW_MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_raw_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(RAW_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 1. Подготовка кэша сырых данных
# ---------------------------------------------------------------------------

def prepare_item_files():
    path_data = WORKSPACE_ROOT / "data" / "full_data.csv"
    path_cost = WORKSPACE_ROOT / "data" / "cost.csv"

    manifest = _load_raw_manifest()
    source_mtime = path_data.stat().st_mtime

    if manifest.get("source_mtime") == source_mtime and RAW_DIR.exists():
        item_files = {
            int(itemcode): Path(path)
            for itemcode, path in manifest.get("item_files", {}).items()
            if Path(path).exists()
        }
        if item_files:
            print(f"Используем кэш сырых данных: {len(item_files)} ITEMCODE")
            return item_files, pd.read_csv(path_cost)[["ITEMCODE", "cost"]]

    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Читаем full_data.csv чанками и сохраняем по ITEMCODE...")
    item_files = {}
    total = 0

    for chunk in pd.read_csv(path_data, usecols=NEEDED_COLS, chunksize=CHUNK_SIZE):
        total += len(chunk)
        itemcodes = np.unique(chunk["ITEMCODE"].to_numpy())
        code_values = chunk["ITEMCODE"].to_numpy()
        for itemcode in itemcodes:
            grp = chunk.loc[code_values == itemcode]
            item_path = RAW_DIR / f"item_{itemcode}.csv"
            write_header = not item_path.exists()
            grp.to_csv(item_path, mode="a", index=False, header=write_header)
            item_files[int(itemcode)] = str(item_path)
        if total % 2_000_000 == 0:
            print(f"  ...прочитано {total:,} строк")

    _save_raw_manifest(
        {
            "source_mtime": source_mtime,
            "item_files": {str(k): v for k, v in item_files.items()},
        }
    )

    print(f"Итого: {total:,} строк, {len(item_files)} уникальных ITEMCODE")
    return item_files, pd.read_csv(path_cost)[["ITEMCODE", "cost"]]


# ---------------------------------------------------------------------------
# 2. Обработка одного ITEMCODE
# ---------------------------------------------------------------------------

def process_item(itemcode, item_path, cost_map, cache):
    # ── Preprocessor ─────────────────────────────────────────────────────────
    try:
        df_item = pd.read_csv(item_path)
        df_item["cost"] = df_item["ITEMCODE"].map(cost_map)
        df_pre = preprocessor(df_item.copy())
    except Exception as e:
        print(f"  [{itemcode}] preprocessor error: {e}")
        return []

    if len(df_pre) < MIN_TOTAL_ROWS:
        print(f"  [{itemcode}] пропуск: {len(df_pre)} строк после препроцессинга")
        return []

    # ── ETL ──────────────────────────────────────────────────────────────────
    try:
        df_etl = etl_with_demand_target(df_pre)
    except Exception as e:
        print(f"  [{itemcode}] ETL error: {e}")
        return []

    if len(df_etl) < MIN_TRAIN_ROWS:
        print(f"  [{itemcode}] пропуск: {len(df_etl)} строк после ETL")
        return []

    # ── Обучение модели ───────────────────────────────────────────────────────
    max_date = df_etl["DATE_"].max()
    cutoff   = max_date - pd.Timedelta(days=7)

    train_full    = df_etl[df_etl["DATE_"] < cutoff].copy()
    target_series = train_full["AMOUNT_0D_target"].copy()
    train_df      = train_full.drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore")
    if "UNITPRICE" not in train_df.columns:
        print(f"  [{itemcode}] пропуск: в train_df нет признака UNITPRICE")
        return []
    train_df = train_df[["UNITPRICE"]].copy()

    valid         = train_df.notna().all(axis=1) & target_series.notna()
    train_df      = train_df[valid]
    target_series = target_series[valid]

    if len(train_df) < MIN_TRAIN_ROWS:
        print(f"  [{itemcode}] пропуск: только {len(train_df)} строк для обучения")
        return []

    model, last_trained_str = _load_cached_model(itemcode, cache)
    current_date = max_date
    is_monday = current_date.weekday() == 0
    need_retrain = (
        model is None
        or last_trained_str is None
        or (current_date - pd.to_datetime(last_trained_str)).days >= 7
        or is_monday
    )

    if need_retrain:
        try:
            model = demand_model(train=train_df, target=target_series)
            _save_model(itemcode, current_date, model, cache)
            print(f"  [{itemcode}] модель переобучена и сохранена в кэш")
        except Exception as e:
            print(f"  [{itemcode}] ошибка обучения: {e}")
            return []
    else:
        print(f"  [{itemcode}] загружена модель из кэша ({last_trained_str})")

    # ── Эластичность только в одной базовой точке ────────────────────────────
    if "DATE_" not in df_etl.columns:
        print(f"  [{itemcode}] пропуск: в df_etl нет DATE_")
        return []

    prediction_row = df_etl.sort_values("DATE_").tail(1).copy()
    if prediction_row.empty:
        print(f"  [{itemcode}] пропуск: нет строки для расчёта эластичности")
        return []

    baseprice = np.nan
    price_row = prediction_row.iloc[0]
    if {"TOTALPRICE", "AMOUNT"}.issubset(prediction_row.columns):
        totalprice = pd.to_numeric(price_row.get("TOTALPRICE"), errors="coerce")
        amount = pd.to_numeric(price_row.get("AMOUNT"), errors="coerce")
        if pd.notna(totalprice) and pd.notna(amount) and float(amount) > 0:
            baseprice = float(totalprice) / float(amount)

    if not np.isfinite(baseprice) or baseprice <= 0:
        baseprice = pd.to_numeric(price_row.get("UNITPRICE"), errors="coerce")

    if not np.isfinite(baseprice) or baseprice <= 0:
        print(f"  [{itemcode}] пропуск: не удалось определить BASEPRICE")
        return []

    template = prediction_row.drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore").copy()
    template["UNITPRICE"] = baseprice
    template["BASEPRICE"] = baseprice
    template["Id"] = 0

    missing_cols = [c for c in model.columns_ if c not in template.columns]
    if missing_cols:
        print(f"  [{itemcode}] пропуск: не хватает признаков для модели: {missing_cols}")
        return []

    valid_template = template.dropna(subset=model.columns_).copy()
    if valid_template.empty:
        print(f"  [{itemcode}] пропуск: нет строк без NaN для расчёта эластичности")
        return []

    try:
        eps = model.elasticity(valid_template)
    except Exception as e:
        print(f"  [{itemcode}] ошибка расчёта эластичности: {e}")
        return []

    elasticity_value = round(float(eps[0]), 4)
    print(f"  [{itemcode}] готово: Elasticity={elasticity_value}")
    return [{
        "ItemCode": int(itemcode),
        "Elasticity": elasticity_value,
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    item_files_df, cost_df = prepare_item_files()
    cost_map = dict(cost_df.values)
    cache       = _load_model_cache()
    item_codes  = sorted(item_files_df.keys())
    total_items = len(item_codes)

    print(f"\nОбрабатываем {total_items} item code(-ов)...\n")

    all_results = []
    for i, itemcode in enumerate(item_codes, 1):
        print(f"[{i}/{total_items}] ITEMCODE={itemcode}")
        rows = process_item(itemcode, item_files_df[itemcode], cost_map, cache)
        all_results.extend(rows)

    if not all_results:
        print("\nНет результатов для записи.")
        return

    result_df = (
        pd.DataFrame(all_results)
        .sort_values(["ItemCode"])
        .reset_index(drop=True)
    )

    out_path = OUTPUT_DIR / "elasticity_by_itemcode.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\nСохранено: {out_path}")
    print(f"  Строк:        {len(result_df)}")
    print(f"  Item codes:   {result_df['ItemCode'].nunique()}")
    print("\nПервые строки:")
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
