"""
generate_costs.py
=================
Генерирует динамическую себестоимость для всех товаров.

Алгоритм:
  - Начальная себестоимость = median_price * uniform(0.40, 0.60).
  - Каждую неделю себестоимость ТОЛЬКО растёт: cost += step,
    где step ~ Exponential(mean = step_mean) для данного товара.
    step_mean пропорционален ценовому диапазону товара.
  - Сохраняется одна строка на товар на неделю: DATE_ (понедельник),
    ITEMCODE, cost.

Выход: data/cost.csv  (заменяет старый статичный файл)
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
OUT_PATH  = Path("data/cost.csv")

# Диапазон генерации
DATE_START = "2021-01-01"
DATE_END   = "2024-12-31"


# ─────────────────────────────────────────────────────────────────────────────
# Функция генерации себестоимости для одного товара (недельный ряд)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_weekly_costs(
    item_code: int,
    median_price: float,
    price_range: float,
    weeks: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Монотонно возрастающая «лестница» себестоимости.

    - Стартуем с 40–60% медианной цены.
    - Каждую неделю добавляем случайный шаг ~ Exponential(step_mean),
      где step_mean = price_range * 0.002  (т.е. ~0.2% от диапазона цен в среднем).
    - Себестоимость никогда не убывает.
    """
    n = len(weeks)

    # Стартовое значение
    start_fraction = float(rng.uniform(0.40, 0.60))
    cost = median_price * start_fraction
    cost = max(cost, 0.01)

    # Средний размер шага: небольшой относительно диапазона
    step_mean = max(price_range * 0.002, 0.001)

    costs = np.empty(n)
    for i in range(n):
        costs[i] = cost
        step = float(rng.exponential(step_mean))
        cost += step

    return costs


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных (Order_Details + Orders)
# ─────────────────────────────────────────────────────────────────────────────

print("[1/4] Загрузка данных...")
od = pd.read_csv("data/Order_Details.csv",
                 usecols=["ITEMCODE", "UNITPRICE", "AMOUNT"])

# Рассчитываем цену за единицу (UNITPRICE уже есть, но проверяем корректность)
od = od[(od["AMOUNT"] > 0) & od["UNITPRICE"].gt(0) & od["UNITPRICE"].notna()].copy()

# Убираем выбросы IQR per item
q1 = od.groupby("ITEMCODE")["UNITPRICE"].transform(lambda x: x.quantile(0.25))
q3 = od.groupby("ITEMCODE")["UNITPRICE"].transform(lambda x: x.quantile(0.75))
iqr = q3 - q1
od = od[od["UNITPRICE"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)].copy()

item_stats = (
    od.groupby("ITEMCODE")["UNITPRICE"]
    .agg(
        min_price="min",
        max_price="max",
        median_price="median",
    )
    .reset_index()
)
item_stats["price_range"] = item_stats["max_price"] - item_stats["min_price"]
item_stats.loc[item_stats["price_range"] <= 0, "price_range"] = item_stats["median_price"] * 0.1

print(f"       Товаров: {len(item_stats):,}")
del od; gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Построение недельной сетки
# ─────────────────────────────────────────────────────────────────────────────

# Все понедельники в диапазоне
weeks = pd.date_range(start=DATE_START, end=DATE_END, freq="W-MON")
n_weeks = len(weeks)
print(f"[2/4] Недель в сетке: {n_weeks}  ({DATE_START} … {DATE_END})")


# ─────────────────────────────────────────────────────────────────────────────
# Генерация себестоимости
# ─────────────────────────────────────────────────────────────────────────────

print("[3/4] Генерация себестоимости...")

CHUNK = 2000          # пишем батчами по 2000 товаров
out_chunks: list[pd.DataFrame] = []
n_items = len(item_stats)

for start in tqdm(range(0, n_items, CHUNK), desc="Генерация", unit="chunk"):
    batch = item_stats.iloc[start : start + CHUNK]
    rows = []
    for _, row in batch.iterrows():
        item_rng = np.random.default_rng(int(row["ITEMCODE"]) ^ SEED)
        costs = _generate_weekly_costs(
            item_code    = int(row["ITEMCODE"]),
            median_price = float(row["median_price"]),
            price_range  = float(row["price_range"]),
            weeks        = weeks,
            rng          = item_rng,
        )
        rows.append({
            "ITEMCODE": int(row["ITEMCODE"]),
            "costs":    costs,
        })

    chunk_df = pd.DataFrame({
        "DATE_":    weeks,
        **{str(r["ITEMCODE"]): r["costs"] for r in rows}
    })
    # Переводим в «длинный» формат
    chunk_long = chunk_df.melt(id_vars="DATE_", var_name="ITEMCODE", value_name="cost")
    chunk_long["ITEMCODE"] = chunk_long["ITEMCODE"].astype(int)
    out_chunks.append(chunk_long)

    del chunk_df, chunk_long, rows, batch
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Сохранение
# ─────────────────────────────────────────────────────────────────────────────

print("[4/4] Сохранение...")
result = pd.concat(out_chunks, ignore_index=True)
result["DATE_"] = result["DATE_"].dt.strftime("%Y-%m-%d")
result["cost"]  = result["cost"].round(4)
result = result.sort_values(["ITEMCODE", "DATE_"]).reset_index(drop=True)

result.to_csv(OUT_PATH, index=False)
print(f"      Сохранено: {OUT_PATH}  ({len(result):,} строк)")
print(f"      Пример:")
print(result.head(10).to_string(index=False))
