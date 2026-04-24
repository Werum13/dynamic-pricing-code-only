"""
item_price_analyzer.py — Автономный модуль анализа цены по товару
=================================================================
Запускается независимо от оркестратора — по запросу для конкретного товара.

Цепочка:
  1. Читает KVI-файлы (kvi_scores_full, substitute_map, behavioral_features)
  2. Определяет «семью» товара: сам item + его субституты/комплементы/каннибалы
  3. Для каждого члена семьи считает точечную эластичность
     на заданную дату и цену (log-log регрессия по окну ±window_days)
  4. Запускает совместную оптимизацию цены для всей семьи
  5. Возвращает DataFrame с рекомендациями + сохраняет JSON-отчёт

Python API:
    from item_price_analyzer import analyze_item
    result = analyze_item(item_code=1234, date="2024-03-01", price=99.9)

CLI:
    python item_price_analyzer.py --item 1234 --date 2024-03-01 --price 99.9
    python item_price_analyzer.py --item 1234 --date 2024-03-01           # price = текущая из данных
    python item_price_analyzer.py --item 1234 --date 2024-03-01 --window 30 --output my_report.json
"""

import argparse
import json
import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from digit_optimization_module import digit_optimize_family_prices

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Пути (относительно этого файла) ──────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Входные KVI-файлы
KVI_SCORES_PATH    = OUTPUT_DIR / "kvi_scores_full.csv"
KVI_CANDIDATES_PATH= OUTPUT_DIR / "kvi_candidates.csv"
SUBSTITUTE_MAP_PATH= OUTPUT_DIR / "substitute_map.json"
BEHAVIORAL_PATH    = OUTPUT_DIR / "behavioral_features.csv"
ELASTICITY_BASE_PATH = OUTPUT_DIR / "elasticity_by_itemid.csv"
COST_PATH          = DATA_DIR   / "cost.csv"

# Источник сырых транзакций (DATE_, ITEMCODE, PRICE, AMOUNT)
TRANSACTIONS_PATH  = DATA_DIR / "transactions.csv"   # переименуй под свой файл

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("item_price_analyzer")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА KVI-КОНТЕКСТА
# ══════════════════════════════════════════════════════════════════════════════

def load_kvi_context() -> dict:
    """
    Загружает все KVI-файлы и возвращает единый контекст.

    Returns
    -------
    dict с ключами:
        kvi_scores    : pd.DataFrame (ITEMCODE + все метрики)
        kvi_set       : set[int]     (ITEMCODE из kvi_candidates)
        sub_map       : dict         (substitute_map.json as-is)
        behavioral    : pd.DataFrame
        cost_map      : dict {ITEMCODE: cost}
        elast_base    : dict {ITEMCODE: (BASEPRICE, elasticity)}
    """
    ctx = {}

    # kvi_scores_full
    if KVI_SCORES_PATH.exists():
        df = pd.read_csv(KVI_SCORES_PATH)
        df = _normalize_itemcode(df)
        ctx["kvi_scores"] = df
        log.info(f"kvi_scores_full: {len(df)} товаров")
    else:
        ctx["kvi_scores"] = pd.DataFrame()
        log.warning("kvi_scores_full.csv не найден")

    # kvi_candidates → set
    if KVI_CANDIDATES_PATH.exists():
        df_kvi = pd.read_csv(KVI_CANDIDATES_PATH)
        df_kvi = _normalize_itemcode(df_kvi)
        ctx["kvi_set"] = set(df_kvi["ITEMCODE"].dropna().astype(int))
    else:
        ctx["kvi_set"] = set()
        log.warning("kvi_candidates.csv не найден — KVI-список пуст")

    # substitute_map
    if SUBSTITUTE_MAP_PATH.exists():
        with open(SUBSTITUTE_MAP_PATH, encoding="utf-8") as f:
            ctx["sub_map"] = json.load(f)
        log.info(f"substitute_map: {len(ctx['sub_map'])} товаров с связями")
    else:
        ctx["sub_map"] = {}
        log.warning("substitute_map.json не найден — связи между товарами недоступны")

    # behavioral_features
    if BEHAVIORAL_PATH.exists():
        df_b = pd.read_csv(BEHAVIORAL_PATH)
        df_b = _normalize_itemcode(df_b)
        ctx["behavioral"] = df_b
    else:
        ctx["behavioral"] = pd.DataFrame()

    # cost
    if COST_PATH.exists():
        df_c = pd.read_csv(COST_PATH)
        df_c = _normalize_itemcode(df_c)
        ctx["cost_map"] = dict(zip(df_c["ITEMCODE"].astype(int), df_c["cost"]))
    else:
        ctx["cost_map"] = {}
        log.warning("cost.csv не найден — маржинальный анализ недоступен")

    # elasticity_by_itemid (базовая точка)
    if ELASTICITY_BASE_PATH.exists():
        df_e = pd.read_csv(ELASTICITY_BASE_PATH)
        df_e = _normalize_itemcode(df_e)
        elast_base = {}
        for _, row in df_e.iterrows():
            ic = int(row["ITEMCODE"])
            bp = _get_col(row, ["BASEPRICE", "baseprice", "base_price"], default=None)
            el = _get_col(row, ["elasticity", "elast", "elasticity_value"], default=None)
            if bp is not None and el is not None:
                elast_base[ic] = (float(bp), float(el))
        ctx["elast_base"] = elast_base
    else:
        ctx["elast_base"] = {}

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# 2. ОПРЕДЕЛЕНИЕ «СЕМЬИ» ТОВАРА
# ══════════════════════════════════════════════════════════════════════════════

def get_item_family(item_code: int, sub_map: dict) -> dict:
    """
    Возвращает семью товара из substitute_map.

    Returns
    -------
    dict:
        target       : item_code сам по себе
        substitutes  : list[int]
        complements  : list[int]
        cannibals    : list[int]
        all_items    : list[int]  — все члены семьи включая target
    """
    key = str(item_code)
    entry = sub_map.get(key, sub_map.get(item_code, {}))

    subs   = [int(x) for x in entry.get("substitutes", [])]
    comps  = [int(x) for x in entry.get("complements", [])]
    canns  = [int(x) for x in entry.get("cannibals",   [])]

    all_items = list(dict.fromkeys([item_code] + subs + comps + canns))

    family = {
        "target":      item_code,
        "substitutes": subs,
        "complements": comps,
        "cannibals":   canns,
        "all_items":   all_items,
    }

    log.info(
        f"Семья товара {item_code}: "
        f"субституты={subs}, комплементы={comps}, каннибалы={canns}"
    )
    return family


# ══════════════════════════════════════════════════════════════════════════════
# 3. ТОЧЕЧНАЯ ЭЛАСТИЧНОСТЬ
# ══════════════════════════════════════════════════════════════════════════════

def load_transactions(item_codes: list[int]) -> pd.DataFrame:
    """
    Загружает транзакции для списка товаров.
    Ожидает колонки: DATE_, ITEMCODE, PRICE, AMOUNT (или их аналоги).
    """
    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Файл транзакций не найден: {TRANSACTIONS_PATH}\n"
            f"Переименуй свой файл или измени TRANSACTIONS_PATH в начале модуля."
        )

    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["DATE_"])
    df = _normalize_itemcode(df)

    # Нормализуем имена колонок
    df = _rename_col(df, ["DATE_", "date", "Date", "ORDER_DATE"], "DATE_")
    df = _rename_col(df, ["PRICE", "price", "UNIT_PRICE", "unit_price"], "PRICE")
    df = _rename_col(df, ["AMOUNT", "amount", "QTY", "qty", "quantity"], "AMOUNT")

    required = {"ITEMCODE", "DATE_", "PRICE", "AMOUNT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле транзакций не хватает колонок: {missing}")

    df["DATE_"] = pd.to_datetime(df["DATE_"])
    df = df[df["ITEMCODE"].isin(item_codes)].copy()
    log.info(f"Транзакции загружены: {len(df)} строк для {len(item_codes)} товаров")
    return df


def compute_point_elasticity(
    df_item: pd.DataFrame,
    date: pd.Timestamp,
    price: float,
    window_days: int = 30,
    min_obs: int = 10,
) -> dict:
    """
    Считает точечную эластичность для одного товара методом log-log OLS
    на временно́м окне [date - window_days, date + window_days].

    Если данных недостаточно — возвращает None (вызывающий код
    подставит fallback из elasticity_by_itemid.csv).

    Parameters
    ----------
    df_item      : DataFrame с колонками DATE_, PRICE, AMOUNT для одного ITEMCODE
    date         : центральная дата окна
    price        : гипотетическая/фактическая цена (используется как ref-точка)
    window_days  : полуширина окна в днях
    min_obs      : минимальное количество наблюдений для регрессии

    Returns
    -------
    dict:
        elasticity      : float | None
        baseprice_local : float  — средняя цена в окне (опорная точка)
        avg_qty         : float  — средний дневной объём в окне
        n_obs           : int
        method          : str
    """
    date = pd.Timestamp(date)
    lo = date - timedelta(days=window_days)
    hi = date + timedelta(days=window_days)

    window = df_item[(df_item["DATE_"] >= lo) & (df_item["DATE_"] <= hi)].copy()

    # Агрегируем до дня (дневная цена = средневзвешенная, объём = сумма)
    daily = (
        window.groupby("DATE_")
        .apply(lambda g: pd.Series({
            "PRICE":  np.average(g["PRICE"], weights=g["AMOUNT"]),
            "AMOUNT": g["AMOUNT"].sum(),
        }))
        .reset_index()
    )

    n = len(daily)
    result_base = {
        "baseprice_local": float(daily["PRICE"].mean()) if n > 0 else price,
        "avg_qty":         float(daily["AMOUNT"].mean()) if n > 0 else 0.0,
        "n_obs":           n,
    }

    if n < min_obs:
        log.warning(f"  Недостаточно данных для log-log регрессии: {n} < {min_obs}")
        return {**result_base, "elasticity": None, "method": "insufficient_data"}

    # Убираем нулевые цены и объёмы
    daily = daily[(daily["PRICE"] > 0) & (daily["AMOUNT"] > 0)]
    if len(daily) < min_obs:
        return {**result_base, "elasticity": None, "method": "zero_filter"}

    # Log-log OLS: ln(Q) = a + ε·ln(P)
    log_p = np.log(daily["PRICE"].values)
    log_q = np.log(daily["AMOUNT"].values)

    # Через numpy lstsq (без scipy зависимости)
    X = np.column_stack([np.ones(len(log_p)), log_p])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_q, rcond=None)
    eps = float(coeffs[1])

    # Санитарная проверка
    if not np.isfinite(eps) or abs(eps) > 20:
        log.warning(f"  Нереалистичная эластичность {eps:.2f} — отклонено")
        return {**result_base, "elasticity": None, "method": "unrealistic"}

    log.info(f"  Точечная эластичность: ε={eps:.3f} (n={n}, окно ±{window_days}д)")
    return {**result_base, "elasticity": eps, "method": f"log_log_ols_window{window_days}d"}


def get_elasticity_for_family(
    family: dict,
    transactions_df: pd.DataFrame,
    date: pd.Timestamp,
    hypothetical_prices: dict,
    ctx: dict,
    window_days: int,
) -> dict:
    """
    Считает эластичность для каждого члена семьи.
    Если данных недостаточно — берёт fallback из elasticity_by_itemid.csv.

    Returns
    -------
    dict {item_code: {"elasticity": float, "baseprice": float, "avg_qty": float, "method": str}}
    """
    result = {}
    for ic in family["all_items"]:
        df_item = transactions_df[transactions_df["ITEMCODE"] == ic]
        hyp_price = hypothetical_prices.get(ic)

        # Фактическая/гипотетическая цена как центр окна
        if hyp_price is None and len(df_item) > 0:
            hyp_price = float(df_item["PRICE"].median())

        elast_info = compute_point_elasticity(
            df_item, date, price=hyp_price or 0, window_days=window_days
        )

        # Fallback к базовой эластичности из pipeline
        if elast_info["elasticity"] is None and ic in ctx["elast_base"]:
            bp_base, eps_base = ctx["elast_base"][ic]
            log.info(f"  ITEMCODE {ic}: fallback к базовой ε={eps_base:.3f}")
            elast_info["elasticity"]      = eps_base
            elast_info["baseprice_local"] = bp_base
            elast_info["method"]          = "fallback_baseprice"

        # Если вообще ничего нет
        if elast_info["elasticity"] is None:
            elast_info["elasticity"] = -1.2
            elast_info["method"]     = "fallback_default"
            log.warning(f"  ITEMCODE {ic}: нет данных, дефолтная ε=-1.2")

        result[ic] = {
            "elasticity":   elast_info["elasticity"],
            "baseprice":    elast_info.get("baseprice_local", hyp_price or 0),
            "avg_qty":      elast_info.get("avg_qty", 1.0),
            "n_obs":        elast_info.get("n_obs", 0),
            "method":       elast_info["method"],
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. ОПТИМИЗАЦИЯ ЦЕН ДЛЯ СЕМЬИ
# ══════════════════════════════════════════════════════════════════════════════

def optimize_family_prices(
    family: dict,
    elast_map: dict,
    ctx: dict,
    hypothetical_prices: dict,
    kvi_set: set,
    params: dict,
) -> pd.DataFrame:
    return digit_optimize_family_prices(
        family=family,
        elast_map=elast_map,
        ctx=ctx,
        hypothetical_prices=hypothetical_prices,
        kvi_set=kvi_set,
        params=params,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. ГЛАВНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

def analyze_item(
    item_code: int,
    date: str,
    price: Optional[float] = None,
    window_days: int = 30,
    params: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Главная функция анализа цены для конкретного товара.

    Parameters
    ----------
    item_code    : ITEMCODE целевого товара
    date         : дата анализа (строка "YYYY-MM-DD" или pd.Timestamp)
    price        : гипотетическая цена (если None — берётся из данных)
    window_days  : ширина окна для расчёта эластичности (дней, по умолчанию 30)
    params       : переопределение гиперпараметров оптимизации
    output_path  : путь для сохранения JSON-отчёта (если None — не сохраняется)

    Returns
    -------
    pd.DataFrame с рекомендациями для целевого товара и его семьи
    """
    import time
    t0 = time.time()

    item_code = int(item_code)
    date      = pd.Timestamp(date)

    log.info(f"=== analyze_item: ITEMCODE={item_code}, date={date.date()}, price={price} ===")

    # Гиперпараметры
    default_params = {
        "LAMBDA_KVI": 10.0, "DELTA_KVI": 0.05,
        "MIN_MARGIN_PCT": 0.05, "MAX_PRICE_MULT": 2.5,
    }
    hp_path = BASE_DIR / "hyperparameters.json"
    if hp_path.exists():
        with open(hp_path) as f:
            hp = json.load(f)
            default_params.update(hp.get("agent6", hp))
    if params:
        default_params.update(params)

    # 1. Загружаем KVI-контекст
    ctx = load_kvi_context()

    # 2. Определяем семью
    family = get_item_family(item_code, ctx["sub_map"])

    # 3. Гипотетические цены (target = заданная, остальные = из данных)
    hyp_prices = {item_code: price} if price else {}

    # 4. Загружаем транзакции для всей семьи
    try:
        txn = load_transactions(family["all_items"])
    except FileNotFoundError as e:
        log.warning(f"{e}\nПереходим к fallback-эластичности из elasticity_by_itemid.csv")
        txn = pd.DataFrame(columns=["ITEMCODE", "DATE_", "PRICE", "AMOUNT"])

    # 5. Считаем эластичности
    elast_map = get_elasticity_for_family(
        family, txn, date, hyp_prices, ctx, window_days
    )

    # 6. Оптимизируем
    result_df = optimize_family_prices(
        family, elast_map, ctx, hyp_prices,
        kvi_set=ctx["kvi_set"],
        params=default_params,
    )

    # 7. Добавляем KVI-скор из kvi_scores_full если есть
    if not ctx["kvi_scores"].empty and "kvi_score_final" in ctx["kvi_scores"].columns:
        kvi_score_map = dict(zip(
            ctx["kvi_scores"]["ITEMCODE"].astype(int),
            ctx["kvi_scores"]["kvi_score_final"]
        ))
        result_df["kvi_score"] = result_df["ITEMCODE"].map(kvi_score_map)

    duration = round(time.time() - t0, 2)
    log.info(f"Анализ завершён за {duration}с. Товаров в семье: {len(result_df)}")

    # 8. Сохраняем отчёт
    if output_path:
        report = {
            "item_code":    item_code,
            "date":         str(date.date()),
            "price_query":  price,
            "window_days":  window_days,
            "duration_sec": duration,
            "family":       family,
            "elasticities": {str(k): v for k, v in elast_map.items()},
            "recommendations": result_df.to_dict(orient="records"),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        log.info(f"Отчёт сохранён: {output_path}")

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_itemcode(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит любой вариант названия колонки к ITEMCODE."""
    for c in df.columns:
        if c.lower() in ("itemcode", "itemid", "item_code", "item_id"):
            if c != "ITEMCODE":
                df = df.rename(columns={c: "ITEMCODE"})
            break
    return df


def _rename_col(df: pd.DataFrame, candidates: list, target: str) -> pd.DataFrame:
    for c in candidates:
        if c in df.columns and c != target:
            return df.rename(columns={c: target})
        if c in df.columns:
            return df
    return df


def _get_col(row, candidates: list, default=None):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Анализ цены и оптимизация для конкретного товара и даты"
    )
    parser.add_argument("--item",    type=int,   required=True,  help="ITEMCODE")
    parser.add_argument("--date",    type=str,   required=True,  help="Дата анализа YYYY-MM-DD")
    parser.add_argument("--price",   type=float, default=None,   help="Гипотетическая цена (опционально)")
    parser.add_argument("--window",  type=int,   default=30,     help="Окно эластичности в днях (default=30)")
    parser.add_argument("--output",  type=str,   default=None,   help="Путь для JSON-отчёта")
    parser.add_argument("--lambda-kvi", type=float, default=None, help="Переопределить LAMBDA_KVI")
    args = parser.parse_args()

    extra_params = {}
    if args.lambda_kvi is not None:
        extra_params["LAMBDA_KVI"] = args.lambda_kvi

    out_path = args.output or str(
        OUTPUT_DIR / f"item_analysis_{args.item}_{args.date}.json"
    )

    df = analyze_item(
        item_code=args.item,
        date=args.date,
        price=args.price,
        window_days=args.window,
        params=extra_params if extra_params else None,
        output_path=out_path,
    )

    print("\n" + "═" * 70)
    print(df.to_string(index=False))
    print("═" * 70)
