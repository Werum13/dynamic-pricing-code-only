"""
AGENT 4: KVI_SCORER
====================
Рассчитывает итоговый KVI-score для каждого товара.

INPUT:
    - output/kvi/behavioral_features.csv  (от Agent 2)
    - output/kvi/substitute_map.json      (от Agent 3)
        - output/kvi/elasticity_by_itemid.csv  (item-level elasticity по ITEMID)
    - data/LSTCSV* или data/elasticity.csv (если есть, иначе прокси из данных)

OUTPUT:
    - output/kvi/kvi_scores_full.csv
    - output/kvi/kvi_candidates.csv
"""

from pathlib import Path

import pandas as pd
import numpy as np
import json, os, time

from elasticity_utils import standardize_elasticity_source

BASE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE)
DATA = os.path.join(WORKSPACE_ROOT, 'data')
OUT  = os.path.join(WORKSPACE_ROOT, 'output', 'kvi')
os.makedirs(OUT, exist_ok=True)

log = {"agent": "KVI_SCORER", "status": "SUCCESS",
       "rows_processed": 0, "warnings": [], "output_files": []}

def minmax_norm(series):
    """Min-max нормализация. Защита от нулевого диапазона."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

def main():
    print("[AGENT 4] KVI_SCORER starting...", flush=True)
    t0 = time.time()

    bf_path      = os.path.join(OUT, 'behavioral_features.csv')
    sub_map_path = os.path.join(OUT, 'substitute_map.json')
    elast_path   = os.path.join(OUT, 'elasticity_by_itemid.csv')

    for p in [bf_path, sub_map_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run Agents 2 & 3 first.")

    df = pd.read_csv(bf_path, dtype={'ITEMID': str})
    print(f"[AGENT 4] Loaded behavioral_features: {len(df):,} items", flush=True)

    # Исключаем товары с order_count < 30
    df = df[df['order_count'] >= 30].copy()
    print(f"[AGENT 4] After order_count>=30 filter: {len(df):,} items", flush=True)

    # ── Эластичность ────────────────────────────────────────────────────────
    if os.path.exists(elast_path):
        elast = pd.read_csv(elast_path, dtype={'ITEMID': str})
        print(
            f"[AGENT 4] Loaded standardized elasticity: {len(elast):,} ITEMIDs",
            flush=True,
        )
        df = df.merge(elast, on='ITEMID', how='left')
        df['no_elasticity'] = df['elasticity'].isna().astype(int)
        print(f"[AGENT 4] Elasticity matched: {(~df['elasticity'].isna()).sum():,} items", flush=True)
    else:
        elast_df, elast_meta = standardize_elasticity_source(
            Path(DATA),
            Path(DATA) / 'Categories_ENG.csv',
            Path(OUT),
        )

        if elast_df is not None:
            print(
                f"[AGENT 4] Standardized elasticity built from {elast_meta.get('source_path')} "
                f"({elast_meta.get('source_kind')}, {elast_meta.get('mapping_method')})",
                flush=True,
            )
            df = df.merge(elast_df, on='ITEMID', how='left')
            df['no_elasticity'] = df['elasticity'].isna().astype(int)
            print(f"[AGENT 4] Elasticity matched: {(~df['elasticity'].isna()).sum():,} items", flush=True)
        else:
            # Прокси-эластичность: используем price_cv как прокси чувствительности к цене.
            # Чем выше price_cv (больше разброс цен), тем более чувствителен покупатель.
            # Масштабируем в диапазон [-3, -0.1] (отрицательные значения как у эластичности).
            log["warnings"].append("elasticity source not found — used proxy elasticity based on price_cv.")
            df['no_elasticity'] = 1
            cv = df['price_cv'].fillna(0).clip(0, 1)
            df['elasticity'] = -(0.1 + cv * 2.9)  # диапазон [-3.0, -0.1]
            print("[AGENT 4] Proxy elasticity computed from price_cv.", flush=True)

    # ── Шаг 1: Нормализация внутри CATEGORY2 ────────────────────────────────
    print("[AGENT 4] Normalizing metrics within CATEGORY2...", flush=True)
    norm_cols = {
        'elasticity_norm':    lambda g: minmax_norm(g['elasticity'].abs()),
        'penetration_norm':   lambda g: minmax_norm(g['penetration_rate']),
        'frequency_norm':     lambda g: minmax_norm(g['freq_per_buyer']),
        'basket_share_norm':  lambda g: minmax_norm(g['avg_basket_share']),
        'price_cv_norm':      lambda g: minmax_norm(g['price_cv'].fillna(0)),
        'repeat_buyer_norm':  lambda g: minmax_norm(g['repeat_buyer_rate']),
    }

    df_norm = df.copy()
    for col, func in norm_cols.items():
        df_norm[col] = (
            df_norm.groupby('CATEGORY2', group_keys=False)
            .apply(lambda g: func(g))
        )

    # ── Шаг 2: Базовый score ─────────────────────────────────────────────────
    df_norm['kvi_score_base'] = (
        0.30 * df_norm['elasticity_norm']
      + 0.25 * df_norm['penetration_norm']
      + 0.20 * df_norm['frequency_norm']
      + 0.15 * df_norm['basket_share_norm']
      + 0.05 * df_norm['price_cv_norm']
      + 0.05 * df_norm['repeat_buyer_norm']
    )

    # ── Шаг 3: Поправки ──────────────────────────────────────────────────────
    with open(sub_map_path, 'r', encoding='utf-8') as f:
        sub_map = json.load(f)

    # a) Субститутная поправка
    score_map = df_norm.set_index('ITEMID')['kvi_score_base'].to_dict()
    def substitute_penalty(row):
        subs = sub_map.get(row['ITEMID'], {}).get('substitutes', [])
        for s in subs:
            if score_map.get(s, 0) > row['kvi_score_base']:
                return 0.90  # понижаем на 10%
        return 1.0
    df_norm['substitute_penalty'] = df_norm.apply(substitute_penalty, axis=1)

    # b) Сезонная поправка: CV квартальных продаж
    q_cols = ['q1_orders', 'q2_orders', 'q3_orders', 'q4_orders']
    q_data = df_norm[q_cols].values.astype(float)
    q_mean = q_data.mean(axis=1)
    q_std  = q_data.std(axis=1)
    q_cv   = np.where(q_mean > 0, q_std / q_mean, 0)
    df_norm['seasonal_penalty'] = np.where(q_cv > 0.5, 0.85, 1.0)

    # c) Бренд-поправка: топ-5 брендов по penetration в CATEGORY1
    brand_pen = (df_norm.groupby(['CATEGORY1', 'BRAND'])['penetration_rate']
                 .mean().reset_index()
                 .sort_values('penetration_rate', ascending=False))
    top_brands = set()
    for cat1, grp in brand_pen.groupby('CATEGORY1'):
        top_brands.update(grp.head(5)['BRAND'].tolist())
    df_norm['brand_bonus'] = df_norm['BRAND'].apply(lambda b: 1.10 if b in top_brands else 1.0)

    # Финальный score
    df_norm['kvi_score_final'] = (
        df_norm['kvi_score_base']
        * df_norm['substitute_penalty']
        * df_norm['seasonal_penalty']
        * df_norm['brand_bonus']
    )

    # ── Шаг 4: Ранжирование ──────────────────────────────────────────────────
    df_norm['kvi_rank_in_category2'] = (
        df_norm.groupby('CATEGORY2')['kvi_score_final']
        .rank(method='first', ascending=False).astype(int)
    )
    df_norm['kvi_rank_in_category3'] = (
        df_norm[df_norm['CATEGORY3'].notna()]
        .groupby('CATEGORY3')['kvi_score_final']
        .rank(method='first', ascending=False).astype(int)
    )

    # ── Сохранение ────────────────────────────────────────────────────────────
    full_cols = [
        'ITEMID', 'ITEMNAME', 'CATEGORY1', 'CATEGORY2', 'CATEGORY3', 'CATEGORY4', 'BRAND',
        'elasticity', 'no_elasticity',
        'elasticity_mean', 'elasticity_std', 'elasticity_min', 'elasticity_max',
        'elasticity_latest', 'elasticity_method', 'n_obs',
        'elasticity_norm', 'penetration_norm', 'frequency_norm',
        'basket_share_norm', 'price_cv_norm', 'repeat_buyer_norm',
        'kvi_score_base', 'substitute_penalty', 'seasonal_penalty', 'brand_bonus',
        'kvi_score_final', 'kvi_rank_in_category2', 'kvi_rank_in_category3',
        'order_count', 'unique_buyers', 'penetration_rate', 'avg_price',
        'price_cv', 'avg_basket_share', 'freq_per_buyer', 'repeat_buyer_rate'
    ]
    full_cols = [c for c in full_cols if c in df_norm.columns]
    full_path = os.path.join(OUT, 'kvi_scores_full.csv')
    df_norm[full_cols].to_csv(full_path, index=False)

    # kvi_candidates: топ-3 в CATEGORY2, order_count >= 30, есть elasticity (кроме прокси)
    cand_df = df_norm[
        (df_norm['kvi_rank_in_category2'] <= 3)
        & (df_norm['order_count'] >= 30)
        & (df_norm['no_elasticity'] == 0)
    ][['ITEMID', 'ITEMNAME', 'CATEGORY2', 'BRAND',
       'kvi_score_final', 'elasticity', 'penetration_rate']]

    # если elasticity.csv не было — кандидаты всё равно включаем (прокси помечены)
    if not os.path.exists(elast_path):
        cand_df = df_norm[df_norm['kvi_rank_in_category2'] <= 3][
            ['ITEMID', 'ITEMNAME', 'CATEGORY2', 'BRAND',
             'kvi_score_final', 'elasticity', 'penetration_rate']
        ]
        log["warnings"].append("Кандидаты включают товары с прокси-эластичностью (no_elasticity=1).")

    cand_path = os.path.join(OUT, 'kvi_candidates.csv')
    cand_df.sort_values('kvi_score_final', ascending=False).to_csv(cand_path, index=False)

    log["rows_processed"] = len(df_norm)
    log["output_files"]   = [full_path, cand_path]
    print(f"[AGENT 4] Done: {len(df_norm):,} items scored, {len(cand_df):,} candidates  ({round(time.time()-t0,1)}s)", flush=True)

    with open(os.path.join(OUT, 'agent4_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

if __name__ == '__main__':
    main()
