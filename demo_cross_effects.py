"""
demo_cross_effects.py
=====================
Демонстрация Этапа 1: классификация пар и построение Γ-матрицы.
Запускается на similarity таблице из эмбеддингов.
"""

import io
import pandas as pd
from cross_effects import (
    CrossEffectConfig,
    GammaMatrix,
    PairClassifier,
    apply_cross_effect,
    summarize_gamma,
)

# ─── Загрузка similarity таблицы ─────────────────────────────────────────────
sim_df = pd.read_csv("similarities.csv")

# RAW = """product,similar_product,score
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY 9 V BLISTER,0.8737297654151917
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY AAA*4 SLIM,0.8454064130783081
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY AA*4 PEN,0.836486279964447
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY PEN 4PCS,0.8176235556602478
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY PEN 2 PCS,0.8014150857925415
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK XTRA HEAVY PEN SLIM 4PCS,0.7838951945304871
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK MAX ALKALINE 9V SLIM,0.7252216339111328
# BATTERY KODAK XTRA HEAVY 9 V,BATTERY KODAK CHARGER,0.7042202353477478
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AA 6+2 MAX ALKALINE PEN,1.0000001192092896
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AA*4 MAX ALKALINE PEN,0.9714757204055786
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AA 8+4 ALKALINE PEN,0.9478984475135803
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AA*4 ULTRA ALKALINE PEN,0.9161170125007629
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AAA 6+2 MAX ALKALINE SLIM,0.9041379690170288
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AAA 6+2 MAX ALKALINE SLIM,0.9041379690170288
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AAA 8+4 ALKALINE THIN,0.8757684230804443
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK AAA*4 MAX ALKALINE SLIM,0.8634637594223022
# BATTERY KODAK AA 6+2 MAX ALKALINE PEN,BATTERY KODAK MAX ALKALINE 9V SLIM,0.845506489276886"""

# sim_df = pd.read_csv(io.StringIO(RAW))

# ─── Конфигурация ─────────────────────────────────────────────────────────────
# Порог 0.85 — сильный заменитель (напр. AA 6+2 vs AA 8+4)
# Порог 0.70 — слабый заменитель (напр. XTRA HEAVY vs CHARGER — сомнительно, но sim > 0.70)
cfg = CrossEffectConfig(
    high_thresh  = 0.85,
    low_thresh   = 0.70,
    gamma_max    = 1.5,    # сильная каннибализация
    gamma_weak   = 0.3,    # слабая
    gamma_compl  = -0.5,   # дополнители (пока вручную не задаём)
    max_cross_share = 0.30,
    # Пример ручной пометки дополнителей:
    # manual_complements=[("BATTERY KODAK CHARGER", "BATTERY KODAK AA 6+2 MAX ALKALINE PEN")]
)

# ─── Классификация ────────────────────────────────────────────────────────────
classifier  = PairClassifier(cfg)
classified  = classifier.classify(sim_df)

print("=" * 65)
print("КЛАССИФИКАЦИЯ ПАР")
print("=" * 65)
print(classified.to_string(index=False))

print("\n" + "=" * 65)
print("СТАТИСТИКА ПО ТИПАМ")
print("=" * 65)
print(summarize_gamma(classified).to_string(index=False))

# ─── Построение матрицы Γ ─────────────────────────────────────────────────────
gamma_matrix = GammaMatrix.from_classified(classified)
print(f"\n{gamma_matrix}")

# ─── Пример использования (Этап 2 preview) ───────────────────────────────────
# Допустим, базовый спрос на "BATTERY KODAK XTRA HEAVY 9 V" = 100 шт.
# Цена соседа "BATTERY KODAK MAX ALKALINE 9V SLIM" выросла на 10 руб.
# → покупатели перетекают на наш товар.

item_i = "BATTERY KODAK XTRA HEAVY 9 V"

current_prices = {
    "BATTERY KODAK XTRA HEAVY 9 V BLISTER"    : 120.0,
    "BATTERY KODAK XTRA HEAVY AAA*4 SLIM"      : 95.0,
    "BATTERY KODAK MAX ALKALINE 9V SLIM"       : 115.0,  # вырос с 105 → +10
    "BATTERY KODAK CHARGER"                    : 350.0,
}

ref_prices = {
    "BATTERY KODAK XTRA HEAVY 9 V BLISTER"    : 120.0,
    "BATTERY KODAK XTRA HEAVY AAA*4 SLIM"      : 95.0,
    "BATTERY KODAK MAX ALKALINE 9V SLIM"       : 105.0,  # базовая
    "BATTERY KODAK CHARGER"                    : 350.0,
}

q_base = 100.0

delta = gamma_matrix.cross_demand_delta(item_i, current_prices, ref_prices)
q_adj = apply_cross_effect(q_base, delta, cfg)

print("\n" + "=" * 65)
print("ПРИМЕР CROSS-ЭФФЕКТА")
print("=" * 65)
print(f"Товар i:            {item_i}")
print(f"Базовый спрос Q:    {q_base:.1f}")
print(f"Cross-delta ΔQ:     {delta:+.3f}")
print(f"Скорректированный Q: {q_adj:.3f}")

# ─── Матрица Γ как DataFrame (для экспорта / визуализации) ────────────────────
gamma_df = gamma_matrix.to_dataframe()
print("\n" + "=" * 65)
print("GAMMA MATRIX (DataFrame)")
print("=" * 65)
print(gamma_df.to_string(index=False))
