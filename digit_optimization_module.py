from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def demand_power(price: float, q0: float, p0: float, elasticity: float) -> float:
    if price <= 0 or p0 <= 0 or q0 <= 0:
        return 0.0
    return float(q0) * float((price / p0) ** elasticity)


def _default_deltas() -> list[float]:
    return [-0.09, -0.05, -0.02, 0.0, 0.02, 0.05, 0.09]


def digit_optimize_family_prices(
    family: dict[str, Any],
    elast_map: dict[int, dict[str, Any]],
    ctx: dict[str, Any],
    hypothetical_prices: dict[int, float],
    kvi_set: set[int],
    params: dict[str, Any],
) -> pd.DataFrame:
    items = family["all_items"]
    rows: list[dict[str, Any]] = []

    deltas = params.get("DIGIT_PRICE_DELTAS", _default_deltas())
    deltas = sorted(set(float(x) for x in deltas))
    if 0.0 not in deltas:
        deltas.append(0.0)
        deltas = sorted(deltas)

    lambda_kvi = float(params.get("LAMBDA_KVI", 10.0))
    kvi_delta_cap = float(params.get("DELTA_KVI", 0.05))
    min_margin_pct = float(params.get("MIN_MARGIN_PCT", 0.05))
    max_price_mult = float(params.get("MAX_PRICE_MULT", 2.5))
    min_price_mult = float(params.get("MIN_PRICE_MULT", 0.5))

    for ic in items:
        info = elast_map[ic]
        baseprice = float(info.get("baseprice", 0.0) or 0.0)
        ref_price = float(hypothetical_prices.get(ic, baseprice) or baseprice)
        if ref_price <= 0:
            ref_price = max(baseprice, 1.0)

        elasticity = float(info.get("elasticity", -1.2))
        if not np.isfinite(elasticity):
            elasticity = -1.2
        elasticity = float(np.clip(elasticity, -8.0, -0.01))

        avg_qty = float(info.get("avg_qty", 1.0) or 1.0)
        avg_qty = max(avg_qty, 1e-6)

        cost = float(ctx["cost_map"].get(ic, ref_price * 0.6))
        is_kvi = ic in kvi_set

        role = "target" if ic == family["target"] else (
            "substitute" if ic in family["substitutes"] else
            "complement" if ic in family["complements"] else
            "cannibal"
        )

        floor_price = max(cost * (1.0 + min_margin_pct), ref_price * min_price_mult)
        cap_price = ref_price * (1.0 + kvi_delta_cap) if is_kvi else ref_price * max_price_mult
        if cap_price < floor_price:
            cap_price = floor_price

        best = None
        for dlt in deltas:
            if is_kvi and abs(dlt) > kvi_delta_cap + 1e-12:
                continue

            candidate_price = ref_price * (1.0 + dlt)
            if not (floor_price <= candidate_price <= cap_price):
                continue

            candidate_qty = demand_power(candidate_price, avg_qty, ref_price, elasticity)
            margin_new = (candidate_price - cost) * candidate_qty
            penalty = lambda_kvi * (dlt ** 2) if is_kvi else 0.0
            objective = margin_new - penalty

            if best is None or objective > best["objective"]:
                best = {
                    "price": candidate_price,
                    "qty": candidate_qty,
                    "margin": margin_new,
                    "penalty": penalty,
                    "objective": objective,
                }

        if best is None:
            fallback_price = floor_price
            fallback_qty = demand_power(fallback_price, avg_qty, ref_price, elasticity)
            best = {
                "price": fallback_price,
                "qty": fallback_qty,
                "margin": (fallback_price - cost) * fallback_qty,
                "penalty": 0.0,
                "objective": (fallback_price - cost) * fallback_qty,
            }

        q_cur = demand_power(ref_price, avg_qty, ref_price, elasticity)
        m_cur = (ref_price - cost) * q_cur

        rows.append(
            {
                "ITEMCODE": ic,
                "role": role,
                "is_kvi": is_kvi,
                "current_price": round(ref_price, 4),
                "recommended_price": round(best["price"], 4),
                "price_change_pct": round((best["price"] - ref_price) / ref_price * 100.0, 2) if ref_price > 0 else None,
                "elasticity": round(elasticity, 4),
                "elasticity_method": info.get("method", "unknown"),
                "n_obs_window": int(info.get("n_obs", 0)),
                "cost": round(cost, 4),
                "margin_current": round(m_cur, 2),
                "margin_new": round(best["margin"], 2),
                "margin_delta": round(best["margin"] - m_cur, 2),
                "demand_current": round(q_cur, 2),
                "demand_new": round(best["qty"], 2),
            }
        )

    return pd.DataFrame(rows)
