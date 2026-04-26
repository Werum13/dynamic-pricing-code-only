from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
for p in [MODULE_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from digit_optimization_module import digit_optimize_family_prices  # noqa: E402
from elasticity_day import estimate_family_elasticity_for_day  # noqa: E402
from kvi_context import get_item_family, load_kvi_context  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("item_price_analyzer")


def analyze_item(
    item_code: int,
    date: str,
    price: Optional[float] = None,
    window_days: int = 30,
    params: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    t0 = time.time()

    item_code = int(item_code)
    date_ts = pd.Timestamp(date).normalize()
    log.info("analyze_item: ITEMCODE=%s, date=%s, price=%s", item_code, date_ts.date(), price)

    default_params = {
        "LAMBDA_KVI": 10.0,
        "DELTA_KVI": 0.05,
        "MIN_MARGIN_PCT": 0.05,
        "MAX_PRICE_MULT": 2.5,
        "MIN_PRICE_MULT": 0.5,
        "DIGIT_PRICE_DELTAS": [-0.09, -0.05, -0.02, 0.0, 0.02, 0.05, 0.09],
    }
    hp_path = None
    for candidate in [PROJECT_ROOT / "hyperparameters.json", PROJECT_ROOT / "KVI" / "hyperparameters.json"]:
        if candidate.exists():
            hp_path = candidate
            break

    if hp_path is not None:
        with open(hp_path, encoding="utf-8") as f:
            hp = json.load(f)
            default_params.update(hp.get("agent6", hp))
    if params:
        default_params.update(params)

    ctx = load_kvi_context()
    family = get_item_family(item_code, ctx["sub_map"])
    hypothetical_prices = {item_code: float(price)} if price is not None else {}

    elast_map = estimate_family_elasticity_for_day(
        family=family,
        date=date_ts,
        hypothetical_prices=hypothetical_prices,
        ctx=ctx,
        window_days=window_days,
    )

    result_df = digit_optimize_family_prices(
        family=family,
        elast_map=elast_map,
        ctx=ctx,
        hypothetical_prices=hypothetical_prices,
        kvi_set=ctx["kvi_set"],
        params=default_params,
    )

    if not ctx["kvi_scores"].empty and "kvi_score_final" in ctx["kvi_scores"].columns:
        score_map = dict(zip(ctx["kvi_scores"]["ITEMCODE"].astype(int), ctx["kvi_scores"]["kvi_score_final"]))
        result_df["kvi_score"] = result_df["ITEMCODE"].map(score_map)

    duration = round(time.time() - t0, 2)
    log.info("analyze_item finished in %ss, family size=%s", duration, len(result_df))

    if output_path:
        report = {
            "item_code": item_code,
            "date": str(date_ts.date()),
            "price_query": price,
            "window_days": window_days,
            "duration_sec": duration,
            "family": family,
            "elasticities": {str(k): v for k, v in elast_map.items()},
            "recommendations": result_df.to_dict(orient="records"),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        log.info("Saved report: %s", out)

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Item-level day simulation: elasticity -> KVI -> policy optimization")
    parser.add_argument("--item", type=int, required=True, help="ITEMCODE")
    parser.add_argument("--date", type=str, required=True, help="Analysis date YYYY-MM-DD")
    parser.add_argument("--price", type=float, default=None, help="Optional hypothetical target-item price")
    parser.add_argument("--window", type=int, default=30, help="Rolling window size for stats")
    parser.add_argument("--output", type=str, default=None, help="Path to JSON output report")
    parser.add_argument("--lambda-kvi", type=float, default=None, help="Optional LAMBDA_KVI override")
    args = parser.parse_args()

    extra_params = {}
    if args.lambda_kvi is not None:
        extra_params["LAMBDA_KVI"] = args.lambda_kvi

    output_path = args.output or str(OUTPUT_DIR / f"item_analysis_{args.item}_{args.date}.json")
    df = analyze_item(
        item_code=args.item,
        date=args.date,
        price=args.price,
        window_days=args.window,
        params=extra_params if extra_params else None,
        output_path=output_path,
    )
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
