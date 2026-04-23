"""Small auto-refresh pipeline for the KVI analysis workspace.

Usage:
    python kvi_pipeline.py
    python kvi_pipeline.py --force

The script checks whether the final outputs are missing or older than the
inputs/code. If so, it runs kvi_orchestrator.py. Otherwise it prints a short
summary from the existing outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent
DATA_DIR = WORKSPACE_ROOT / "data"
OUTPUT_DIR = WORKSPACE_ROOT / "output" / "kvi"

SOURCE_PATHS = [
    BASE_DIR / "kvi_orchestrator.py",
    BASE_DIR / "elasticity_utils.py",
    BASE_DIR / "kvi_validator.py",
    BASE_DIR / "feature_engineer.py",
    BASE_DIR / "substitute_detector.py",
    BASE_DIR / "kvi_scorer.py",
    BASE_DIR / "report_builder.py",
    BASE_DIR / "hyperparameters.json",
    DATA_DIR / "Orders.csv",
    DATA_DIR / "Order_Details.csv",
    DATA_DIR / "Categories_ENG.csv",
]


def elasticity_source_paths() -> list[Path]:
    lst_sources = sorted(
        [path for path in DATA_DIR.glob("LSTCSV*") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    paths = list(lst_sources)
    legacy = DATA_DIR / "elasticity.csv"
    if legacy.exists():
        paths.append(legacy)
    return paths

FINAL_OUTPUTS = [
    OUTPUT_DIR / "pipeline_log.json",
    OUTPUT_DIR / "kvi_report.html",
    OUTPUT_DIR / "kvi_final_list.xlsx",
    OUTPUT_DIR / "elasticity_by_itemid.csv",
    OUTPUT_DIR / "agent3_corr_hist.png",
    OUTPUT_DIR / "agent3_lift_hist.png",
]


def latest_existing_mtime(paths: list[Path]) -> tuple[float, Path | None]:
    latest_mtime = 0.0
    latest_path: Path | None = None
    for path in paths:
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_mtime, latest_path


def oldest_existing_mtime(paths: list[Path]) -> tuple[float, list[Path]]:
    mtimes: list[float] = []
    missing: list[Path] = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        mtimes.append(path.stat().st_mtime)
    return (min(mtimes) if mtimes else 0.0), missing


def needs_refresh(force: bool) -> tuple[bool, str]:
    if force:
        return True, "forced rebuild"

    final_mtime, missing_outputs = oldest_existing_mtime(FINAL_OUTPUTS)
    if missing_outputs:
        missing_list = ", ".join(path.name for path in missing_outputs)
        return True, f"missing outputs: {missing_list}"

    source_mtime, source_path = latest_existing_mtime(SOURCE_PATHS + elasticity_source_paths())
    if source_mtime > final_mtime:
        changed = source_path.name if source_path else "unknown input"
        return True, f"{changed} is newer than the final outputs"

    return False, "outputs are up to date"


def run_orchestrator() -> int:
    print("[PIPELINE] Running orchestrator...", flush=True)
    result = subprocess.run([sys.executable, str(BASE_DIR / "kvi_orchestrator.py")], cwd=BASE_DIR)
    return result.returncode


def print_summary() -> None:
    candidates_path = OUTPUT_DIR / "kvi_candidates.csv"
    if not candidates_path.exists():
        print("[PIPELINE] No KVI candidates found yet.", flush=True)
        return

    try:
        import pandas as pd

        candidates = pd.read_csv(candidates_path)
        top5 = candidates.sort_values("kvi_score_final", ascending=False).head(5)

        print(f"[PIPELINE] KVI candidates: {len(candidates)}", flush=True)
        if "CATEGORY2" in candidates.columns:
            print(f"[PIPELINE] CATEGORY2 groups: {candidates['CATEGORY2'].nunique()}", flush=True)
        print("[PIPELINE] Top 5:", flush=True)
        for _, row in top5.iterrows():
            item_name = row.get("ITEMNAME", row.get("ITEMID", "?"))
            category2 = row.get("CATEGORY2", "-")
            score = row.get("kvi_score_final", 0.0)
            print(f"  - {item_name} | {category2} | score={score:.3f}", flush=True)
    except Exception as exc:  # pragma: no cover - summary is best effort
        print(f"[PIPELINE] Summary unavailable: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-refresh KVI analysis pipeline")
    parser.add_argument("--force", action="store_true", help="rebuild outputs even if they look current")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    refresh_needed, reason = needs_refresh(args.force)
    if refresh_needed:
        print(f"[PIPELINE] Rebuild needed: {reason}", flush=True)
        code = run_orchestrator()
        if code != 0:
            return code
    else:
        print(f"[PIPELINE] {reason}", flush=True)

    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())