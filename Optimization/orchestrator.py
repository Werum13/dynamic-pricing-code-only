from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from dataset_builder import build_dataset
from model_builder import build_model
from policy_evaluator import evaluate_policy
from relation_features import build_relation_features
from scenario_generator import build_scenarios


def _resolve_path(base_dir: Path, path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    for key, value in list(config["paths"].items()):
        config["paths"][key] = _resolve_path(config_dir, value)
    return config


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.max_rows is not None:
        config["dataset"]["max_rows"] = int(args.max_rows)
    if args.max_items is not None:
        config["dataset"]["max_items"] = int(args.max_items)
    if args.history_days is not None:
        config["dataset"]["history_days"] = int(args.history_days)


def _configure_logger(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "orchestrator.log"

    logger = logging.getLogger("optimization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _ensure_output_dirs(output_dir: Path) -> None:
    for name in ["logs", "tables", "models", "reports", "manifests"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated ML pricing branch")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")), help="Path to config.json")
    parser.add_argument("--max-rows", type=int, help="Optional raw row cap for dry-runs")
    parser.add_argument("--max-items", type=int, help="Optional item cap for dry-runs")
    parser.add_argument("--history-days", type=int, help="Optional history window override")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    _apply_overrides(config, args)

    output_dir = Path(config["paths"]["output_dir"])
    _ensure_output_dirs(output_dir)
    logger = _configure_logger(output_dir)
    logger.info("Starting isolated ML pricing branch")
    logger.info("Using config: %s", config_path)

    artifacts: dict[str, str] = {}
    artifacts.update(build_dataset(config, logger))
    artifacts.update(build_relation_features(config, logger))
    artifacts.update(build_scenarios(config, logger))
    artifacts.update(build_model(config, logger))
    artifacts.update(evaluate_policy(config, logger))

    run_manifest = {
        "config_path": str(config_path),
        "artifacts": artifacts,
        "overrides": {
            "max_rows": args.max_rows,
            "max_items": args.max_items,
            "history_days": args.history_days,
        },
    }
    manifest_path = output_dir / "manifests" / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    logger.info("Run manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()
