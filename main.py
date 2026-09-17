#!/usr/bin/env python3
"""
Main entry point for Encoding × Modality jailbreaking experiments.

Usage:
    python main.py                  # reads conf/experiment/default.yaml
    python main.py eval_gpt4o      # reads conf/experiment/eval_gpt4o.yaml
    python main.py full            # reads conf/experiment/full.yaml
"""
import sys
from pathlib import Path

from src.experiment import run_experiment_from_preset
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONF_DIR = Path(__file__).resolve().parent / "conf"


def main() -> None:
    # Single positional arg: experiment name (default: "default")
    experiment = sys.argv[1] if len(sys.argv) > 1 else "default"

    logger.info(f"Running experiment: {experiment}")

    try:
        results = run_experiment_from_preset(experiment, CONF_DIR)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Summary (results may be empty if async hasn't returned)
    if results:
        success = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - success
        logger.info(f"Complete: {success} succeeded, {failed} failed")
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
