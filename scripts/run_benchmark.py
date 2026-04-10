#!/usr/bin/env python3
"""CLI entry point: run the full acceptance-rate benchmark.

Usage
-----
    python scripts/run_benchmark.py                          # defaults
    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/run_benchmark.py --pair llama2-7b-tinyllama --datasets humaneval gsm8k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.utils import load_config, seed_everything, get_device
from src.models import load_model_pair
from src.datasets import load_all_datasets, iterate_samples
from src.measure import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run acceptance-rate benchmark")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Name of model pair to use (default: first in config)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to run (default: all)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max_samples for all datasets",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg)

    # Select model pair
    pairs_cfg = cfg["model_pairs"]
    if args.pair:
        pair_cfg = next(
            (p for p in pairs_cfg if p["name"] == args.pair), None
        )
        if pair_cfg is None:
            logger.error(f"Model pair '{args.pair}' not found in config")
            sys.exit(1)
    else:
        pair_cfg = pairs_cfg[0]

    # Filter datasets if requested
    datasets_cfg = cfg["datasets"]
    if args.datasets:
        datasets_cfg = {k: v for k, v in datasets_cfg.items() if k in args.datasets}

    # Override max_samples if requested
    if args.max_samples:
        for ds_cfg in datasets_cfg.values():
            ds_cfg["max_samples"] = args.max_samples

    # Load models
    pair = load_model_pair(pair_cfg, device)

    # Load datasets
    all_samples = load_all_datasets(datasets_cfg)
    flat_samples = list(iterate_samples(all_samples))
    logger.info(f"Total samples to measure: {len(flat_samples)}")

    # Run
    records = run_benchmark(
        pair=pair,
        samples=flat_samples,
        tree_cfg=cfg["tree"],
        meas_cfg=cfg["measurement"],
        datasets_cfg=datasets_cfg,
    )

    logger.success(f"Done. {len(records)} records collected.")


if __name__ == "__main__":
    main()
