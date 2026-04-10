#!/usr/bin/env python3
"""CLI entry point: aggregate raw logs and produce figures.

Usage
-----
    python scripts/run_analysis.py results/raw/llama2-7b-tinyllama_full.jsonl
    python scripts/run_analysis.py results/raw/*.jsonl --out-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from analysis.aggregate import aggregate
from analysis.plot import plot_all
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Aggregate & plot benchmark results")
    parser.add_argument(
        "raw_files", nargs="+",
        help="Path(s) to raw JSONL log files",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config (used for analysis settings)",
    )
    parser.add_argument(
        "--tables-dir", type=str, default="results/tables",
        help="Directory for CSV summary tables",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="results/figures",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--format", type=str, default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure output format",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    analysis_cfg = cfg.get("analysis", {})

    # If multiple files, concatenate them
    if len(args.raw_files) > 1:
        from src.utils import load_jsonl, save_jsonl, ensure_dir
        all_records = []
        for f in args.raw_files:
            all_records.extend(load_jsonl(f))
        merged_path = Path(args.tables_dir) / "_merged.jsonl"
        ensure_dir(args.tables_dir)
        save_jsonl(all_records, merged_path)
        raw_path = str(merged_path)
    else:
        raw_path = args.raw_files[0]

    # Aggregate
    logger.info("Aggregating raw records...")
    tables = aggregate(raw_path, args.tables_dir)

    # Plot
    logger.info("Generating figures...")
    fmt = args.format or analysis_cfg.get("output_format", "pdf")
    figsize = tuple(analysis_cfg.get("figsize", [8, 5]))
    dpi = analysis_cfg.get("dpi", 300)

    plot_all(
        tables=tables,
        out_dir=args.figures_dir,
        fmt=fmt,
        figsize=figsize,
        dpi=dpi,
    )

    logger.success("Analysis complete.")


if __name__ == "__main__":
    main()
