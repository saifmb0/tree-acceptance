"""Aggregate raw JSONL logs into summary DataFrames and save as CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.metrics import (
    alpha_by_depth,
    alpha_by_depth_and_task,
    alpha_by_depth_and_position,
    alpha_by_depth_task_position,
    cumulative_acceptance,
    draft_calibration,
    entropy_vs_acceptance,
    expected_accepted_length,
    records_to_df,
    summary_table,
)
from src.utils import ensure_dir, load_jsonl


def aggregate(raw_path: str, out_dir: str = "results/tables") -> dict[str, pd.DataFrame]:
    """Load raw JSONL and produce all aggregate tables.

    Parameters
    ----------
    raw_path : str  Path to a JSONL file with per-node records.
    out_dir  : str  Directory to write CSV tables.

    Returns
    -------
    dict mapping table name → DataFrame
    """
    ensure_dir(out_dir)
    logger.info(f"Loading raw records from {raw_path}")
    records = load_jsonl(raw_path)
    df = records_to_df(records)
    logger.info(f"Loaded {len(df)} records, {df['sample_id'].nunique()} samples")

    tables: dict[str, pd.DataFrame] = {}

    # 1. Summary
    tables["summary"] = summary_table(df)

    # 2. α by depth
    tables["alpha_depth"] = alpha_by_depth(df)

    # 3. α by depth × task
    tables["alpha_depth_task"] = alpha_by_depth_and_task(df)

    # 4. α by depth × position
    tables["alpha_depth_pos"] = alpha_by_depth_and_position(df)

    # 5. α by depth × task × position
    tables["alpha_depth_task_pos"] = alpha_by_depth_task_position(df)

    # 6. Cumulative acceptance
    tables["cumulative"] = cumulative_acceptance(df)

    # 7. Expected accepted length
    tables["expected_length"] = expected_accepted_length(df)

    # 8. Entropy vs acceptance
    tables["entropy_accept"] = entropy_vs_acceptance(df)

    # 9. Draft calibration
    tables["draft_calibration"] = draft_calibration(df)

    # Save all tables
    for name, table in tables.items():
        path = Path(out_dir) / f"{name}.csv"
        table.to_csv(path, index=False)
        logger.info(f"  Saved {name} → {path} ({len(table)} rows)")

    return tables
