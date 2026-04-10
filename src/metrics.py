"""Compute summary metrics from raw per-node records.

Key metrics
-----------
- α(d)             : mean acceptance probability at tree depth d
- α(d, task)       : conditioned on task type
- α(d, pos_bin)    : conditioned on token position bin
- E[L]             : expected accepted chain length per step
- P(L ≥ d)         : cumulative acceptance (chain reaches depth d)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert raw record list to a DataFrame."""
    df = pd.DataFrame(records)
    # Ensure numeric types
    for col in [
        "token_position", "position_bin", "tree_depth", "node_id",
        "parent_id", "token_id", "draft_prob", "target_prob",
        "acceptance_prob", "target_entropy",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── α by depth ───────────────────────────────────────────────────────

def alpha_by_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Mean acceptance probability grouped by tree depth.

    Returns DataFrame with columns: depth, alpha_mean, alpha_std, count.
    """
    g = df.groupby("tree_depth")["acceptance_prob"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["depth", "alpha_mean", "alpha_std", "count"]
    return out


def alpha_by_depth_and_task(df: pd.DataFrame) -> pd.DataFrame:
    """Mean acceptance probability grouped by (tree_depth, task_type)."""
    g = df.groupby(["tree_depth", "task_type"])["acceptance_prob"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["depth", "task_type", "alpha_mean", "alpha_std", "count"]
    return out


def alpha_by_depth_and_position(df: pd.DataFrame) -> pd.DataFrame:
    """Mean acceptance probability grouped by (tree_depth, position_bin)."""
    g = df.groupby(["tree_depth", "position_bin"])["acceptance_prob"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = ["depth", "position_bin", "alpha_mean", "alpha_std", "count"]
    return out


def alpha_by_depth_task_position(df: pd.DataFrame) -> pd.DataFrame:
    """Full 3-way breakdown: (depth, task_type, position_bin)."""
    g = df.groupby(["tree_depth", "task_type", "position_bin"])["acceptance_prob"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "depth", "task_type", "position_bin",
        "alpha_mean", "alpha_std", "count",
    ]
    return out


# ── Expected accepted length ─────────────────────────────────────────

def expected_accepted_length(df: pd.DataFrame) -> pd.DataFrame:
    """Expected accepted chain length E[L] per task type.

    For each generation step (sample_id × token_position), we simulate
    acceptance along the *greedy path* (depth 1 → max_depth, following the
    node with highest draft_prob at each depth) and record the first
    rejection depth.  E[L] is the mean of those lengths.

    As a simpler approximation (since we have α per depth), we compute:
      E[L] ≈ Σ_{d=1}^{D} Π_{d'=1}^{d} α(d')

    This is returned per task type.
    """
    adt = alpha_by_depth_and_task(df)
    tasks = adt["task_type"].unique()
    results = []
    for task in tasks:
        sub = adt[adt["task_type"] == task].sort_values("depth")
        alphas = sub["alpha_mean"].values
        # Cumulative product
        cum_prod = np.cumprod(alphas)
        el = cum_prod.sum()  # E[L] ≈ Σ P(L ≥ d)
        results.append({
            "task_type": task,
            "expected_length": el,
            "max_depth": len(alphas),
        })
    return pd.DataFrame(results)


# ── Cumulative acceptance P(L ≥ d) ──────────────────────────────────

def cumulative_acceptance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute P(L ≥ d) for each task type.

    Uses the chain rule: P(L ≥ d) = Π_{d'=1}^{d} α(d', task).
    """
    adt = alpha_by_depth_and_task(df)
    rows = []
    for task in adt["task_type"].unique():
        sub = adt[adt["task_type"] == task].sort_values("depth")
        alphas = sub["alpha_mean"].values
        cum_prod = np.cumprod(alphas)
        for i, (d, cp) in enumerate(
            zip(sub["depth"].values, cum_prod)
        ):
            rows.append({
                "task_type": task,
                "depth": int(d),
                "prob_chain_ge_d": cp,
            })
    return pd.DataFrame(rows)


# ── Entropy vs acceptance ────────────────────────────────────────────

def entropy_vs_acceptance(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Bin target entropy and compute mean acceptance per bin."""
    if "target_entropy" not in df.columns:
        logger.warning("No target_entropy in data; skipping")
        return pd.DataFrame()

    df = df.dropna(subset=["target_entropy", "acceptance_prob"])
    df["entropy_bin"] = pd.cut(df["target_entropy"], bins=n_bins)
    g = df.groupby(["entropy_bin", "task_type"])["acceptance_prob"]
    out = g.agg(["mean", "count"]).reset_index()
    out.columns = ["entropy_bin", "task_type", "alpha_mean", "count"]
    out["entropy_mid"] = out["entropy_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else np.nan
    )
    return out


# ── Draft calibration ────────────────────────────────────────────────

def draft_calibration(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Bin draft confidence and compute actual acceptance rate per bin.

    Tests whether the draft model's confidence ≈ true acceptance rate
    (EAGLE-2's calibration insight).
    """
    df = df.dropna(subset=["draft_prob", "acceptance_prob"])
    df["draft_conf_bin"] = pd.cut(df["draft_prob"], bins=n_bins)
    g = df.groupby("draft_conf_bin")["acceptance_prob"]
    out = g.agg(["mean", "count"]).reset_index()
    out.columns = ["draft_conf_bin", "actual_accept_rate", "count"]
    out["draft_conf_mid"] = out["draft_conf_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else np.nan
    )
    return out


# ── Summary table ────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-task summary with key statistics."""
    rows = []
    for task, sub in df.groupby("task_type"):
        rows.append({
            "task_type": task,
            "n_samples": sub["sample_id"].nunique(),
            "n_nodes": len(sub),
            "mean_alpha": sub["acceptance_prob"].mean(),
            "std_alpha": sub["acceptance_prob"].std(),
            "median_alpha": sub["acceptance_prob"].median(),
            "mean_target_entropy": sub["target_entropy"].mean()
            if "target_entropy" in sub.columns else np.nan,
            "mean_draft_prob": sub["draft_prob"].mean(),
            "mean_target_prob": sub["target_prob"].mean(),
        })
    return pd.DataFrame(rows)
