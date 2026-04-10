"""Publication-quality plots for the acceptance-rate study.

Produces the six key figures described in NARRATIVE.md:
  1. Depth–acceptance curves (one line per task)
  2. Position heatmaps (one per task)
  3. Expected accepted length bar chart
  4. Cumulative acceptance CDF
  5. Draft calibration scatter
  6. Entropy vs. acceptance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils import ensure_dir

# ── Style ────────────────────────────────────────────────────────────

TASK_COLORS = {
    "code":      "#2196F3",
    "math":      "#FF5722",
    "chat":      "#4CAF50",
    "reasoning": "#9C27B0",
}
TASK_ORDER = ["code", "math", "chat", "reasoning"]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def _save(fig, name: str, out_dir: str, fmt: str = "pdf", dpi: int = 300):
    ensure_dir(out_dir)
    path = Path(out_dir) / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved figure → {path}")


# ── 1. Depth–acceptance curves ───────────────────────────────────────

def plot_depth_acceptance(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (8, 5),
):
    """Fig 1: α(d) per task type — the key figure of the paper."""
    df = tables["alpha_depth_task"]
    fig, ax = plt.subplots(figsize=figsize)

    for task in TASK_ORDER:
        sub = df[df["task_type"] == task].sort_values("depth")
        if sub.empty:
            continue
        ax.plot(
            sub["depth"], sub["alpha_mean"],
            marker="o", linewidth=2, label=task.capitalize(),
            color=TASK_COLORS.get(task, "gray"),
        )
        ax.fill_between(
            sub["depth"],
            sub["alpha_mean"] - sub["alpha_std"],
            sub["alpha_mean"] + sub["alpha_std"],
            alpha=0.15,
            color=TASK_COLORS.get(task, "gray"),
        )

    ax.set_xlabel("Tree Depth $d$")
    ax.set_ylabel(r"Mean Acceptance Rate $\alpha(d)$")
    ax.set_title("Token Acceptance Rate vs. Speculation Depth")
    ax.legend(title="Task")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, int(df["depth"].max()) + 1))

    _save(fig, "fig1_depth_acceptance", out_dir, fmt)


# ── 2. Position heatmaps ────────────────────────────────────────────

def plot_position_heatmaps(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (10, 4),
):
    """Fig 2: Heatmap of α(d, position_bin) for each task."""
    df = tables["alpha_depth_task_pos"]
    tasks = [t for t in TASK_ORDER if t in df["task_type"].unique()]

    fig, axes = plt.subplots(
        1, len(tasks), figsize=(figsize[0], figsize[1]),
        sharey=True, squeeze=False,
    )

    for idx, task in enumerate(tasks):
        ax = axes[0, idx]
        sub = df[df["task_type"] == task]
        if sub.empty:
            ax.set_title(task.capitalize())
            continue
        pivot = sub.pivot_table(
            index="depth", columns="position_bin",
            values="alpha_mean", aggfunc="mean",
        )
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd_r", vmin=0, vmax=1,
            cbar=(idx == len(tasks) - 1),
            cbar_kws={"label": r"$\alpha$"} if idx == len(tasks) - 1 else {},
        )
        ax.set_title(task.capitalize())
        ax.set_xlabel("Position Bin")
        if idx == 0:
            ax.set_ylabel("Tree Depth $d$")
        else:
            ax.set_ylabel("")
        ax.invert_yaxis()

    fig.suptitle(
        "Acceptance Rate by Speculation Depth and Token Position",
        y=1.02, fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "fig2_position_heatmaps", out_dir, fmt)


# ── 3. Expected accepted length bar chart ────────────────────────────

def plot_expected_length(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (6, 4),
):
    """Fig 3: E[L] per task type."""
    df = tables["expected_length"]
    fig, ax = plt.subplots(figsize=figsize)

    tasks = [t for t in TASK_ORDER if t in df["task_type"].values]
    vals = [
        df[df["task_type"] == t]["expected_length"].values[0] for t in tasks
    ]
    colors = [TASK_COLORS.get(t, "gray") for t in tasks]

    bars = ax.bar(
        [t.capitalize() for t in tasks], vals,
        color=colors, edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("Expected Accepted Length $E[L]$")
    ax.set_title("Expected Speculative Chain Length by Task")

    # Annotate bars
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{v:.2f}", ha="center", va="bottom", fontsize=10,
        )

    _save(fig, "fig3_expected_length", out_dir, fmt)


# ── 4. Cumulative acceptance CDF ────────────────────────────────────

def plot_cumulative_acceptance(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (8, 5),
):
    """Fig 4: P(L ≥ d) per task type."""
    df = tables["cumulative"]
    fig, ax = plt.subplots(figsize=figsize)

    for task in TASK_ORDER:
        sub = df[df["task_type"] == task].sort_values("depth")
        if sub.empty:
            continue
        ax.plot(
            sub["depth"], sub["prob_chain_ge_d"],
            marker="s", linewidth=2, label=task.capitalize(),
            color=TASK_COLORS.get(task, "gray"),
        )

    ax.set_xlabel("Depth $d$")
    ax.set_ylabel(r"$P(L \geq d)$")
    ax.set_title("Cumulative Acceptance: Probability of Chain Reaching Depth $d$")
    ax.legend(title="Task")
    ax.set_ylim(0, 1.05)

    _save(fig, "fig4_cumulative_acceptance", out_dir, fmt)


# ── 5. Draft calibration scatter ────────────────────────────────────

def plot_draft_calibration(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (6, 6),
):
    """Fig 5: Draft confidence vs. actual acceptance rate."""
    df = tables["draft_calibration"]
    if df.empty:
        logger.warning("Draft calibration table empty; skipping plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df["draft_conf_mid"], df["actual_accept_rate"],
        s=df["count"] / df["count"].max() * 200,
        alpha=0.7, edgecolor="black", linewidth=0.3,
        color="#2196F3",
    )
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Draft Model Confidence")
    ax.set_ylabel("Actual Acceptance Rate")
    ax.set_title("Draft Model Calibration\n(validates EAGLE-2 insight)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    _save(fig, "fig5_draft_calibration", out_dir, fmt)


# ── 6. Entropy vs. acceptance ───────────────────────────────────────

def plot_entropy_vs_acceptance(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (8, 5),
):
    """Fig 6: Target entropy vs. acceptance probability, coloured by task."""
    df = tables["entropy_accept"]
    if df.empty:
        logger.warning("Entropy-acceptance table empty; skipping plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for task in TASK_ORDER:
        sub = df[df["task_type"] == task].dropna(subset=["entropy_mid"])
        if sub.empty:
            continue
        ax.plot(
            sub["entropy_mid"], sub["alpha_mean"],
            marker="o", linewidth=1.5, label=task.capitalize(),
            color=TASK_COLORS.get(task, "gray"),
        )

    ax.set_xlabel("Target Distribution Entropy (nats)")
    ax.set_ylabel(r"Mean Acceptance Rate $\alpha$")
    ax.set_title("Acceptance Rate vs. Target Entropy")
    ax.legend(title="Task")
    ax.set_ylim(0, 1.05)

    _save(fig, "fig6_entropy_vs_acceptance", out_dir, fmt)


# ── Convenience: generate all figures ────────────────────────────────

def plot_all(
    tables: dict[str, pd.DataFrame],
    out_dir: str = "results/figures",
    fmt: str = "pdf",
    figsize: tuple = (8, 5),
    dpi: int = 300,
):
    """Generate all six key figures."""
    logger.info(f"Generating all figures → {out_dir}/")
    plot_depth_acceptance(tables, out_dir, fmt, figsize)
    plot_position_heatmaps(tables, out_dir, fmt, (figsize[0] + 2, figsize[1]))
    plot_expected_length(tables, out_dir, fmt, (figsize[0] - 2, figsize[1] - 1))
    plot_cumulative_acceptance(tables, out_dir, fmt, figsize)
    plot_draft_calibration(tables, out_dir, fmt, (figsize[1], figsize[1]))
    plot_entropy_vs_acceptance(tables, out_dir, fmt, figsize)
    logger.success("All figures generated.")
