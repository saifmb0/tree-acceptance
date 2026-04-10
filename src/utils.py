"""Utility helpers: seeding, logging, I/O."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger


# ── Seeding ──────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


# ── I/O helpers ──────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_jsonl(records: list[dict], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Saved {len(records)} records → {p}")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Config loading ───────────────────────────────────────────────────

def load_config(path: str = "configs/default.yaml") -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Device helpers ───────────────────────────────────────────────────

def get_device(cfg: dict | None = None) -> torch.device:
    if cfg and cfg.get("device"):
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_mem_summary() -> str:
    """Return a short string describing GPU memory usage."""
    if not torch.cuda.is_available():
        return "No CUDA"
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_mem / 1e9
        lines.append(f"GPU{i}: {alloc:.1f}/{total:.1f} GB")
    return " | ".join(lines)
