"""Dataset loading and prompt formatting for each task type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset
from loguru import logger


@dataclass
class Sample:
    """A single benchmark sample ready for generation."""

    id: str
    task_type: str  # code | math | chat | reasoning
    prompt: str
    reference: str | None = None  # optional ground-truth for later analysis


# ── Per-dataset loaders ──────────────────────────────────────────────

def _load_humaneval(cfg: dict) -> list[Sample]:
    ds = load_dataset("openai_humaneval", split=cfg["split"])
    samples = []
    for i, row in enumerate(ds):
        if i >= cfg.get("max_samples", 164):
            break
        samples.append(Sample(
            id=f"humaneval_{row['task_id']}",
            task_type="code",
            prompt=row["prompt"],
            reference=row.get("canonical_solution"),
        ))
    return samples


def _load_math(cfg: dict) -> list[Sample]:
    ds = load_dataset("hendrycks/competition_math", split=cfg["split"])
    samples = []
    for i, row in enumerate(ds):
        if i >= cfg.get("max_samples", 200):
            break
        prompt = (
            "Solve the following math problem step by step.\n\n"
            f"Problem: {row['problem']}\n\nSolution:"
        )
        samples.append(Sample(
            id=f"math_{i}",
            task_type="math",
            prompt=prompt,
            reference=row.get("solution"),
        ))
    return samples


def _load_sharegpt(cfg: dict) -> list[Sample]:
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split=cfg["split"],
    )
    samples = []
    for i, row in enumerate(ds):
        if len(samples) >= cfg.get("max_samples", 200):
            break
        convs = row.get("conversations", [])
        # Take the first human turn as the prompt
        human_turns = [c for c in convs if c.get("from") == "human"]
        if not human_turns:
            continue
        prompt_text = human_turns[0].get("value", "").strip()
        if len(prompt_text) < 10:
            continue
        # Get first assistant turn as reference if available
        assistant_turns = [c for c in convs if c.get("from") == "gpt"]
        ref = assistant_turns[0].get("value") if assistant_turns else None
        samples.append(Sample(
            id=f"sharegpt_{i}",
            task_type="chat",
            prompt=prompt_text,
            reference=ref,
        ))
    return samples


def _load_gsm8k(cfg: dict) -> list[Sample]:
    ds = load_dataset("openai/gsm8k", cfg.get("config", "main"), split=cfg["split"])
    samples = []
    for i, row in enumerate(ds):
        if i >= cfg.get("max_samples", 200):
            break
        prompt = (
            "Solve the following problem step by step, showing your reasoning.\n\n"
            f"Question: {row['question']}\n\nAnswer:"
        )
        samples.append(Sample(
            id=f"gsm8k_{i}",
            task_type="reasoning",
            prompt=prompt,
            reference=row.get("answer"),
        ))
    return samples


# ── Dispatcher ───────────────────────────────────────────────────────

_LOADERS = {
    "humaneval": _load_humaneval,
    "math":      _load_math,
    "sharegpt":  _load_sharegpt,
    "gsm8k":     _load_gsm8k,
}


def load_all_datasets(datasets_cfg: dict) -> dict[str, list[Sample]]:
    """Load all configured datasets.

    Returns
    -------
    dict mapping dataset name → list of Sample
    """
    all_samples: dict[str, list[Sample]] = {}
    for ds_name, ds_cfg in datasets_cfg.items():
        loader = _LOADERS.get(ds_name)
        if loader is None:
            logger.warning(f"No loader for dataset '{ds_name}', skipping")
            continue
        logger.info(f"Loading dataset: {ds_name}")
        samples = loader(ds_cfg)
        all_samples[ds_name] = samples
        logger.info(f"  → {len(samples)} samples loaded")
    return all_samples


def iterate_samples(
    all_samples: dict[str, list[Sample]],
) -> Iterator[Sample]:
    """Flat iterator over all samples from all datasets."""
    for samples in all_samples.values():
        yield from samples
