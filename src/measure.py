"""Core measurement loop.

For each sample, we step through the target model's autoregressive generation
one token at a time. At each step we:
  1. Build a draft tree rooted at the current prefix.
  2. Verify the tree with the target model.
  3. Log per-node acceptance data.
  4. Advance the prefix by one *target* token (greedy).

This gives us a complete record of acceptance probabilities indexed by
(sample_id, task_type, token_position, tree_depth, branch_rank).
"""

from __future__ import annotations

import time
from dataclasses import asdict

import torch
from loguru import logger
from tqdm import tqdm

from .datasets import Sample
from .models import ModelPair
from .tree import DraftTree, build_draft_tree, verify_tree, compute_target_entropy
from .utils import save_jsonl, ensure_dir


@torch.no_grad()
def measure_sample(
    pair: ModelPair,
    sample: Sample,
    tree_cfg: dict,
    meas_cfg: dict,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Run the measurement protocol for a single sample.

    Returns a list of per-node records (dicts) ready for JSONL logging.
    """

    tokenizer = pair.tokenizer
    target = pair.target_model
    draft = pair.draft_model
    device = pair.device

    # Tokenise the prompt
    enc = tokenizer(
        sample.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    prefix_len = input_ids.shape[1]
    records: list[dict] = []

    # Step through autoregressive generation of the target model
    generated_tokens = 0
    current_ids = input_ids
    current_mask = attention_mask

    while generated_tokens < max_new_tokens:
        # 1. Build draft tree from current prefix
        tree = build_draft_tree(
            draft_model=draft,
            input_ids=current_ids,
            attention_mask=current_mask,
            max_depth=tree_cfg.get("max_depth", 6),
            top_k=tree_cfg.get("top_k", 10),
            max_branch=tree_cfg.get("max_branch", 5),
            max_nodes=tree_cfg.get("max_nodes", 64),
            temperature=tree_cfg.get("temperature", 0.0),
        )

        if tree.size == 0:
            break

        # 2. Verify with target model
        tree = verify_tree(
            target_model=target,
            input_ids=current_ids,
            attention_mask=current_mask,
            tree=tree,
        )

        # 3. Optionally compute target entropy at each node
        entropies = compute_target_entropy(
            target_model=target,
            input_ids=current_ids,
            attention_mask=current_mask,
            tree=tree,
        )

        # 4. Log per-node records
        position_bin_size = meas_cfg.get("position_bin_size", 32)
        for node_id in range(tree.size):
            records.append({
                "sample_id": sample.id,
                "task_type": sample.task_type,
                "model_pair": "",  # filled by caller
                "token_position": generated_tokens,
                "position_bin": generated_tokens // position_bin_size,
                "tree_depth": tree.depth[node_id],
                "node_id": node_id,
                "parent_id": tree.parent[node_id],
                "token_id": tree.tokens[node_id],
                "draft_prob": tree.draft_probs[node_id],
                "target_prob": tree.target_probs[node_id],
                "acceptance_prob": tree.acceptance_probs[node_id],
                "target_entropy": entropies[node_id],
            })

        # 5. Advance by one greedy target token
        # Get the target model's next token from its own distribution
        out = target(
            input_ids=current_ids,
            attention_mask=current_mask,
            use_cache=False,
        )
        next_logits = out.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Extend the prefix
        current_ids = torch.cat([current_ids, next_token], dim=1)
        current_mask = torch.cat(
            [current_mask, torch.ones(1, 1, dtype=torch.long, device=device)],
            dim=1,
        )
        generated_tokens += 1

        # Memory guard: truncate if prefix grows too long
        if current_ids.shape[1] > 2048 + max_new_tokens:
            break

    return records


def run_benchmark(
    pair: ModelPair,
    samples: list[Sample],
    tree_cfg: dict,
    meas_cfg: dict,
    datasets_cfg: dict,
) -> list[dict]:
    """Run the full benchmark across all samples.

    Parameters
    ----------
    pair : ModelPair
    samples : list[Sample]
    tree_cfg : dict from config
    meas_cfg : dict from config
    datasets_cfg : dict mapping dataset name → config (for max_new_tokens)

    Returns
    -------
    list of all per-node records
    """

    all_records: list[dict] = []
    log_dir = meas_cfg.get("log_dir", "results/raw")
    ensure_dir(log_dir)

    logger.info(
        f"Starting benchmark: {len(samples)} samples, "
        f"model pair = {pair.name}"
    )
    t0 = time.time()

    # Map task_type → max_new_tokens from dataset configs
    task_max_tokens = {}
    for ds_name, ds_cfg in datasets_cfg.items():
        task_max_tokens[ds_cfg.get("task_type", ds_name)] = ds_cfg.get(
            "max_new_tokens", 256
        )

    for i, sample in enumerate(tqdm(samples, desc="Measuring")):
        max_new = task_max_tokens.get(sample.task_type, 256)

        try:
            records = measure_sample(
                pair=pair,
                sample=sample,
                tree_cfg=tree_cfg,
                meas_cfg=meas_cfg,
                max_new_tokens=max_new,
            )
            # Tag with model pair name
            for r in records:
                r["model_pair"] = pair.name
            all_records.extend(records)

        except Exception as e:
            logger.warning(f"Error on sample {sample.id}: {e}")
            continue

        # Periodically save checkpoint
        if (i + 1) % 50 == 0:
            ckpt_path = f"{log_dir}/{pair.name}_checkpoint_{i+1}.jsonl"
            save_jsonl(all_records, ckpt_path)
            logger.info(
                f"Checkpoint at sample {i+1}: "
                f"{len(all_records)} records, "
                f"elapsed = {time.time() - t0:.0f}s"
            )

    elapsed = time.time() - t0
    logger.success(
        f"Benchmark complete: {len(all_records)} records "
        f"from {len(samples)} samples in {elapsed:.0f}s"
    )

    # Final save
    out_path = f"{log_dir}/{pair.name}_full.jsonl"
    save_jsonl(all_records, out_path)

    return all_records
