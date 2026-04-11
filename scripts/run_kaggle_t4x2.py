"""
Characterising Token Acceptance in Speculative Decoding
α vs. Tree Depth × Task Type × Token Position

Hardware : Kaggle T4 × 2 (16 GB each)
Runtime  : ~4-6 GPU-hours

Usage:
    python run_kaggle_t4x2.py [--outdir results] [--max-samples 50]
"""

# ── 0. Imports ────────────────────────────────────────────────────────
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display needed on Kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm


def _fix_torchvision_cuda_mismatch() -> None:
    """Remove broken torchvision install if CUDA majors don't match torch.

    On Kaggle/Colab, users often upgrade torch without upgrading torchvision,
    which can crash *any* transformers import chain that touches image_utils.
    """
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        msg = str(exc)
        if "compiled with different CUDA major versions" in msg:
            print("[runtime-fix] Detected torch/torchvision CUDA mismatch; uninstalling torchvision.")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"],
                check=False,
            )
            for mod in list(sys.modules.keys()):
                if mod.startswith("torchvision"):
                    del sys.modules[mod]
        else:
            print(f"[runtime-fix] torchvision import warning: {exc}")


_fix_torchvision_cuda_mismatch()
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 1. CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--outdir",      default="results")
parser.add_argument("--max-samples", type=int, default=50,
                    help="per-dataset sample cap")
parser.add_argument("--max-depth",   type=int, default=6)
parser.add_argument("--top-k",       type=int, default=10)
parser.add_argument("--max-branch",  type=int, default=5)
parser.add_argument("--max-nodes",   type=int, default=64)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--target-id",   default="TheBloke/Llama-2-7B-Chat-GPTQ")
parser.add_argument("--draft-id",    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument(
    "--allow-partial-tasks",
    action="store_true",
    help="Allow benchmark to run even if some task types are missing datasets.",
)
args = parser.parse_args()

# ── 2. Timestamp & output dir ─────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(OUTDIR / f"run_{TS}.log", level="DEBUG", rotation="50 MB")

# ── 3. Config ─────────────────────────────────────────────────────────
CFG = {
    "target_id":  args.target_id,
    "draft_id":   args.draft_id,
    "max_depth":  args.max_depth,
    "top_k":      args.top_k,
    "max_branch": args.max_branch,
    "max_nodes":  args.max_nodes,
    "temperature": args.temperature,
    "position_bin_size": 32,
    "max_new_tokens": {
        "code":      256,
        "math":      512,
        "chat":      512,
        "reasoning": 512,
    },
    "max_samples": {
        "humaneval": args.max_samples,
        "math":      args.max_samples,
        "sharegpt":  args.max_samples,
        "gsm8k":     args.max_samples,
    },
}

with open(OUTDIR / f"config_{TS}.json", "w") as f:
    json.dump(CFG, f, indent=2)
logger.info(f"Config saved → {OUTDIR}/config_{TS}.json")

# ── 4. Seeds & device ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            prop = torch.cuda.get_device_properties(i)
            total = getattr(prop, "total_memory", None) or getattr(prop, "total_mem", None)
            total_str = f"{total/1e9:.1f} GB" if total is not None else "unknown"
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {total_str}")
        except Exception:
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ── 5. Load models ────────────────────────────────────────────────────
# ── Model loading ─────────────────────────────────────────────────────
# auto-gptq 0.7.1 is incompatible with torch 2.11+ (removed C++ API).
# transformers >= 4.35 + optimum load GPTQ models natively: just call
# from_pretrained with no extra quantization_config — it reads the model's
# own config.json.  For plain fp16 drafts we pass torch_dtype explicitly.
def _ensure_gptq_backend_ready() -> None:
    """Ensure optimum GPTQ backend has a valid QuantizeConfig symbol.

    Some Kaggle stacks ship an optimum/transformers combo where
    optimum.gptq.quantizer expects QuantizeConfig to exist but it was not
    imported (NameError at runtime). We patch it from available providers.
    """
    try:
        import optimum.gptq.quantizer as oq
    except Exception as exc:
        raise RuntimeError(
            "optimum GPTQ backend is not importable. Install/upgrade optimum."
        ) from exc

    if hasattr(oq, "QuantizeConfig"):
        return

    def _providers():
        found = []
        try:
            from gptqmodel import QuantizeConfig as _QC
            found.append(("gptqmodel", _QC))
        except Exception:
            pass
        try:
            from auto_gptq import QuantizeConfig as _QC
            found.append(("auto_gptq", _QC))
        except Exception:
            pass
        return found

    providers = _providers()

    if not providers:
        logger.warning(
            "No GPTQ QuantizeConfig provider found. Attempting to install `gptqmodel`..."
        )
        import subprocess
        install_cmd = [sys.executable, "-m", "pip", "install", "-q", "gptqmodel"]
        proc = subprocess.run(install_cmd, check=False)
        if proc.returncode != 0:
            logger.warning("Automatic install of `gptqmodel` failed.")
        providers = _providers()

    if providers:
        src, qc_cls = providers[0]
        oq.QuantizeConfig = qc_cls
        logger.info(f"Patched optimum GPTQ QuantizeConfig from {src}.")
        return

    raise RuntimeError(
        "GPTQ backend missing QuantizeConfig. Install one provider with:\n"
        "  pip install gptqmodel\n"
        "or use a compatible auto-gptq environment."
    )


def _load_model(model_id: str, is_target: bool):
    """Load target (GPTQ-only) or draft (fp16)."""
    if is_target:
        # The GPTQ repo stores its quantization_config in config.json;
        # transformers reads it automatically — do NOT pass a second one.
        _ensure_gptq_backend_ready()
        logger.info(f"Loading {model_id} via transformers GPTQ (optimum backend)")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info(f"Loading {model_id} fp16")
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )


logger.info(f"Loading target: {CFG['target_id']}")
target_tokenizer = AutoTokenizer.from_pretrained(
    CFG["target_id"], trust_remote_code=True, padding_side="left"
)
if target_tokenizer.pad_token is None:
    target_tokenizer.pad_token = target_tokenizer.eos_token

target_model = _load_model(CFG["target_id"], is_target=True)
target_model.eval()

logger.info(f"Loading draft: {CFG['draft_id']}")
draft_model = _load_model(CFG["draft_id"], is_target=False)
draft_model.eval()

for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    logger.info(f"  GPU {i}: {alloc:.2f} GB allocated")

# ── 6. Dataset loaders ────────────────────────────────────────────────
@dataclass
class Sample:
    id: str
    task_type: str
    prompt: str
    reference: str = None


def _load_dataset_with_hub_fallback(*args, **kwargs):
    """Try current endpoint first, then retry against huggingface.co if needed."""
    try:
        return load_dataset(*args, **kwargs)
    except Exception as first_exc:
        endpoint_vars = ("HF_ENDPOINT", "HF_HUB_ENDPOINT", "HUGGINGFACE_HUB_ENDPOINT")
        active = {k: os.environ.get(k) for k in endpoint_vars if os.environ.get(k)}
        if not active:
            raise

        logger.warning(
            f"Dataset load failed via mirror ({first_exc}); retrying via huggingface.co"
        )

        for key in endpoint_vars:
            os.environ.pop(key, None)

        try:
            return load_dataset(*args, **kwargs)
        finally:
            for key, value in active.items():
                os.environ[key] = value


def load_humaneval(max_n: int) -> list[Sample]:
    ds = _load_dataset_with_hub_fallback("openai_humaneval", split="test")
    return [
        Sample(f'he_{r["task_id"]}', "code", r["prompt"], r.get("canonical_solution"))
        for i, r in enumerate(ds) if i < max_n
    ]


def load_math(max_n: int) -> list[Sample]:
    ds = _load_dataset_with_hub_fallback("hendrycks/competition_math", split="test")
    return [
        Sample(
            f"math_{i}", "math",
            f'Solve step by step.\nProblem: {r["problem"]}\nSolution:',
            r.get("solution"),
        )
        for i, r in enumerate(ds) if i < max_n
    ]


def load_gsm8k(max_n: int) -> list[Sample]:
    ds = _load_dataset_with_hub_fallback("openai/gsm8k", "main", split="test")
    return [
        Sample(
            f"gsm_{i}", "reasoning",
            f'Show your reasoning step by step.\nQ: {r["question"]}\nA:',
            r.get("answer"),
        )
        for i, r in enumerate(ds) if i < max_n
    ]


def load_sharegpt(max_n: int) -> list[Sample]:
    # Mirror layouts differ: try default config first, then known sub-config.
    candidates = [None, "HTML_cleaned_raw_dataset"]
    last_exc = None
    ds = None
    for cfg_name in candidates:
        try:
            if cfg_name is None:
                ds = _load_dataset_with_hub_fallback(
                    "anon8231489123/ShareGPT_Vicuna_unfiltered", split="train"
                )
            else:
                ds = _load_dataset_with_hub_fallback(
                    "anon8231489123/ShareGPT_Vicuna_unfiltered", cfg_name, split="train"
                )
            break
        except Exception as exc:
            last_exc = exc
    if ds is None:
        raise RuntimeError(f"ShareGPT dataset/config unavailable: {last_exc}")
    samples: list[Sample] = []
    for i, r in enumerate(ds):
        if len(samples) >= max_n:
            break
        convs = r.get("conversations", [])
        human = [c for c in convs if c.get("from") == "human"]
        if not human:
            continue
        txt = human[0].get("value", "").strip()
        if len(txt) < 10:
            continue
        samples.append(Sample(f"sgpt_{i}", "chat", txt))
    return samples


def _safe_load(loader_name: str, fn, max_n: int) -> list[Sample]:
    try:
        rows = fn(max_n)
        logger.info(f"Loaded {loader_name}: {len(rows)} samples")
        return rows
    except Exception as exc:
        logger.warning(f"Skipping dataset {loader_name} due to load error: {exc}")
        return []


all_samples = (
    _safe_load("openai_humaneval", load_humaneval, CFG["max_samples"]["humaneval"])
    + _safe_load("hendrycks/competition_math", load_math, CFG["max_samples"]["math"])
    + _safe_load("openai/gsm8k", load_gsm8k, CFG["max_samples"]["gsm8k"])
    + _safe_load("ShareGPT", load_sharegpt, CFG["max_samples"]["sharegpt"])
)
logger.info(f"Total samples: {len(all_samples)}")
for t in ["code", "math", "reasoning", "chat"]:
    logger.info(f"  {t}: {sum(1 for s in all_samples if s.task_type == t)}")

task_counts = {
    t: sum(1 for s in all_samples if s.task_type == t)
    for t in ["code", "math", "reasoning", "chat"]
}
missing_tasks = [t for t, n in task_counts.items() if n == 0]

if missing_tasks and not args.allow_partial_tasks:
    raise RuntimeError(
        "Missing required task datasets for: "
        f"{', '.join(missing_tasks)}. "
        "Refusing to run partial benchmark. "
        "Fix dataset access or pass --allow-partial-tasks to proceed."
    )

if not all_samples:
    raise RuntimeError(
        "No datasets could be loaded in this environment. "
        "Check network/mirror access or reduce enabled datasets."
    )

# ── 7. Draft tree & verification ──────────────────────────────────────
@dataclass
class DraftTree:
    tokens:          list = field(default_factory=list)
    parent:          list = field(default_factory=list)
    depth:           list = field(default_factory=list)
    draft_probs:     list = field(default_factory=list)
    draft_logprobs:  list = field(default_factory=list)
    target_probs:    list = field(default_factory=list)
    target_logprobs: list = field(default_factory=list)
    acceptance_probs: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.tokens)


def _path_to_root(tree: DraftTree, nid: int) -> list[int]:
    path: list[int] = []
    while nid >= 0:
        path.append(tree.tokens[nid])
        nid = tree.parent[nid]
    path.reverse()
    return path


@torch.no_grad()
def build_draft_tree(
    draft_m, input_ids, attn_mask,
    max_depth=6, top_k=10, max_branch=5, max_nodes=64, temperature=0.0,
) -> DraftTree:
    tree = DraftTree()
    device = input_ids.device

    out = draft_m(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits[:, -1, :]
    probs = F.softmax(
        logits / max(temperature, 1e-8) if temperature > 0 else logits, dim=-1
    )
    tk_p, tk_i = probs.topk(min(top_k, max_branch), dim=-1)
    tk_p, tk_i = tk_p.squeeze(0), tk_i.squeeze(0)

    for j in range(tk_i.shape[0]):
        if tree.size >= max_nodes:
            break
        tree.tokens.append(tk_i[j].item())
        tree.parent.append(-1)
        tree.depth.append(1)
        tree.draft_probs.append(tk_p[j].item())
        tree.draft_logprobs.append(tk_p[j].log().item())

    frontier = list(range(tree.size))
    while frontier and tree.size < max_nodes:
        next_f: list[int] = []
        for nid in frontier:
            if tree.size >= max_nodes or tree.depth[nid] >= max_depth:
                continue
            path = _path_to_root(tree, nid)
            p_t = torch.tensor([path], dtype=torch.long, device=device)
            fi = torch.cat([input_ids, p_t], dim=1)
            fm = torch.ones(1, fi.shape[1], dtype=torch.long, device=device)
            o = draft_m(input_ids=fi, attention_mask=fm, use_cache=False)
            lg = o.logits[:, -1, :]
            p = F.softmax(
                lg / max(temperature, 1e-8) if temperature > 0 else lg, dim=-1
            )
            tp, ti = p.topk(min(top_k, max_branch), dim=-1)
            tp, ti = tp.squeeze(0), ti.squeeze(0)
            for j in range(ti.shape[0]):
                if tree.size >= max_nodes:
                    break
                cid = tree.size
                tree.tokens.append(ti[j].item())
                tree.parent.append(nid)
                tree.depth.append(tree.depth[nid] + 1)
                tree.draft_probs.append(tp[j].item())
                tree.draft_logprobs.append(tp[j].log().item())
                next_f.append(cid)
        frontier = next_f
    return tree


@torch.no_grad()
def verify_tree(target_m, input_ids, attn_mask, tree: DraftTree) -> DraftTree:
    device = input_ids.device
    n = tree.size
    tree.target_probs    = [0.0] * n
    tree.target_logprobs = [0.0] * n
    tree.acceptance_probs = [0.0] * n

    children_of = set(tree.parent)
    leaves = [i for i in range(n) if i not in children_of]
    scored: set[int] = set()

    for leaf in leaves:
        path_ids: list[int] = []
        nid = leaf
        while nid >= 0:
            path_ids.append(nid)
            nid = tree.parent[nid]
        path_ids.reverse()

        path_tokens = [tree.tokens[nid] for nid in path_ids]
        p_t = torch.tensor([path_tokens], dtype=torch.long, device=device)
        fi = torch.cat([input_ids, p_t], dim=1)
        fm = torch.ones(1, fi.shape[1], dtype=torch.long, device=device)
        out = target_m(input_ids=fi, attention_mask=fm, use_cache=False)
        logits = out.logits
        plen = input_ids.shape[1]

        for idx, node_id in enumerate(path_ids):
            if node_id in scored:
                continue
            lpos = plen - 1 + idx
            pv = F.softmax(logits[0, lpos, :], dim=-1)
            tok = tree.tokens[node_id]
            pt = pv[tok].item()
            pd_ = max(tree.draft_probs[node_id], 1e-10)
            tree.target_probs[node_id]    = pt
            tree.target_logprobs[node_id] = pv[tok].log().item() if pt > 0 else -float("inf")
            tree.acceptance_probs[node_id] = min(1.0, pt / pd_)
            scored.add(node_id)
    return tree


@torch.no_grad()
def compute_target_entropy(target_m, input_ids, attn_mask, tree: DraftTree) -> list[float]:
    device = input_ids.device
    n = tree.size
    ents = [0.0] * n
    children_of = set(tree.parent)
    leaves = [i for i in range(n) if i not in children_of]
    done: set[int] = set()

    for leaf in leaves:
        path_ids: list[int] = []
        nid = leaf
        while nid >= 0:
            path_ids.append(nid)
            nid = tree.parent[nid]
        path_ids.reverse()
        pt = [tree.tokens[n_] for n_ in path_ids]
        fi = torch.cat(
            [input_ids, torch.tensor([pt], dtype=torch.long, device=device)], dim=1
        )
        fm = torch.ones(1, fi.shape[1], dtype=torch.long, device=device)
        out = target_m(input_ids=fi, attention_mask=fm, use_cache=False)
        plen = input_ids.shape[1]
        for idx, node_id in enumerate(path_ids):
            if node_id in done:
                continue
            lv = out.logits[0, plen - 1 + idx, :]
            pv = F.softmax(lv, dim=-1)
            lp = F.log_softmax(lv, dim=-1)
            ents[node_id] = -(pv * lp).sum().item()
            done.add(node_id)
    return ents

# ── 8. Measurement loop ───────────────────────────────────────────────
# CSV columns — declared once so the header is written at row 0
_CSV_FIELDS = [
    "sample_id", "task_type", "token_position", "position_bin",
    "tree_depth", "node_id", "parent_id", "token_id",
    "draft_prob", "target_prob", "acceptance_prob", "target_entropy",
]


@torch.no_grad()
def measure_sample(
    sample: Sample,
    csv_writer: csv.DictWriter,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Run one sample through the draft-then-verify loop.

    Records are yielded line-by-line into *csv_writer* as they arrive.
    Returns the list of dicts for in-memory aggregation.
    """
    enc = target_tokenizer(
        sample.prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(DEVICE)
    cur_ids  = enc["input_ids"]
    cur_mask = enc["attention_mask"]
    records: list[dict] = []
    gen = 0

    while gen < max_new_tokens:
        tree = build_draft_tree(
            draft_model, cur_ids, cur_mask,
            max_depth=CFG["max_depth"],
            top_k=CFG["top_k"],
            max_branch=CFG["max_branch"],
            max_nodes=CFG["max_nodes"],
            temperature=CFG["temperature"],
        )
        if tree.size == 0:
            break

        tree = verify_tree(target_model, cur_ids, cur_mask, tree)
        ents = compute_target_entropy(target_model, cur_ids, cur_mask, tree)

        pbs = CFG["position_bin_size"]
        for nid in range(tree.size):
            row = {
                "sample_id":       sample.id,
                "task_type":       sample.task_type,
                "token_position":  gen,
                "position_bin":    gen // pbs,
                "tree_depth":      tree.depth[nid],
                "node_id":         nid,
                "parent_id":       tree.parent[nid],
                "token_id":        tree.tokens[nid],
                "draft_prob":      tree.draft_probs[nid],
                "target_prob":     tree.target_probs[nid],
                "acceptance_prob": tree.acceptance_probs[nid],
                "target_entropy":  ents[nid],
            }
            csv_writer.writerow(row)   # ← written immediately, one row at a time
            records.append(row)

        # Greedy next token from target
        out = target_model(input_ids=cur_ids, attention_mask=cur_mask, use_cache=False)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if nt.item() == target_tokenizer.eos_token_id:
            break

        cur_ids  = torch.cat([cur_ids, nt], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(1, 1, dtype=torch.long, device=DEVICE)], dim=1
        )
        gen += 1
        if cur_ids.shape[1] > 2048 + max_new_tokens:
            break

    return records


# ── 9. Run benchmark ──────────────────────────────────────────────────
raw_csv_path  = OUTDIR / f"raw_records_{TS}.csv"
jsonl_path    = OUTDIR / f"raw_records_{TS}.jsonl"

all_records: list[dict] = []
t0 = time.time()

# Open CSV once; write header then rows one-by-one inside measure_sample()
with (
    open(raw_csv_path,  "w", newline="", buffering=1) as csv_fh,   # line-buffered
    open(jsonl_path,    "w", buffering=1) as jsonl_fh,
):
    csv_writer = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS)
    csv_writer.writeheader()

    for i, sample in enumerate(tqdm(all_samples, desc="Benchmark")):
        max_nt = CFG["max_new_tokens"].get(sample.task_type, 256)
        try:
            recs = measure_sample(sample, csv_writer, max_new_tokens=max_nt)
            for r in recs:
                jsonl_fh.write(json.dumps(r) + "\n")   # also line-by-line JSONL
            all_records.extend(recs)
        except Exception as e:
            logger.warning(f"Error on {sample.id}: {e}")
            continue

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"[{i+1}/{len(all_samples)}] {len(all_records)} records, {elapsed:.0f}s"
            )

elapsed_total = time.time() - t0
logger.info(
    f"Done: {len(all_records)} records from {len(all_samples)} samples "
    f"in {elapsed_total:.0f}s"
)
logger.info(f"CSV  → {raw_csv_path}")
logger.info(f"JSONL→ {jsonl_path}")

# ── 10. Analysis & metrics ────────────────────────────────────────────
df = pd.DataFrame(all_records)
for c in _CSV_FIELDS[2:]:      # numeric columns
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

logger.info(f"DataFrame: {df.shape}")
logger.info(f"Tasks: {df['task_type'].value_counts().to_dict()}")
logger.info(f"Depth range: {df['tree_depth'].min()} – {df['tree_depth'].max()}")
logger.info(f"Mean α: {df['acceptance_prob'].mean():.4f}")

# α by depth × task
adt = (
    df.groupby(["tree_depth", "task_type"])["acceptance_prob"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
adt.columns = ["depth", "task_type", "alpha_mean", "alpha_std", "count"]

# α by depth × position
adp = (
    df.groupby(["tree_depth", "position_bin"])["acceptance_prob"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
adp.columns = ["depth", "position_bin", "alpha_mean", "alpha_std", "count"]

# Full 3-way
adtp = (
    df.groupby(["tree_depth", "task_type", "position_bin"])["acceptance_prob"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
adtp.columns = ["depth", "task_type", "position_bin", "alpha_mean", "alpha_std", "count"]

# Cumulative acceptance P(L >= d)
cum_rows: list[dict] = []
for task in adt["task_type"].unique():
    sub = adt[adt["task_type"] == task].sort_values("depth")
    cp = np.cumprod(sub["alpha_mean"].values)
    for d, c in zip(sub["depth"].values, cp):
        cum_rows.append({"task_type": task, "depth": int(d), "prob_chain_ge_d": c})
cum_df = pd.DataFrame(cum_rows)

# Expected accepted length
el_rows: list[dict] = []
for task in adt["task_type"].unique():
    sub = adt[adt["task_type"] == task].sort_values("depth")
    el = np.cumprod(sub["alpha_mean"].values).sum()
    el_rows.append({"task_type": task, "expected_length": el})
el_df = pd.DataFrame(el_rows)

# Summary per task
summ = (
    df.groupby("task_type")
    .agg(
        n_samples     =("sample_id",       "nunique"),
        n_nodes       =("acceptance_prob",  "count"),
        mean_alpha    =("acceptance_prob",  "mean"),
        std_alpha     =("acceptance_prob",  "std"),
        mean_entropy  =("target_entropy",   "mean"),
    )
    .reset_index()
)

# Write aggregate CSVs — line by line using DictWriter
def _write_csv_lineby(path: Path, df_out: pd.DataFrame) -> None:
    with open(path, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=list(df_out.columns))
        w.writeheader()
        for row in df_out.itertuples(index=False):
            w.writerow(row._asdict())

_write_csv_lineby(OUTDIR / f"alpha_by_depth_task_{TS}.csv",         adt)
_write_csv_lineby(OUTDIR / f"alpha_by_depth_position_{TS}.csv",     adp)
_write_csv_lineby(OUTDIR / f"alpha_depth_task_position_{TS}.csv",   adtp)
_write_csv_lineby(OUTDIR / f"cumulative_acceptance_{TS}.csv",       cum_df)
_write_csv_lineby(OUTDIR / f"expected_length_{TS}.csv",             el_df)
_write_csv_lineby(OUTDIR / f"summary_by_task_{TS}.csv",             summ)
logger.info("Aggregate CSVs saved.")

print("\n── Expected Accepted Length ──")
print(el_df.to_string(index=False))
print("\n── Summary by Task ──")
print(summ.to_string(index=False))

# ── 11. Figures ───────────────────────────────────────────────────────
COLORS = {"code": "#2196F3", "math": "#FF5722", "chat": "#4CAF50", "reasoning": "#9C27B0"}
TASKS  = ["code", "math", "chat", "reasoning"]


def _savefig(name: str) -> None:
    path = OUTDIR / f"{name}_{TS}.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Figure saved → {path}")
    plt.close()


# Fig 1 – Depth–acceptance curves
fig, ax = plt.subplots(figsize=(8, 5))
for task in TASKS:
    sub = adt[adt["task_type"] == task].sort_values("depth")
    if sub.empty:
        continue
    ax.plot(sub["depth"], sub["alpha_mean"], "o-", lw=2,
            label=task.capitalize(), color=COLORS[task])
    ax.fill_between(
        sub["depth"],
        sub["alpha_mean"] - sub["alpha_std"],
        sub["alpha_mean"] + sub["alpha_std"],
        alpha=0.15, color=COLORS[task],
    )
ax.set_xlabel("Tree Depth $d$")
ax.set_ylabel(r"Mean Acceptance Rate $\alpha(d)$")
ax.set_title("Token Acceptance Rate vs. Speculation Depth")
ax.legend(title="Task"); ax.set_ylim(0, 1.05)
plt.tight_layout(); _savefig("fig1_depth_acceptance")

# Fig 2 – Position heatmaps
present = [t for t in TASKS if t in adtp["task_type"].unique()]
fig, axes = plt.subplots(
    1, len(present), figsize=(3 * len(present) + 2, 4), sharey=True, squeeze=False
)
for idx, task in enumerate(present):
    ax = axes[0, idx]
    sub = adtp[adtp["task_type"] == task]
    piv = sub.pivot_table(
        index="depth", columns="position_bin", values="alpha_mean", aggfunc="mean"
    )
    sns.heatmap(
        piv, ax=ax, cmap="YlOrRd_r", vmin=0, vmax=1,
        cbar=(idx == len(present) - 1),
        cbar_kws={"label": r"$\alpha$"} if idx == len(present) - 1 else {},
    )
    ax.set_title(task.capitalize())
    ax.set_xlabel("Position Bin")
    ax.set_ylabel("Depth $d$" if idx == 0 else "")
    ax.invert_yaxis()
fig.suptitle("Acceptance Rate by Depth & Token Position", y=1.02)
plt.tight_layout(); _savefig("fig2_position_heatmaps")

# Fig 3 – Expected accepted length
fig, ax = plt.subplots(figsize=(6, 4))
t_ord = [t for t in TASKS if t in el_df["task_type"].values]
vals  = [el_df[el_df["task_type"] == t]["expected_length"].values[0] for t in t_ord]
bars  = ax.bar(
    [t.capitalize() for t in t_ord], vals,
    color=[COLORS[t] for t in t_ord], edgecolor="black", lw=0.5,
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
            f"{v:.2f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("$E[L]$")
ax.set_title("Expected Speculative Chain Length by Task")
plt.tight_layout(); _savefig("fig3_expected_length")

# Fig 4 – Cumulative acceptance CDF
fig, ax = plt.subplots(figsize=(8, 5))
for task in TASKS:
    sub = cum_df[cum_df["task_type"] == task].sort_values("depth")
    if sub.empty:
        continue
    ax.plot(sub["depth"], sub["prob_chain_ge_d"], "s-", lw=2,
            label=task.capitalize(), color=COLORS[task])
ax.set_xlabel("Depth $d$")
ax.set_ylabel(r"$P(L \geq d)$")
ax.set_title(r"Cumulative Acceptance: Chain Reaching Depth $d$")
ax.legend(title="Task"); ax.set_ylim(0, 1.05)
plt.tight_layout(); _savefig("fig4_cumulative_acceptance")

# Fig 5 – Draft calibration
cal = df.dropna(subset=["draft_prob", "acceptance_prob"]).copy()
cal["draft_bin"] = pd.cut(cal["draft_prob"], bins=20)
cal_agg = cal.groupby("draft_bin")["acceptance_prob"].agg(["mean", "count"]).reset_index()
cal_agg["mid"] = cal_agg["draft_bin"].apply(
    lambda x: x.mid if hasattr(x, "mid") else np.nan
)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    cal_agg["mid"], cal_agg["mean"],
    s=cal_agg["count"] / cal_agg["count"].max() * 200,
    alpha=0.7, edgecolor="black", lw=0.3, color="#2196F3",
)
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Draft Model Confidence")
ax.set_ylabel("Actual Acceptance Rate")
ax.set_title("Draft Model Calibration\n(validates EAGLE-2 insight)")
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout(); _savefig("fig5_draft_calibration")

# Fig 6 – Entropy vs acceptance
ent_df = df.dropna(subset=["target_entropy", "acceptance_prob"]).copy()
ent_df["ent_bin"] = pd.cut(ent_df["target_entropy"], bins=20)
ent_agg = (
    ent_df.groupby(["ent_bin", "task_type"])["acceptance_prob"]
    .mean()
    .reset_index()
)
ent_agg.columns = ["ent_bin", "task_type", "alpha_mean"]
ent_agg["ent_mid"] = ent_agg["ent_bin"].apply(
    lambda x: x.mid if hasattr(x, "mid") else np.nan
)
fig, ax = plt.subplots(figsize=(8, 5))
for task in TASKS:
    sub = ent_agg[ent_agg["task_type"] == task].dropna(subset=["ent_mid"])
    if sub.empty:
        continue
    ax.plot(sub["ent_mid"], sub["alpha_mean"], "o-", lw=1.5,
            label=task.capitalize(), color=COLORS[task])
ax.set_xlabel("Target Distribution Entropy (nats)")
ax.set_ylabel(r"Mean Acceptance Rate $\alpha$")
ax.set_title("Acceptance Rate vs. Target Entropy")
ax.legend(title="Task"); ax.set_ylim(0, 1.05)
plt.tight_layout(); _savefig("fig6_entropy_vs_acceptance")

# ── 12. Key findings ──────────────────────────────────────────────────
print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print("\n1. DEPTH-ACCEPTANCE DECAY:")
for task in TASKS:
    sub = adt[adt["task_type"] == task].sort_values("depth")
    if sub.empty:
        continue
    a = sub["alpha_mean"].values
    print(f"   {task:10s}: d1={a[0]:.3f}  d{len(a)}={a[-1]:.3f}  decay={a[0]-a[-1]:.3f}")

print("\n2. EXPECTED ACCEPTED LENGTH:")
for _, row in el_df.iterrows():
    print(f"   {row['task_type']:10s}: E[L]={row['expected_length']:.2f}")

print("\n3. MEAN ACCEPTANCE BY TASK:")
for task in TASKS:
    sub = df[df["task_type"] == task]
    if sub.empty:
        continue
    print(f"   {task:10s}: α={sub['acceptance_prob'].mean():.4f} ± {sub['acceptance_prob'].std():.4f}")

print("\n4. ENTROPY-ACCEPTANCE CORRELATION:")
for task in TASKS:
    sub = df[df["task_type"] == task].dropna(subset=["target_entropy", "acceptance_prob"])
    if len(sub) < 10:
        continue
    corr = sub["target_entropy"].corr(sub["acceptance_prob"])
    print(f"   {task:10s}: ρ={corr:.4f}")

print("\n" + "=" * 60)
logger.info("Run complete.")
