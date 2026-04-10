# Characterising Token Acceptance in Speculative Decoding

> **Gap 2**: First systematic study of the token acceptance rate α across tree
> depth, task type, and token position. Inference-only, no training required.

## Quick Start

### Local (with GPU)

```bash
pip install -r requirements.txt

# Run benchmark (one model pair, all datasets)
python scripts/run_benchmark.py

# Run on specific datasets with fewer samples
python scripts/run_benchmark.py --datasets humaneval gsm8k --max-samples 20

# Analyse results
python scripts/run_analysis.py results/raw/*_full.jsonl
```

### Kaggle T4×2

Upload and run `notebooks/kaggle_t4x2.ipynb` directly. It is self-contained
with all code, dataset loading, and figure generation in a single notebook.
Reduce `max_samples` in the config cell to fit within Kaggle's time limits.

## Project Structure

```
tree-acceptance/
├── NARRATIVE.md              ← Literature positioning & study design
├── README.md                 ← This file
├── requirements.txt
├── configs/
│   └── default.yaml          ← Hyperparameters & model pairs
├── src/
│   ├── models.py             ← Target/draft model loading
│   ├── datasets.py           ← HumanEval, MATH, ShareGPT, GSM8K loaders
│   ├── tree.py               ← Draft tree BFS construction & verification
│   ├── measure.py            ← Per-token measurement loop
│   ├── metrics.py            ← α(d), E[L], P(L≥d), entropy, calibration
│   └── utils.py              ← Seeding, I/O, config
├── analysis/
│   ├── aggregate.py          ← Raw JSONL → summary CSV tables
│   └── plot.py               ← 6 publication-quality figures
├── scripts/
│   ├── run_benchmark.py      ← CLI: run the benchmark
│   └── run_analysis.py       ← CLI: aggregate + plot
├── notebooks/
│   └── kaggle_t4x2.ipynb     ← Self-contained Kaggle notebook
└── results/                  ← Raw logs, tables, figures (gitignored)
```

## What We Measure

For each prompt × each autoregressive step:

1. **Build a draft tree** (BFS, top-k branching, configurable depth/width)
2. **Score every tree node** with the target model
3. **Record**: `(sample_id, task_type, token_position, tree_depth, draft_prob, target_prob, acceptance_prob, target_entropy)`

### Key Figures Produced

| # | Figure | Shows |
|---|--------|-------|
| 1 | **Depth–acceptance curves** | How α decays with depth, one line per task |
| 2 | **Position heatmaps** | α(depth, position) showing drift over generation |
| 3 | **Expected length bar chart** | E[L] per task |
| 4 | **Cumulative acceptance CDF** | P(L ≥ d) per task |
| 5 | **Draft calibration scatter** | Draft confidence vs. actual acceptance |
| 6 | **Entropy vs. acceptance** | Target entropy vs. α, coloured by task |

## Model Pairs

| Target | Draft | Notes |
|--------|-------|-------|
| Llama-2-7B-Chat (GPTQ 4-bit) | TinyLlama-1.1B-Chat | Fits T4 16GB |
| Qwen2.5-7B-Instruct (GPTQ 4-bit) | Qwen2.5-0.5B-Instruct | Modern pair |

## Datasets

| Dataset | Task Type | Typical Samples |
|---------|-----------|-----------------|
| HumanEval | Code | 164 |
| MATH | Mathematical reasoning | 200 |
| ShareGPT | Open-ended chat | 200 |
| GSM8K | Chain-of-thought reasoning | 200 |

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{tree-acceptance-2026,
  title={Characterising Token Acceptance in Speculative Decoding:
         A Systematic Study Across Task Type, Token Position, and Tree Depth},
  year={2026},
}
```
