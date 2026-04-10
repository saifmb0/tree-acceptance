# Characterising Token Acceptance in Speculative Decoding: A Systematic Study Across Task Type, Token Position, and Tree Depth

## The Gap

Every speculative-decoding paper reports the **token acceptance rate** α as a
validation metric — yet no paper makes α *characterisation* the **primary
contribution**. EAGLE-2, Sequoia, Medusa, and others all use α implicitly (to
justify speedup claims, to calibrate dynamic trees, to select optimal tree
topologies) but never systematically study:

| Question | Why it matters |
|---|---|
| **How does α decay with tree depth?** | Determines the *optimal speculation budget*: past a certain depth every extra token is wasted compute. Sequoia's DP tree-optimizer *assumes* a depth–acceptance curve but never publishes one. |
| **How does α vary across task types (code / math / chat / reasoning)?** | Code has rigid syntax → high α on boilerplate, low α on identifiers. Math has formula rails. Chat is entropy-rich. Different tasks need different tree shapes. |
| **How does α evolve over token position in a long generation?** | Early tokens condition on the prompt (low entropy if prompt is specific); mid-generation may be high-entropy; endings may tighten again. If α drifts, a *static* tree is sub-optimal everywhere. |

### Positioning in the Literature

| Paper | What they do with α | What they do NOT do |
|---|---|---|
| **Leviathan et al. (2022)** — *Fast Inference from Transformers via Speculative Decoding* | Define α theoretically; show 2–3× speedup on T5-XXL | No depth analysis, no task breakdown |
| **Chen et al. (2023)** — *Accelerating LLM Decoding with Speculative Sampling* | Use modified rejection sampling to preserve target distribution; report 2–2.5× on Chinchilla 70B | α is a by-product, never analyzed independently |
| **SpecInfer (Miao et al., 2023)** — *Tree-based Speculative Inference & Verification* | Token tree with multi-drafter ensemble; 1.5–3.5× | Report aggregate speedup; no per-depth α curves |
| **Staged Speculative Decoding (Spector & Ré, 2023)** | Tree-structured batches, two-stage speculation | Mention tree depth helps; never measure α vs. depth |
| **Draft & Verify (Zhang et al., 2023)** — *Self-Speculative Decoding* | Layer-skipping; no separate drafter | Report per-dataset speedup; no α decomposition |
| **Medusa (Cai et al., 2024)** | Multiple decoding heads + tree attention; typical acceptance scheme to *boost* α | Show top-k acceptance per head; no depth × task × position analysis |
| **EAGLE (Li et al., 2024)** | Feature-level autoregression for drafting; 2.7–3.5× on LLaMA2-Chat 70B | α is implicit in the tree construction; not studied independently |
| **Sequoia (Chen et al., 2024)** | DP for optimal tree topology; hardware-aware | *Assumes* a depth–acceptance curve to feed the DP; never publishes the raw curves per task |
| **EAGLE-2 (Li et al., 2024)** | Context-aware dynamic draft trees using calibrated confidence scores ≈ α | Key insight: α is context-dependent. But the paper uses this to build better trees, not to *study* α itself |
| **Survey (Xia et al., 2024)** — *Unlocking Efficiency in LLM Inference* | Comprehensive taxonomy of speculative decoding methods | Calls for more empirical analysis of acceptance rates — our paper answers this call |

### Our Contribution

We present the **first systematic empirical study** of the token acceptance rate
α as a function of three axes:

1. **Tree depth** (d = 1 … D): We measure the *conditional* acceptance
   probability P(accept at depth d | accepted at depth d−1) and the
   *cumulative* acceptance probability P(accepted chain length ≥ d).

2. **Task type**: Code generation (HumanEval), mathematical reasoning (MATH),
   open-ended chat (ShareGPT), and chain-of-thought reasoning traces (GSM8K).

3. **Token position**: We bin the output sequence into position windows and
   track how α evolves from the first generated token to the last.

**This is inference-only** — no model training. The study requires only a target
LLM, a draft LLM (or draft heads), and a few GPU-hours on commodity hardware
(Kaggle T4×2).

The results provide:
- **Empirical depth–acceptance curves** that can directly replace Sequoia's
  assumed curves and improve any DP-based tree optimiser.
- **Per-task tree-shape recommendations**: e.g., code may favour deeper, narrower
  trees (high α at depth) while chat may need wider, shallower trees.
- **Position-aware scheduling insights**: if α drops mid-generation, an adaptive
  system should shrink its speculation budget dynamically.
- **A reusable benchmark** that any future speculative-decoding paper can use to
  contextualise their α improvements.

---

## Experimental Design

### Hardware Target
- **Kaggle T4×2**: 2× NVIDIA T4 (16 GB each), ~65 TFLOPS FP16 each
- Models must fit in ≤ 16 GB per GPU (one GPU for target, one for draft; or
  both on one GPU with 4-bit quantization)

### Model Pairs

| Target Model | Draft Model | Notes |
|---|---|---|
| Llama-2-7B-Chat (4-bit GPTQ) | TinyLlama-1.1B-Chat | Classic small-target/tiny-draft pair; fits T4 |
| Qwen2.5-7B-Instruct (4-bit) | Qwen2.5-0.5B-Instruct | Modern pair with strong instruction-following |

### Datasets

| Dataset | Task Type | # Samples | Typical Output Length |
|---|---|---|---|
| **HumanEval** | Code generation | 164 | 50–200 tokens |
| **MATH** (Level 3–5 subset) | Mathematical reasoning | 200 | 100–500 tokens |
| **ShareGPT** (sampled) | Open-ended chat | 200 | 100–800 tokens |
| **GSM8K** | Chain-of-thought reasoning | 200 | 100–400 tokens |

### Measurement Protocol

For each (prompt, model_pair):

1. Run **greedy autoregressive decoding** with the target model to get the
   reference output.

2. For each token position t in the reference output:
   a. Use the draft model to generate a **draft tree** of depth D (we use
      D = 6, branching factor k = 5, giving trees up to ~3900 nodes — but
      we can prune to budget).
   b. Record the **target-model logits** for every node in the tree.
   c. For each node at depth d, compute:
      - `p_target[token]` — target probability of the drafted token
      - `p_draft[token]` — draft probability of the drafted token
      - `accepted = (p_target[token] / p_draft[token] >= u)` where u ~ Uniform(0,1)
        (standard rejection sampling; but for measurement purposes we record
        the *acceptance probability* `min(1, p_target/p_draft)` directly)
   d. Log: (sample_id, task_type, token_position_t, tree_depth_d,
            branch_id, draft_prob, target_prob, acceptance_prob)

3. Aggregate into the three analysis axes.

### Key Metrics

- **α(d)**: Mean acceptance probability at tree depth d, marginalised over
  tasks and positions.
- **α(d, task)**: Same, conditioned on task type.
- **α(d, t_bin)**: Same, conditioned on token position bin.
- **α(d, task, t_bin)**: Full 3-way breakdown.
- **Expected accepted length E[L]**: Expected number of consecutive accepted
  tokens from root.
- **Cumulative acceptance P(L ≥ d)**: Probability the accepted chain reaches
  depth d.

### Plots to Produce

1. **Depth–acceptance curves** (one line per task): x = depth, y = α(d, task).
   The key figure of the paper.

2. **Position heatmaps** (one per task): x = token position bin, y = depth,
   color = α. Shows how the depth profile shifts over the generation.

3. **Expected accepted length by task**: Bar chart comparing E[L] across tasks.

4. **Cumulative acceptance CDF**: x = depth, y = P(L ≥ d), one line per task.

5. **Draft model calibration**: scatter of draft confidence vs. actual
   acceptance rate, validating/extending EAGLE-2's calibration insight.

6. **Entropy vs. acceptance**: per-token target entropy plotted against
   per-token acceptance probability, coloured by task.

---

## Implementation Plan

```
tree-acceptance/
├── NARRATIVE.md              ← this file
├── README.md                 ← quickstart
├── requirements.txt
├── configs/
│   └── default.yaml          ← all hyperparameters
├── src/
│   ├── __init__.py
│   ├── models.py             ← load target/draft model pairs
│   ├── datasets.py           ← load & preprocess datasets
│   ├── tree.py               ← draft tree construction & verification
│   ├── measure.py            ← core measurement loop
│   ├── metrics.py            ← compute α, E[L], P(L≥d), entropy
│   └── utils.py              ← logging, seeding, I/O
├── analysis/
│   ├── __init__.py
│   ├── aggregate.py          ← aggregate raw logs → summary tables
│   └── plot.py               ← produce all figures
├── scripts/
│   ├── run_benchmark.py      ← CLI entry point
│   └── run_analysis.py       ← CLI for post-hoc analysis
├── notebooks/
│   └── kaggle_t4x2.ipynb     ← self-contained Kaggle notebook
└── results/                  ← gitignored; raw logs & figures
```
