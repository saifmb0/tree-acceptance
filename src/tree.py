"""Draft tree construction and speculative verification.

The tree is built greedily by the draft model, then every node is scored
by the target model in a *single* batched forward pass using tree attention
(causal mask that follows the tree topology).

Key data structure
------------------
Each node in the tree is stored in flat arrays indexed by ``node_id``.
``parent[i]`` gives the parent of node i (-1 for the root).
``depth[i]``  gives the depth (root = 0).
``token[i]``  gives the token id drafted at node i.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger


# ── Tree data structure ──────────────────────────────────────────────

@dataclass
class DraftTree:
    """Flat representation of a draft token tree."""

    tokens: list[int] = field(default_factory=list)
    parent: list[int] = field(default_factory=list)
    depth: list[int] = field(default_factory=list)
    draft_logprobs: list[float] = field(default_factory=list)  # log P_draft(token)
    draft_probs: list[float] = field(default_factory=list)     # P_draft(token)

    # Filled after target scoring
    target_logprobs: list[float] = field(default_factory=list)
    target_probs: list[float] = field(default_factory=list)
    acceptance_probs: list[float] = field(default_factory=list)  # min(1, p_t/p_d)

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def max_depth(self) -> int:
        return max(self.depth) if self.depth else 0


# ── Tree construction (draft model) ─────────────────────────────────

@torch.no_grad()
def build_draft_tree(
    draft_model,
    input_ids: torch.Tensor,        # (1, seq_len) — prompt tokens
    attention_mask: torch.Tensor,    # (1, seq_len)
    max_depth: int = 6,
    top_k: int = 10,
    max_branch: int = 5,
    max_nodes: int = 64,
    temperature: float = 0.0,
) -> DraftTree:
    """Build a draft tree by expanding top-k children at each node BFS.

    Parameters
    ----------
    draft_model : CausalLM
        The small draft language model.
    input_ids : Tensor (1, prefix_len)
        Tokenised prompt (already on device).
    attention_mask : Tensor (1, prefix_len)
        Attention mask for the prompt.
    max_depth, top_k, max_branch, max_nodes :
        Tree shape constraints.
    temperature : float
        0 = greedy (use raw logits); >0 = sample.

    Returns
    -------
    DraftTree with tokens, parent, depth, draft_probs filled in.
    """

    tree = DraftTree()
    device = input_ids.device
    prefix_len = input_ids.shape[1]

    # Use KV-cache for efficient multi-step generation
    past_key_values = None

    # ── Seed: run the prompt through the draft model ─────────────
    outputs = draft_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # (1, vocab)

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)

    # Root children: top-k from the first draft position
    topk_probs, topk_ids = probs.topk(min(top_k, max_branch), dim=-1)  # (1, k)
    topk_probs = topk_probs.squeeze(0)  # (k,)
    topk_ids = topk_ids.squeeze(0)

    # BFS frontier: list of (node_id, kv_cache_for_this_path)
    # For memory efficiency on T4, we DON'T clone KV caches per branch.
    # Instead we re-run the draft from scratch for each path.  This is
    # acceptable because the draft model is tiny and max_nodes ≤ 64.

    # Add root-level children
    for j in range(topk_ids.shape[0]):
        if tree.size >= max_nodes:
            break
        tree.tokens.append(topk_ids[j].item())
        tree.parent.append(-1)  # root children have no tree-parent
        tree.depth.append(1)
        tree.draft_probs.append(topk_probs[j].item())
        tree.draft_logprobs.append(topk_probs[j].log().item())

    # ── BFS expansion ────────────────────────────────────────────
    frontier = list(range(tree.size))  # node ids to expand

    while frontier and tree.size < max_nodes:
        next_frontier = []
        for node_id in frontier:
            if tree.size >= max_nodes:
                break
            if tree.depth[node_id] >= max_depth:
                continue

            # Reconstruct the token path from root to this node
            path_tokens = _path_to_root(tree, node_id)

            # Build full input: prompt + path tokens
            path_tensor = torch.tensor(
                [path_tokens], dtype=torch.long, device=device
            )
            full_input = torch.cat([input_ids, path_tensor], dim=1)
            full_mask = torch.ones(
                1, full_input.shape[1], dtype=torch.long, device=device
            )

            # Forward pass through draft model for this path
            out = draft_model(
                input_ids=full_input,
                attention_mask=full_mask,
                use_cache=False,
            )
            logits_node = out.logits[:, -1, :]

            if temperature > 0:
                p = F.softmax(logits_node / temperature, dim=-1)
            else:
                p = F.softmax(logits_node, dim=-1)

            tk_probs, tk_ids = p.topk(
                min(top_k, max_branch), dim=-1
            )
            tk_probs = tk_probs.squeeze(0)
            tk_ids = tk_ids.squeeze(0)

            for j in range(tk_ids.shape[0]):
                if tree.size >= max_nodes:
                    break
                child_id = tree.size
                tree.tokens.append(tk_ids[j].item())
                tree.parent.append(node_id)
                tree.depth.append(tree.depth[node_id] + 1)
                tree.draft_probs.append(tk_probs[j].item())
                tree.draft_logprobs.append(tk_probs[j].log().item())
                next_frontier.append(child_id)

        frontier = next_frontier

    return tree


def _path_to_root(tree: DraftTree, node_id: int) -> list[int]:
    """Return token path from the root to ``node_id`` (inclusive)."""
    path = []
    nid = node_id
    while nid >= 0:
        path.append(tree.tokens[nid])
        nid = tree.parent[nid]
    path.reverse()
    return path


# ── Tree verification (target model) ────────────────────────────────

@torch.no_grad()
def verify_tree(
    target_model,
    input_ids: torch.Tensor,     # (1, prefix_len)
    attention_mask: torch.Tensor,
    tree: DraftTree,
) -> DraftTree:
    """Score every node in the draft tree using the target model.

    For each node we record:
    - target_probs[i]: P_target(tree.tokens[i] | prefix, path_to_parent)
    - acceptance_probs[i]: min(1, target_probs[i] / draft_probs[i])

    We verify each root-to-leaf path by constructing the full sequence
    and running the target model.  For trees ≤64 nodes this is feasible
    on T4.

    Parameters
    ----------
    target_model : CausalLM
    input_ids, attention_mask : prompt tensors
    tree : DraftTree (must already have tokens/parent/depth/draft_probs)

    Returns
    -------
    The same DraftTree with target_probs and acceptance_probs filled.
    """

    device = input_ids.device
    n = tree.size
    tree.target_probs = [0.0] * n
    tree.target_logprobs = [0.0] * n
    tree.acceptance_probs = [0.0] * n

    # Find all leaf nodes (nodes that are not parents of anyone)
    children_of = set(tree.parent)
    leaves = [i for i in range(n) if i not in children_of]

    # For each leaf, get the full root→leaf path and score it
    scored = set()  # node ids already scored
    for leaf in leaves:
        path_ids = []
        nid = leaf
        while nid >= 0:
            path_ids.append(nid)
            nid = tree.parent[nid]
        path_ids.reverse()  # root → leaf order

        # Build the full input: prompt + path tokens
        path_tokens = [tree.tokens[nid] for nid in path_ids]
        path_tensor = torch.tensor(
            [path_tokens], dtype=torch.long, device=device
        )
        # We only need logits up to the second-to-last token in path
        # to score each token using teacher forcing
        full_input = torch.cat([input_ids, path_tensor], dim=1)
        full_mask = torch.ones(
            1, full_input.shape[1], dtype=torch.long, device=device
        )

        outputs = target_model(
            input_ids=full_input,
            attention_mask=full_mask,
            use_cache=False,
        )
        logits = outputs.logits  # (1, total_len, vocab)

        prefix_len = input_ids.shape[1]
        for idx_in_path, node_id in enumerate(path_ids):
            if node_id in scored:
                continue
            # The logit that predicts this token is at position
            # (prefix_len - 1 + idx_in_path), because the logit at
            # position j predicts token j+1.
            logit_pos = prefix_len - 1 + idx_in_path
            logit_vec = logits[0, logit_pos, :]  # (vocab,)
            probs_vec = F.softmax(logit_vec, dim=-1)

            tok = tree.tokens[node_id]
            p_target = probs_vec[tok].item()
            p_draft = max(tree.draft_probs[node_id], 1e-10)

            tree.target_probs[node_id] = p_target
            tree.target_logprobs[node_id] = (
                probs_vec[tok].log().item() if p_target > 0 else -float("inf")
            )
            tree.acceptance_probs[node_id] = min(1.0, p_target / p_draft)
            scored.add(node_id)

    return tree


# ── Utility: compute per-node entropy of target distribution ────────

@torch.no_grad()
def compute_target_entropy(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tree: DraftTree,
) -> list[float]:
    """Compute the entropy of the target distribution at each node.

    Returns list of entropy values (nats), one per tree node.
    This reuses the path-based approach from verify_tree.
    """

    device = input_ids.device
    n = tree.size
    entropies = [0.0] * n

    children_of = set(tree.parent)
    leaves = [i for i in range(n) if i not in children_of]
    computed = set()

    for leaf in leaves:
        path_ids = []
        nid = leaf
        while nid >= 0:
            path_ids.append(nid)
            nid = tree.parent[nid]
        path_ids.reverse()

        path_tokens = [tree.tokens[nid] for nid in path_ids]
        path_tensor = torch.tensor(
            [path_tokens], dtype=torch.long, device=device
        )
        full_input = torch.cat([input_ids, path_tensor], dim=1)
        full_mask = torch.ones(
            1, full_input.shape[1], dtype=torch.long, device=device
        )

        outputs = target_model(
            input_ids=full_input,
            attention_mask=full_mask,
            use_cache=False,
        )
        logits = outputs.logits

        prefix_len = input_ids.shape[1]
        for idx_in_path, node_id in enumerate(path_ids):
            if node_id in computed:
                continue
            logit_pos = prefix_len - 1 + idx_in_path
            logit_vec = logits[0, logit_pos, :]
            probs_vec = F.softmax(logit_vec, dim=-1)
            log_probs = F.log_softmax(logit_vec, dim=-1)
            entropy = -(probs_vec * log_probs).sum().item()
            entropies[node_id] = entropy
            computed.add(node_id)

    return entropies
