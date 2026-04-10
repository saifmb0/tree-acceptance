"""Load target / draft model pairs for speculative decoding measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
)


@dataclass
class ModelPair:
    """Holds a target model, a draft model, and a shared tokeniser."""

    target_model: AutoModelForCausalLM
    draft_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    name: str
    device: torch.device


def _load_single(
    model_id: str,
    device: torch.device,
    quantisation: Optional[str] = None,
    revision: str = "main",
    max_memory_mb: int = 8000,
    trust_remote_code: bool = True,
) -> AutoModelForCausalLM:
    """Load a single causal-LM, optionally GPTQ-quantised."""

    kwargs: dict = dict(
        pretrained_model_name_or_path=model_id,
        revision=revision,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,
    )

    if quantisation and "gptq" in quantisation.lower():
        # For GPTQ models, just load directly - they're pre-quantized
        logger.info(f"Loading GPTQ model: {model_id}")
    else:
        logger.info(f"Loading model (fp16): {model_id}")

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.eval()
    return model


def load_model_pair(pair_cfg: dict, device: torch.device) -> ModelPair:
    """Load a (target, draft) model pair from config dict.

    Parameters
    ----------
    pair_cfg : dict
        Must have keys ``name``, ``target`` (with ``model_id``, etc.),
        and ``draft``.
    device : torch.device

    Returns
    -------
    ModelPair
    """

    tcfg = pair_cfg["target"]
    dcfg = pair_cfg["draft"]

    logger.info(f"Loading model pair: {pair_cfg['name']}")

    # Use the target model's tokenizer (draft must be compatible)
    tokenizer = AutoTokenizer.from_pretrained(
        tcfg["model_id"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = _load_single(
        model_id=tcfg["model_id"],
        device=device,
        quantisation=tcfg.get("quantisation"),
        revision=tcfg.get("revision", "main"),
        max_memory_mb=tcfg.get("max_memory_mb", 8000),
    )

    draft = _load_single(
        model_id=dcfg["model_id"],
        device=device,
        quantisation=dcfg.get("quantisation"),
        revision=dcfg.get("revision", "main"),
        max_memory_mb=dcfg.get("max_memory_mb", 4000),
    )

    logger.success(
        f"Loaded pair '{pair_cfg['name']}': "
        f"target={tcfg['model_id']}, draft={dcfg['model_id']}"
    )

    return ModelPair(
        target_model=target,
        draft_model=draft,
        tokenizer=tokenizer,
        name=pair_cfg["name"],
        device=device,
    )
