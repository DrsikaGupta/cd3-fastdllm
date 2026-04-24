"""
Generation routines for LLaDA / Fast-dLLM-style masked-diffusion language models.

Three generation variants are provided, in ascending order of speed:

1. :func:`generate`
   Vanilla iterative denoising – no KV cache. Runs a full forward pass over
   the entire ``[prompt | generation]`` sequence every step. Supports both
   confidence-only (baseline) and CD³ gating.

2. :func:`generate_with_prefix_cache`
   Runs the model on the prompt *once* to obtain a KV cache, then re-uses it
   for every subsequent denoising step. The generation region is run freshly
   each step. Requires the underlying model to support causal / prefix
   caching (``use_cache=True``).

3. :func:`generate_with_dual_cache`  *(main KV-cache path)*
   Divides the generation region into equal-sized *blocks* processed
   left-to-right.  For each denoising step it:
   - re-uses the cached prompt KV (prefix cache), and
   - re-uses cached KV for any preceding blocks (block cache).
   Only the *current* block's KV is computed fresh.
   CD³ state is maintained per block across steps; converged blocks are
   skipped entirely once all their tokens are locked in.

Note on bidirectional models:
   LLaDA-style masked-diffusion LLMs use *bidirectional* self-attention.
   Standard HuggingFace ``past_key_values`` relies on causal masking.
   When running a strictly bidirectional model you should set
   ``use_cache=False`` and use :func:`generate` / a custom attention mask.
   The ``_with_*_cache`` variants are most useful when the model is
   configured for left-to-right (causal or semi-causal) generation such as
   Dream or Fast-dLLM's block-causal mode.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .cd3 import CD3State, cd3_active_mask
from .confidence import (
    active_masked_confidence_scores,
    masked_confidence_scores,
    select_transfer_index,
    tokens_to_unmask,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _build_initial_sequence(
    prompt_ids: torch.Tensor,
    gen_len: int,
    mask_id: int,
) -> torch.Tensor:
    """Concatenate prompt tokens with a fully-masked generation region."""
    B = prompt_ids.shape[0]
    device = prompt_ids.device
    mask_region = torch.full((B, gen_len), mask_id, dtype=torch.long, device=device)
    return torch.cat([prompt_ids, mask_region], dim=1)


def _apply_transfer(
    x: torch.Tensor,
    logits: torch.Tensor,
    gen_offset: int,
    blk_start: int,
    blk_end: int,
    mask_id: int,
    is_masked_blk: torch.Tensor,
    active_mask: torch.Tensor,
    num_to_unmask: int,
) -> int:
    """Unmask the *num_to_unmask* most-confident active masked positions.

    Args:
        x:             Full sequence tensor ``[B, total_len]``, modified in-place.
        logits:        ``[B, blk_len, V]`` logits for the current block.
        gen_offset:    Start index of the generation region in *x*.
        blk_start:     Block start offset relative to the generation region.
        blk_end:       Block end offset (exclusive) relative to the gen region.
        mask_id:       Token id used for [MASK].
        is_masked_blk: ``[B, blk_len]`` bool – currently masked in this block.
        active_mask:   ``[B, blk_len]`` bool – CD³ active positions.
        num_to_unmask: Target number of positions to unmask (may be reduced if
                       fewer eligible positions exist).

    Returns:
        Actual number of positions unmasked.
    """
    if num_to_unmask <= 0 or not is_masked_blk.any():
        return 0

    scores = active_masked_confidence_scores(logits, is_masked_blk, active_mask)
    eligible = int((scores > float("-inf")).sum().item())
    if eligible == 0:
        return 0

    batch_idx, pos_idx = select_transfer_index(scores, num_to_unmask)
    predicted = logits.argmax(dim=-1)  # [B, blk_len]
    x[batch_idx, gen_offset + blk_start + pos_idx] = predicted[batch_idx, pos_idx]
    return int(pos_idx.numel())


# --------------------------------------------------------------------------- #
# 1. generate  (no cache)                                                     #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    gen_len: int,
    num_steps: int,
    mask_id: int,
    mode: str = "cd3",
    cd3_kwargs: Optional[Dict] = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Iterative masked-diffusion generation without any KV cache.

    A full forward pass over ``[prompt | generation_region]`` is performed at
    every denoising step.

    Args:
        model:       LLaDA-compatible model (``model(input_ids) → logits``).
        prompt_ids:  ``[B, prompt_len]`` tokenised prompt.
        gen_len:     Number of tokens to generate.
        num_steps:   Number of denoising steps.
        mask_id:     Token id of the ``[MASK]`` token.
        mode:        ``"cd3"`` or ``"confidence"`` (baseline).
        cd3_kwargs:  Keyword arguments forwarded to :func:`cd3_active_mask`
                     (``tau_kl``, ``tau_ent``, ``m``, ``k_min``).
        temperature: Sampling temperature; 0 → greedy argmax.
        verbose:     Log per-step statistics.

    Returns:
        gen_ids: ``[B, gen_len]`` generated token ids.
        info:    Dict with per-step statistics.
    """
    cd3_kwargs = cd3_kwargs or {}
    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    x = _build_initial_sequence(prompt_ids, gen_len, mask_id)
    state = CD3State()
    all_stats: List[Dict] = []

    for step in range(num_steps):
        is_masked = x[:, prompt_len:].eq(mask_id)  # [B, gen_len]
        if not is_masked.any():
            break

        # Forward pass
        out = model(x)
        logits = out.logits[:, prompt_len:, :]  # [B, gen_len, V]

        if temperature > 0:
            logits = logits / temperature

        # Gating
        if mode == "cd3":
            active_mask, state, stats = cd3_active_mask(
                logits, is_masked, state, **cd3_kwargs
            )
        else:
            active_mask = is_masked  # confidence baseline: all masked positions active
            stats = {"active_ratio": 1.0}

        num_to_unmask = tokens_to_unmask(is_masked, num_steps, step)

        _apply_transfer(
            x, logits,
            gen_offset=prompt_len, blk_start=0, blk_end=gen_len,
            mask_id=mask_id,
            is_masked_blk=is_masked,
            active_mask=active_mask,
            num_to_unmask=num_to_unmask,
        )

        if verbose:
            logger.info("step %d/%d %s", step + 1, num_steps, stats)
        all_stats.append(stats)

    return x[:, prompt_len:], {"steps": all_stats}


# --------------------------------------------------------------------------- #
# 2. generate_with_prefix_cache                                               #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_with_prefix_cache(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    gen_len: int,
    num_steps: int,
    mask_id: int,
    mode: str = "cd3",
    cd3_kwargs: Optional[Dict] = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Iterative generation with a *prefix* KV cache.

    The prompt tokens are run through the model *once* at the start to
    populate a KV cache.  Each subsequent denoising step only runs the model
    over the (changing) generation region, re-using the cached prompt KVs.

    Requires the model to support ``use_cache=True`` / ``past_key_values``.

    Args:  (same as :func:`generate`)

    Returns:
        gen_ids: ``[B, gen_len]`` generated token ids.
        info:    Dict with per-step statistics.
    """
    cd3_kwargs = cd3_kwargs or {}
    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    # Warm up the prefix cache
    prefix_out = model(prompt_ids, use_cache=True)
    prefix_kv = prefix_out.past_key_values

    gen_ids = torch.full((B, gen_len), mask_id, dtype=torch.long, device=device)
    state = CD3State()
    all_stats: List[Dict] = []

    for step in range(num_steps):
        is_masked = gen_ids.eq(mask_id)  # [B, gen_len]
        if not is_masked.any():
            break

        # Forward pass: only the generation region, with prefix cached
        out = model(gen_ids, past_key_values=prefix_kv, use_cache=False)
        logits = out.logits  # [B, gen_len, V]

        if temperature > 0:
            logits = logits / temperature

        # Gating
        if mode == "cd3":
            active_mask, state, stats = cd3_active_mask(
                logits, is_masked, state, **cd3_kwargs
            )
        else:
            active_mask = is_masked
            stats = {"active_ratio": 1.0}

        num_to_unmask = tokens_to_unmask(is_masked, num_steps, step)

        # Apply transfer to gen_ids (offset = 0 within gen_ids)
        if num_to_unmask > 0 and is_masked.any():
            scores = active_masked_confidence_scores(logits, is_masked, active_mask)
            eligible = int((scores > float("-inf")).sum().item())
            if eligible > 0:
                batch_idx, pos_idx = select_transfer_index(scores, num_to_unmask)
                predicted = logits.argmax(dim=-1)
                gen_ids[batch_idx, pos_idx] = predicted[batch_idx, pos_idx]

        if verbose:
            logger.info("step %d/%d %s", step + 1, num_steps, stats)
        all_stats.append(stats)

    return gen_ids, {"steps": all_stats}


# --------------------------------------------------------------------------- #
# 3. generate_with_dual_cache  (main KV-cache path)                          #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_with_dual_cache(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    gen_len: int,
    num_steps: int,
    mask_id: int,
    block_length: int = 32,
    mode: str = "cd3",
    cd3_kwargs: Optional[Dict] = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Block-wise generation with dual KV cache (prefix + block).

    The generation region is divided into equal-sized *blocks*.  At each
    denoising step, blocks are processed left-to-right.  For block *b*:

    - The *prefix cache* holds KV for all prompt tokens (computed once).
    - The *block cache* holds KV for blocks 0 … b-1 (extended each step).

    CD³ state is tracked *per block* across all denoising steps.  Once every
    token in a block has converged (active_mask all-False for non-masked
    positions), the block is skipped in subsequent steps.

    This is the primary high-speed path, equivalent to Fast-dLLM's
    ``generate_with_kvcache`` with optional CD³ gating.

    Args:
        model:        LLaDA-compatible model with ``use_cache`` support.
        prompt_ids:   ``[B, prompt_len]`` tokenised prompt.
        gen_len:      Number of tokens to generate.
        num_steps:    Number of denoising steps.
        mask_id:      Token id of the ``[MASK]`` token.
        block_length: Tokens per block (default 32).
        mode:         ``"cd3"`` or ``"confidence"``.
        cd3_kwargs:   Kwargs for :func:`cd3_active_mask`.
        temperature:  Sampling temperature.
        verbose:      Log per-step / per-block statistics.

    Returns:
        gen_ids: ``[B, gen_len]`` generated token ids.
        info:    Dict with per-step statistics and block-level detail.
    """
    cd3_kwargs = cd3_kwargs or {}
    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    # Pad gen_len to a multiple of block_length
    pad = (-gen_len) % block_length
    padded_gen_len = gen_len + pad

    num_blocks = padded_gen_len // block_length
    gen_ids = torch.full((B, padded_gen_len), mask_id, dtype=torch.long, device=device)

    # Per-block CD³ state, reset at the start of each generation call
    block_states: List[CD3State] = [CD3State() for _ in range(num_blocks)]

    # ------------------------------------------------------------------ #
    # Prefix cache: run the prompt once                                   #
    # ------------------------------------------------------------------ #
    prefix_out = model(prompt_ids, use_cache=True)
    prefix_kv = prefix_out.past_key_values

    all_step_stats: List[Dict] = []

    for step in range(num_steps):
        is_masked_all = gen_ids.eq(mask_id)  # [B, padded_gen_len]
        if not is_masked_all.any():
            break

        step_stats: Dict[str, object] = {"step": step, "blocks": []}

        # Current KV extending prefix for each block processed left-to-right
        running_kv = prefix_kv

        for b in range(num_blocks):
            blk_start = b * block_length
            blk_end = blk_start + block_length

            blk_ids = gen_ids[:, blk_start:blk_end]       # [B, block_length]
            is_masked_blk = is_masked_all[:, blk_start:blk_end]  # [B, block_length]

            # Skip fully unmasked blocks after the first pass
            if step > 0 and not is_masked_blk.any():
                # Still need to extend running_kv with the converged block
                blk_out = model(blk_ids, past_key_values=running_kv, use_cache=True)
                running_kv = blk_out.past_key_values
                continue

            # Forward pass for this block
            blk_out = model(blk_ids, past_key_values=running_kv, use_cache=True)
            logits = blk_out.logits  # [B, block_length, V]
            running_kv = blk_out.past_key_values  # extend for next block

            if temperature > 0:
                logits = logits / temperature

            # ---------------------------------------------------------- #
            # Gating                                                       #
            # ---------------------------------------------------------- #
            if mode == "cd3":
                active_mask, block_states[b], blk_stats = cd3_active_mask(
                    logits, is_masked_blk, block_states[b], **cd3_kwargs
                )
            else:
                active_mask = is_masked_blk
                blk_stats = {"active_ratio": 1.0}

            # ---------------------------------------------------------- #
            # Transfer: unmask top-confidence active-masked tokens        #
            # ---------------------------------------------------------- #
            num_to_unmask = tokens_to_unmask(is_masked_blk, num_steps, step)

            scores = active_masked_confidence_scores(logits, is_masked_blk, active_mask)
            eligible = int((scores > float("-inf")).sum().item())
            if eligible > 0 and num_to_unmask > 0:
                batch_idx, pos_idx = select_transfer_index(scores, num_to_unmask)
                predicted = logits.argmax(dim=-1)
                gen_ids[batch_idx, blk_start + pos_idx] = predicted[batch_idx, pos_idx]

            blk_stats["block"] = b
            blk_stats["num_unmasked"] = num_to_unmask
            cast_stats = step_stats["blocks"]
            assert isinstance(cast_stats, list)
            cast_stats.append(blk_stats)

            if verbose:
                logger.info("step %d block %d %s", step + 1, b, blk_stats)

        all_step_stats.append(step_stats)

    # Trim padding
    return gen_ids[:, :gen_len], {"steps": all_step_stats}


# --------------------------------------------------------------------------- #
# Convenience wrapper                                                          #
# --------------------------------------------------------------------------- #


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    gen_len: int = 256,
    num_steps: int = 128,
    mode: str = "cd3",
    variant: str = "dual_cache",
    block_length: int = 32,
    cd3_kwargs: Optional[Dict] = None,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[str, Dict]:
    """End-to-end text generation from a raw string prompt.

    Handles tokenisation, dispatches to the requested generation variant, and
    decodes the output back to a string.

    Args:
        model:        LLaDA-compatible model.
        tokenizer:    HuggingFace tokenizer with ``mask_token_id`` attribute.
        prompt:       Raw text prompt.
        gen_len:      Number of tokens to generate.
        num_steps:    Denoising steps.
        mode:         ``"cd3"`` or ``"confidence"``.
        variant:      One of ``"basic"``, ``"prefix_cache"``, ``"dual_cache"``.
        block_length: Block size for ``dual_cache`` variant.
        cd3_kwargs:   Passed to :func:`cd3_active_mask`.
        temperature:  Sampling temperature.
        device:       Device override; defaults to model's device.
        verbose:      Enable step-level logging.

    Returns:
        text:  Decoded generation (str).
        info:  Statistics dict from the generation routine.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mask_id: int = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            "Tokenizer does not have a mask_token_id. "
            "Ensure you are using a tokenizer for a masked-diffusion model "
            "(e.g. GSAI-ML/LLaDA-8B-Instruct)."
        )

    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"].to(device)

    kwargs: Dict = dict(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=gen_len,
        num_steps=num_steps,
        mask_id=mask_id,
        mode=mode,
        cd3_kwargs=cd3_kwargs or {},
        temperature=temperature,
        verbose=verbose,
    )

    if variant == "basic":
        gen_ids, info = generate(**kwargs)
    elif variant == "prefix_cache":
        gen_ids, info = generate_with_prefix_cache(**kwargs)
    elif variant == "dual_cache":
        gen_ids, info = generate_with_dual_cache(block_length=block_length, **kwargs)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: basic, prefix_cache, dual_cache.")

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text, info
