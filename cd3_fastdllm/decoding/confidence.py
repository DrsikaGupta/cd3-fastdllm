"""
Confidence-based scoring utilities for diffusion decoding.

Used as the *baseline* gating strategy (Fast-dLLM / confidence-only) and as
the scoring function inside the CD³ pipeline to select *which* active tokens
to unmask at each denoising step.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Core confidence helpers                                                      #
# --------------------------------------------------------------------------- #


def confidence_scores(logits: torch.Tensor) -> torch.Tensor:
    """Max-softmax confidence score for each token position.

    Args:
        logits: ``[B, T, V]`` raw model logits.

    Returns:
        scores: ``[B, T]`` values in ``(0, 1]``.
    """
    return F.softmax(logits, dim=-1).max(dim=-1).values


def masked_confidence_scores(
    logits: torch.Tensor,
    is_masked: torch.Tensor,
) -> torch.Tensor:
    """Confidence scores at masked positions; ``-inf`` elsewhere.

    Args:
        logits:    ``[B, T, V]`` raw model logits.
        is_masked: ``[B, T]`` bool, True where the token is still masked.

    Returns:
        scores: ``[B, T]`` with ``-inf`` for already-unmasked positions.
    """
    scores = confidence_scores(logits)
    return scores.masked_fill(~is_masked, float("-inf"))


def active_masked_confidence_scores(
    logits: torch.Tensor,
    is_masked: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Confidence scores for positions that are *both* masked and active.

    Used by the CD³ pipeline to restrict the transfer candidates to the active
    subset determined by :func:`cd3_fastdllm.decoding.cd3.cd3_active_mask`.

    Args:
        logits:      ``[B, T, V]``
        is_masked:   ``[B, T]`` bool
        active_mask: ``[B, T]`` bool

    Returns:
        scores: ``[B, T]`` with ``-inf`` for positions that are either
                unmasked or inactive.
    """
    scores = confidence_scores(logits)
    eligible = is_masked & active_mask
    return scores.masked_fill(~eligible, float("-inf"))


# --------------------------------------------------------------------------- #
# Token-selection helper                                                       #
# --------------------------------------------------------------------------- #


def select_transfer_index(
    scores: torch.Tensor,
    num_transfer: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the top-*k* token positions by score (across all batch items).

    Positions with ``-inf`` score are never selected.  The actual number of
    selected positions is ``min(num_transfer, num_eligible)``.

    Args:
        scores:       ``[B, T]`` scores; ``-inf`` marks ineligible positions.
        num_transfer: number of positions to select *per batch item*.

    Returns:
        batch_idx: ``[N]`` batch indices of selected positions.
        pos_idx:   ``[N]`` position indices of selected positions.
    """
    B, T = scores.shape

    # Clamp num_transfer to the minimum number of eligible positions across
    # the batch so that topk never receives k=0 or k > T.
    eligible_counts = (scores > float("-inf")).sum(dim=-1)  # [B]
    k = max(1, min(num_transfer, int(eligible_counts.min().item())))
    k = min(k, T)

    topk = scores.topk(k=k, dim=-1)  # values, indices: [B, k]
    batch_idx = (
        torch.arange(B, device=scores.device)
        .unsqueeze(1)
        .expand_as(topk.indices)
        .flatten()
    )
    pos_idx = topk.indices.flatten()
    return batch_idx, pos_idx


# --------------------------------------------------------------------------- #
# Noise schedule helpers                                                       #
# --------------------------------------------------------------------------- #


def linear_unmask_schedule(
    gen_len: int,
    num_steps: int,
    step: int,
) -> int:
    """Number of tokens to unmask at denoising *step* (0-indexed).

    Uses a uniform linear schedule so that all tokens are unmasked after
    ``num_steps`` steps.  The last step unmasks all remaining tokens.

    Args:
        gen_len:   Total number of tokens to generate.
        num_steps: Total number of denoising steps.
        step:      Current step index (0-indexed).

    Returns:
        Number of tokens to unmask at this step (≥1).
    """
    if num_steps <= 1:
        return gen_len
    base = gen_len // num_steps
    # Distribute remainder across the first steps
    remainder = gen_len % num_steps
    return base + (1 if step < remainder else 0)


def tokens_to_unmask(
    is_masked: torch.Tensor,
    num_steps: int,
    step: int,
) -> int:
    """Adaptive schedule: unmask to reach target fraction of masked tokens.

    Targets a linear ramp so that by step ``num_steps - 1``, all tokens are
    unmasked.  More robust than a purely fixed schedule when some tokens were
    already unmasked before the loop started.

    Args:
        is_masked: ``[B, T]`` bool current mask.
        num_steps: Total denoising steps.
        step:      Current step index (0-indexed).

    Returns:
        Number of tokens to unmask (≥1, ≤ number currently masked).
    """
    total_masked = int(is_masked.sum().item())
    if total_masked == 0:
        return 0
    steps_left = num_steps - step
    return max(1, -(-total_masked // steps_left))  # ceil division
