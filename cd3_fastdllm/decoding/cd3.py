"""
CD³ (Convergence-Driven Diffusion Decoding) gating.

Implements rule-based convergence detection using:
  - KL(p_s || p_{s-1}): distribution shift between consecutive denoising steps
  - Entropy H(p_s): uncertainty of current prediction
  - Flip: whether argmax changed from previous step
  - Hysteresis: require M consecutive stable steps before locking
  - K_min: minimum active tokens to maintain parallel throughput
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CD3State:
    """Tracks per-position convergence state across denoising steps.

    One instance is maintained per block and reset between different generation
    calls (but persisted across denoising steps within a single generation).

    Attributes:
        prev_logp:     [B, T, V] log-probabilities from the previous step.
        prev_argmax:   [B, T]    argmax token from the previous step.
        stable_count:  [B, T]    number of consecutive steps where position was
                                 deemed stable (used for hysteresis).
    """

    prev_logp: Optional[torch.Tensor] = field(default=None, repr=False)
    prev_argmax: Optional[torch.Tensor] = field(default=None, repr=False)
    stable_count: Optional[torch.Tensor] = field(default=None, repr=False)

    def reset(self) -> None:
        """Clear state so the next call acts as if it is the first step."""
        self.prev_logp = None
        self.prev_argmax = None
        self.stable_count = None

    def is_initialized(self) -> bool:
        return self.prev_logp is not None


def cd3_active_mask(
    logits: torch.Tensor,
    is_masked: torch.Tensor,
    state: CD3State,
    tau_kl: float = 0.02,
    tau_ent: float = 2.0,
    m: int = 2,
    k_min: int = 1,
) -> Tuple[torch.Tensor, CD3State, Dict[str, float]]:
    """Compute the CD³ active mask for a block of logits.

    A position is *inactive* (converged) if it has been predicted stably for
    at least *m* consecutive steps (low KL, low entropy, no argmax flip, and
    not currently masked).  Inactive positions are skipped in subsequent
    forward passes, reducing compute.

    Args:
        logits:    ``[B, T, V]`` raw model logits for the current block.
        is_masked: ``[B, T]`` bool – True where the token is still masked.
        state:     :class:`CD3State` updated in-place and returned.
        tau_kl:    KL-divergence threshold for convergence check.
        tau_ent:   Entropy (nats) threshold for convergence check.
        m:         Number of consecutive stable steps required (hysteresis).
        k_min:     Minimum number of active positions to keep per batch item.
                   Prevents pathological collapse of the active set.

    Returns:
        active_mask: ``[B, T]`` bool – positions that should be updated.
        state:       Updated :class:`CD3State`.
        stats:       Diagnostic scalars (entropy_mean, kl_mean, flip_rate,
                     active_ratio).
    """
    B, T, V = logits.shape
    device = logits.device

    logp = F.log_softmax(logits, dim=-1)  # [B, T, V]
    p = logp.exp()

    # Entropy H(p) = -(p * log p).sum(-1), shape [B, T]
    entropy = -(p * logp).sum(dim=-1)

    # Argmax predictions [B, T]
    argmax = logits.argmax(dim=-1)

    # ------------------------------------------------------------------ #
    # First call: initialise state and mark everything active             #
    # ------------------------------------------------------------------ #
    if not state.is_initialized():
        state.prev_logp = logp.detach()
        state.prev_argmax = argmax.detach()
        state.stable_count = torch.zeros((B, T), device=device, dtype=torch.int32)
        active = torch.ones((B, T), device=device, dtype=torch.bool)
        stats: Dict[str, float] = {
            "entropy_mean": entropy.mean().item(),
            "kl_mean": 0.0,
            "flip_rate": 0.0,
            "active_ratio": 1.0,
        }
        return active, state, stats

    # ------------------------------------------------------------------ #
    # KL divergence KL(p_s || p_{s-1}) = sum_v p_s * (log p_s - log p_{s-1})
    # Shape [B, T].  Clamped to ≥0 for numerical stability.              #
    # ------------------------------------------------------------------ #
    kl = (p * (logp - state.prev_logp)).sum(dim=-1).clamp(min=0.0)

    # Whether the most-likely token changed [B, T]
    flip = argmax.ne(state.prev_argmax)

    # ------------------------------------------------------------------ #
    # Convergence criterion (all must hold AND position must be unmasked) #
    # ------------------------------------------------------------------ #
    converged_now = (kl < tau_kl) & (entropy < tau_ent) & (~flip) & (~is_masked)

    # Hysteresis: increment stable counter on convergence, reset otherwise
    stable = state.stable_count
    stable = torch.where(converged_now, stable + 1, torch.zeros_like(stable))

    # Fully converged after *m* consecutive stable steps
    converged = stable >= m

    # Active = not converged, OR still masked (masked tokens must be processed)
    active = (~converged) | is_masked

    # ------------------------------------------------------------------ #
    # K_min: guarantee at least k_min active positions per batch item     #
    # ------------------------------------------------------------------ #
    if k_min > 0:
        for b in range(B):
            active_count = int(active[b].sum().item())
            if active_count < k_min:
                n_needed = k_min - active_count
                # Prioritise high-entropy inactive positions
                candidate_entropy = entropy[b].clone()
                candidate_entropy[active[b]] = float("-inf")
                top_idx = candidate_entropy.topk(k=n_needed, largest=True).indices
                active[b, top_idx] = True

    # ------------------------------------------------------------------ #
    # Update state                                                        #
    # ------------------------------------------------------------------ #
    state.prev_logp = logp.detach()
    state.prev_argmax = argmax.detach()
    state.stable_count = stable.detach()

    stats = {
        "entropy_mean": entropy.mean().item(),
        "kl_mean": kl.mean().item(),
        "flip_rate": flip.float().mean().item(),
        "active_ratio": active.float().mean().item(),
    }
    return active, state, stats
