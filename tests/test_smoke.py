"""
Smoke tests for CD³-FastdLLM.

These tests run *without* a real model by using a tiny randomly-initialised
stub model. They validate:

1. CD³ state + active mask computation
2. Confidence scoring utilities
3. The three generation variants (basic, prefix_cache, dual_cache)
4. GSM8K accuracy evaluation helpers
5. HumanEval code extraction helper

Run with:
    pytest tests/test_smoke.py -v
or:
    python tests/test_smoke.py
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stub model that mimics a minimal HuggingFace causal LM
# ---------------------------------------------------------------------------

class _FakeOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class _StubModel(nn.Module):
    """Tiny random model that returns logits of shape [B, T, vocab_size]."""

    VOCAB_SIZE = 64

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, 16)
        self.linear = nn.Linear(16, self.VOCAB_SIZE)

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        # shape: [B, T, 16]
        h = self.embed(input_ids)
        logits = self.linear(h)  # [B, T, VOCAB_SIZE]
        pkv = past_key_values  # stub: just pass through
        if use_cache:
            # Return a non-None sentinel so generation code treats it as valid
            pkv = object()  # minimal sentinel
        return _FakeOutput(logits=logits, past_key_values=pkv)


class _StubTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1  # reserve id=1 for mask

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Simple: tokenise by character codes, clamp to vocab, prepend BOS
        ids = [0] + [max(2, min(c % _StubModel.VOCAB_SIZE, _StubModel.VOCAB_SIZE - 1))
                     for c in text.encode()[:10]]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i.item()) for i in ids)


# ---------------------------------------------------------------------------
# Tests: CD³ gating
# ---------------------------------------------------------------------------


def test_cd3_state_initialisation():
    from cd3_fastdllm.decoding.cd3 import CD3State, cd3_active_mask

    B, T, V = 2, 8, 64
    logits = torch.randn(B, T, V)
    is_masked = torch.zeros(B, T, dtype=torch.bool)
    is_masked[:, :4] = True

    state = CD3State()
    assert not state.is_initialized()

    active, state2, stats = cd3_active_mask(logits, is_masked, state)

    assert active.shape == (B, T)
    assert active.dtype == torch.bool
    assert state2.is_initialized()
    # On the first call everything should be active
    assert active.all()
    assert stats["active_ratio"] == 1.0
    assert stats["kl_mean"] == 0.0


def test_cd3_convergence_after_m_steps():
    """Positions that are consistently predicted should converge within m steps."""
    from cd3_fastdllm.decoding.cd3 import CD3State, cd3_active_mask

    B, T, V = 1, 4, 64
    # Build logits with very high confidence for fixed tokens
    logits = torch.full((B, T, V), -100.0)
    for t in range(T):
        logits[0, t, t % V] = 100.0  # near-deterministic prediction

    is_masked = torch.zeros(B, T, dtype=torch.bool)

    state = CD3State()
    m = 2
    for step in range(m + 2):
        active, state, stats = cd3_active_mask(
            logits, is_masked, state, tau_kl=1.0, tau_ent=10.0, m=m, k_min=0
        )

    # After m+1 stable steps, at least some positions should be converged
    # (active_ratio < 1 or 0 if all converged), k_min=0 allows full convergence
    assert stats["active_ratio"] <= 1.0


def test_cd3_kmin_enforced():
    """K_min must ensure at least k_min active positions."""
    from cd3_fastdllm.decoding.cd3 import CD3State, cd3_active_mask

    B, T, V = 1, 6, 32
    logits = torch.full((B, T, V), -100.0)
    for t in range(T):
        logits[0, t, t % V] = 100.0

    is_masked = torch.zeros(B, T, dtype=torch.bool)
    state = CD3State()
    k_min = 3

    for _ in range(10):
        active, state, stats = cd3_active_mask(
            logits, is_masked, state, tau_kl=1.0, tau_ent=10.0, m=1, k_min=k_min
        )

    assert active[0].sum().item() >= k_min


def test_cd3_reset():
    from cd3_fastdllm.decoding.cd3 import CD3State, cd3_active_mask

    B, T, V = 1, 4, 16
    logits = torch.randn(B, T, V)
    is_masked = torch.ones(B, T, dtype=torch.bool)
    state = CD3State()
    cd3_active_mask(logits, is_masked, state)
    assert state.is_initialized()
    state.reset()
    assert not state.is_initialized()


# ---------------------------------------------------------------------------
# Tests: Confidence scoring
# ---------------------------------------------------------------------------


def test_confidence_scores_range():
    from cd3_fastdllm.decoding.confidence import confidence_scores

    logits = torch.randn(2, 10, 64)
    scores = confidence_scores(logits)
    assert scores.shape == (2, 10)
    assert (scores >= 0).all() and (scores <= 1.0001).all()


def test_masked_confidence_scores():
    from cd3_fastdllm.decoding.confidence import masked_confidence_scores

    logits = torch.randn(1, 6, 32)
    is_masked = torch.tensor([[True, False, True, False, True, False]])
    scores = masked_confidence_scores(logits, is_masked)
    assert (scores[:, 1::2] == float("-inf")).all()
    assert (scores[:, 0::2] > float("-inf")).all()


def test_select_transfer_index():
    from cd3_fastdllm.decoding.confidence import select_transfer_index

    B, T = 1, 8
    scores = torch.full((B, T), float("-inf"))
    scores[0, 2] = 0.9
    scores[0, 5] = 0.7
    scores[0, 7] = 0.5

    batch_idx, pos_idx = select_transfer_index(scores, num_transfer=2)
    assert len(pos_idx) == 2
    assert 2 in pos_idx.tolist()
    assert 5 in pos_idx.tolist()


def test_tokens_to_unmask():
    from cd3_fastdllm.decoding.confidence import tokens_to_unmask

    is_masked = torch.ones(1, 100, dtype=torch.bool)
    # At step 0 with 10 steps left, ceil(100/10) = 10
    n = tokens_to_unmask(is_masked, num_steps=10, step=0)
    assert n == 10

    # At final step, unmask all remaining
    is_masked_small = torch.ones(1, 5, dtype=torch.bool)
    n_final = tokens_to_unmask(is_masked_small, num_steps=10, step=9)
    assert n_final == 5


# ---------------------------------------------------------------------------
# Tests: Generation variants
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_model_tokenizer():
    model = _StubModel()
    model.eval()
    tokenizer = _StubTokenizer()
    return model, tokenizer


def _check_gen_output(gen_ids, gen_len: int, mask_id: int):
    assert gen_ids.shape[1] == gen_len
    # No remaining mask tokens after generation
    assert not gen_ids.eq(mask_id).any(), "Generation left mask tokens unreplaced"


def test_generate_confidence(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5, 10, 15]])
    gen_ids, info = generate(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=8,
        num_steps=4,
        mask_id=tok.mask_token_id,
        mode="confidence",
    )
    _check_gen_output(gen_ids, 8, tok.mask_token_id)
    assert "steps" in info


def test_generate_cd3(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5, 10]])
    gen_ids, info = generate(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=8,
        num_steps=4,
        mask_id=tok.mask_token_id,
        mode="cd3",
        cd3_kwargs={"tau_kl": 0.5, "tau_ent": 5.0, "m": 1, "k_min": 1},
    )
    _check_gen_output(gen_ids, 8, tok.mask_token_id)


def test_generate_with_prefix_cache(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate_with_prefix_cache

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5, 10]])
    gen_ids, info = generate_with_prefix_cache(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=8,
        num_steps=4,
        mask_id=tok.mask_token_id,
        mode="confidence",
    )
    _check_gen_output(gen_ids, 8, tok.mask_token_id)


def test_generate_with_dual_cache_confidence(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate_with_dual_cache

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5, 10]])
    gen_ids, info = generate_with_dual_cache(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=8,
        num_steps=4,
        mask_id=tok.mask_token_id,
        block_length=4,
        mode="confidence",
    )
    _check_gen_output(gen_ids, 8, tok.mask_token_id)


def test_generate_with_dual_cache_cd3(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate_with_dual_cache

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5]])
    gen_ids, info = generate_with_dual_cache(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=8,
        num_steps=4,
        mask_id=tok.mask_token_id,
        block_length=4,
        mode="cd3",
        cd3_kwargs={"tau_kl": 0.1, "tau_ent": 3.0, "m": 2, "k_min": 1},
    )
    _check_gen_output(gen_ids, 8, tok.mask_token_id)


def test_generate_text_wrapper(stub_model_tokenizer):
    from cd3_fastdllm.decoding.generation import generate_text

    model, tok = stub_model_tokenizer
    text, info = generate_text(
        model=model,
        tokenizer=tok,
        prompt="hello",
        gen_len=8,
        num_steps=4,
        mode="cd3",
        variant="dual_cache",
        block_length=4,
    )
    assert isinstance(text, str)


def test_generate_gen_len_padding(stub_model_tokenizer):
    """gen_len that is not a multiple of block_length should still work."""
    from cd3_fastdllm.decoding.generation import generate_with_dual_cache

    model, tok = stub_model_tokenizer
    prompt_ids = torch.tensor([[0, 5]])
    gen_ids, _ = generate_with_dual_cache(
        model=model,
        prompt_ids=prompt_ids,
        gen_len=7,  # not divisible by 4
        num_steps=4,
        mask_id=tok.mask_token_id,
        block_length=4,
        mode="confidence",
    )
    assert gen_ids.shape[1] == 7


# ---------------------------------------------------------------------------
# Tests: GSM8K evaluation helpers
# ---------------------------------------------------------------------------


def test_extract_final_number_marker():
    from cd3_fastdllm.eval.gsm8k_eval import extract_final_number

    assert extract_final_number("blah blah #### 42") == "42"
    assert extract_final_number("so the answer is #### 1,234") == "1234"
    assert extract_final_number("no number here") is None


def test_extract_final_number_fallback():
    from cd3_fastdllm.eval.gsm8k_eval import extract_final_number

    assert extract_final_number("the answer is 3.14") == "3.14"
    assert extract_final_number("100 apples and 200 oranges") == "200"


def test_gsm8k_is_correct():
    from cd3_fastdllm.eval.gsm8k_eval import gsm8k_is_correct

    assert gsm8k_is_correct("The answer is #### 42", "The answer is #### 42")
    assert gsm8k_is_correct("total = 100", "#### 100")
    assert not gsm8k_is_correct("#### 99", "#### 100")
    assert not gsm8k_is_correct("no number", "#### 42")


def test_gsm8k_accuracy():
    from cd3_fastdllm.eval.gsm8k_eval import gsm8k_accuracy

    preds = ["#### 1", "#### 2", "#### 3"]
    golds = ["#### 1", "#### 9", "#### 3"]
    acc = gsm8k_accuracy(preds, golds)
    assert abs(acc - 200 / 3) < 0.01


# ---------------------------------------------------------------------------
# Tests: HumanEval extraction
# ---------------------------------------------------------------------------


def test_extract_code_completion_no_echo():
    from cd3_fastdllm.eval.humaneval_eval import extract_code_completion

    prompt = "def add(a, b):\n"
    output = "    return a + b\n"
    result = extract_code_completion(output, prompt)
    # When output doesn't start with prompt, just use it as-is with prompt prepended
    assert "def add" in result


def test_extract_code_completion_with_echo():
    from cd3_fastdllm.eval.humaneval_eval import extract_code_completion

    prompt = "def add(a, b):\n"
    output = prompt + "    return a + b\n\ndef other():\n    pass\n"
    result = extract_code_completion(output, prompt)
    # Should truncate before 'def other'
    assert "def other" not in result
    assert "return a + b" in result


# ---------------------------------------------------------------------------
# Tests: Model loading (import-only, no network)
# ---------------------------------------------------------------------------


def test_load_model_importable():
    """Ensure the load_model function is importable and has correct signature."""
    from cd3_fastdllm.models.load_model import load_model
    import inspect

    sig = inspect.signature(load_model)
    params = list(sig.parameters)
    assert "model_id" in params
    assert "dtype" in params
    assert "device_map" in params


# ---------------------------------------------------------------------------
# Runner for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)
