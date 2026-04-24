"""
Model loading utilities.

Loads a LLaDA-style masked-diffusion language model (``LLaDAModelLM`` or any
HuggingFace ``AutoModelForMaskedLM`` / ``AutoModelForCausalLM`` compatible
checkpoint) together with the associated tokenizer.

Default model: ``GSAI-ML/LLaDA-8B-Instruct``
This checkpoint is publicly available on HuggingFace and is the canonical
LLaDA-8B model used in the Fast-dLLM paper benchmarks.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def load_model(
    model_id: str = _DEFAULT_MODEL_ID,
    dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, object]:
    """Load a LLaDA-compatible model and tokenizer from HuggingFace.

    The function tries ``AutoModelForCausalLM`` first (covers Dream-style
    checkpoints and LLaDA configs that register as causal), then falls back to
    ``AutoModel`` for plain encoder checkpoints.

    Args:
        model_id:          HuggingFace model repository id.
        dtype:             Weight dtype.  Defaults to ``torch.bfloat16`` when a
                           CUDA device is available, otherwise ``torch.float32``.
        device_map:        Passed to ``from_pretrained``. Use ``"auto"`` for
                           automatic multi-GPU placement or ``"cpu"`` to force CPU.
        trust_remote_code: Needed for LLaDA which ships custom modelling code.
        cache_dir:         Optional HuggingFace cache directory override.

    Returns:
        model:     The loaded model in eval mode.
        tokenizer: The associated HuggingFace tokenizer.

    Example::

        from cd3_fastdllm.models import load_model
        model, tokenizer = load_model()
        mask_id = tokenizer.mask_token_id
    """
    # Late imports to avoid hard dependency at import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logger.info("Loading tokenizer from %s …", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    logger.info("Loading model %s (dtype=%s, device_map=%s) …", model_id, dtype, device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    model.eval()

    # Ensure mask_token is registered
    if tokenizer.mask_token is None:
        logger.warning(
            "Tokenizer has no mask_token. If the model uses a custom mask id "
            "(e.g. 126336 for LLaDA-8B), set tokenizer.mask_token manually."
        )

    return model, tokenizer
