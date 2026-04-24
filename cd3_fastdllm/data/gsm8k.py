"""GSM8K dataset loader."""

from __future__ import annotations

from typing import List, Dict


def load_gsm8k(split: str = "test"):
    """Load the GSM8K dataset from HuggingFace.

    Args:
        split: ``"train"`` or ``"test"``.

    Returns:
        HuggingFace ``Dataset`` with fields ``question`` and ``answer``.
    """
    from datasets import load_dataset  # type: ignore

    return load_dataset("gsm8k", "main", split=split)


def gsm8k_fewshot_examples(n: int = 5) -> List[Dict[str, str]]:
    """Return *n* fixed few-shot examples from the GSM8K training split.

    Args:
        n: Number of examples.  Must be ≤ size of the training set.

    Returns:
        List of dicts with keys ``"question"`` and ``"answer"``.
    """
    train = load_gsm8k(split="train")
    return [{"question": train[i]["question"], "answer": train[i]["answer"]} for i in range(n)]
