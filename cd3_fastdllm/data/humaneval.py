"""HumanEval dataset loader."""

from __future__ import annotations


def load_humaneval(split: str = "test"):
    """Load the OpenAI HumanEval dataset from HuggingFace.

    Args:
        split: Dataset split (``"test"`` is the only standard split).

    Returns:
        HuggingFace ``Dataset`` with fields including ``task_id``, ``prompt``,
        ``canonical_solution``, ``test``, and ``entry_point``.
    """
    from datasets import load_dataset  # type: ignore

    return load_dataset("openai_humaneval", split=split)
