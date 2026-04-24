"""Prompt formatting utilities for GSM8K (5-shot) and HumanEval (0-shot)."""

from __future__ import annotations

from typing import List, Dict


def format_gsm8k_5shot(question: str, fewshot: List[Dict[str, str]]) -> str:
    """Build a 5-shot GSM8K prompt.

    The prompt follows the LLaDA / instruction-tuned format:
    ``<system>\\n<example_1>\\n…<example_n>\\nQ: <question>\\nA:``

    Args:
        question: The test question.
        fewshot:  List of dicts with keys ``"question"`` and ``"answer"``.

    Returns:
        Formatted prompt string (does NOT include the expected answer).
    """
    parts: List[str] = [
        "You are a helpful math assistant. Solve each math word problem step by step "
        "and end your answer with '#### <number>'.\n",
    ]
    for ex in fewshot:
        parts.append(f"Q: {ex['question']}\nA: {ex['answer']}\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)


def format_humaneval_0shot(prompt: str) -> str:
    """Return the HumanEval *prompt* field unchanged.

    The HumanEval prompt already contains the function signature and docstring,
    so no additional formatting is required for 0-shot evaluation.

    Args:
        prompt: Raw ``prompt`` field from the HumanEval dataset.

    Returns:
        The same prompt, passed through unchanged.
    """
    return prompt
