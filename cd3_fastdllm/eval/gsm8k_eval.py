"""GSM8K accuracy evaluation utilities.

GSM8K gold answers use the format ``… #### <number>`` at the end of the
chain-of-thought.  Predictions are extracted by looking for the last
numeric-looking token in the output.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence


def extract_final_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from a text string.

    Handles both the ``#### <number>`` GSM8K format and plain numerals.
    Strips commas used as thousands separators.

    Args:
        text: Raw model output or gold answer string.

    Returns:
        The last numeric token as a string, or ``None`` if not found.
    """
    # Prefer the explicit GSM8K marker "####"
    marker_match = re.search(r"####\s*([\-+]?\d[\d,]*(?:\.\d+)?)", text)
    if marker_match:
        return marker_match.group(1).replace(",", "")

    # Fall back to last number in the text
    all_nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if all_nums:
        return all_nums[-1].replace(",", "")
    return None


def gsm8k_is_correct(pred: str, gold: str) -> bool:
    """Return True if the predicted answer matches the gold answer numerically.

    Args:
        pred: Model-generated text for one example.
        gold: Gold answer string from the dataset.

    Returns:
        ``True`` if both extract to the same number string.
    """
    gold_num = extract_final_number(gold)
    pred_num = extract_final_number(pred)
    if gold_num is None or pred_num is None:
        return False
    # Normalise: strip leading zeros after decimal, strip trailing dot
    try:
        return float(pred_num) == float(gold_num)
    except ValueError:
        return pred_num == gold_num


def gsm8k_accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Compute accuracy (%) over a list of predictions.

    Args:
        preds: Sequence of model output strings.
        golds: Sequence of gold answer strings (same order).

    Returns:
        Accuracy as a percentage in ``[0, 100]``.
    """
    if not golds:
        return 0.0
    correct = sum(gsm8k_is_correct(p, g) for p, g in zip(preds, golds))
    return 100.0 * correct / len(golds)
