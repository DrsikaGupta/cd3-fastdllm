"""HumanEval pass@1 evaluation utilities.

Uses the official ``evaluate`` library's ``code_eval`` metric which compiles
and executes generated Python code in a sandboxed subprocess.

IMPORTANT: executing generated code is inherently unsafe.  Always run inside
a container / restricted environment.  Set the environment variable
``HF_ALLOW_CODE_EVAL=1`` to acknowledge this and enable evaluation.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Tuple


def extract_code_completion(output: str, prompt: str) -> str:
    """Extract a clean Python completion from model output.

    The model is expected to continue the function body started in *prompt*.
    We strip any non-Python preamble and truncate at the first top-level
    definition after the initial function (i.e. stop at the next ``def`` /
    ``class`` at indent level 0 that is not part of the prompt continuation).

    Args:
        output:  Raw model-generated text (may include the prompt echo).
        prompt:  Original prompt passed to the model.

    Returns:
        Clean Python code string (prompt + completion body).
    """
    # Remove echoed prompt if present
    if output.startswith(prompt):
        completion = output[len(prompt):]
    else:
        completion = output

    # Truncate at next top-level function/class definition
    lines = completion.splitlines(keepends=True)
    body_lines: List[str] = []
    for line in lines:
        if re.match(r"^(def |class )", line) and body_lines:
            break
        body_lines.append(line)

    return prompt + "".join(body_lines)


def humaneval_pass_at_1(
    prompts: Sequence[str],
    completions: Sequence[str],
    tests: Sequence[str],
    entry_points: Sequence[str],
    timeout: float = 3.0,
) -> Tuple[float, List[bool]]:
    """Compute HumanEval pass@1 using the ``evaluate`` code_eval metric.

    Args:
        prompts:       Original function prompts (used to build full programs).
        completions:   Model-generated continuations, one per prompt.
        tests:         Unit-test strings from the HumanEval dataset.
        entry_points:  Function names, used to construct ``check(entry_point)``
                       calls.
        timeout:       Execution timeout per test case (seconds).

    Returns:
        pass_at_1:  Overall pass@1 as a percentage in ``[0, 100]``.
        results:    Per-example bool list.
    """
    try:
        import evaluate  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'evaluate' package is required for HumanEval evaluation. "
            "Install it with:  pip install evaluate"
        ) from exc

    # Allow code execution (user must acknowledge risk)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    metric = evaluate.load("code_eval")

    # Build full programs (completion + test harness)
    programs: List[List[str]] = []
    for prompt, completion, test, ep in zip(prompts, completions, tests, entry_points):
        full_code = extract_code_completion(completion, prompt)
        program = f"{full_code}\n\n{test}\ncheck({ep})"
        programs.append([program])

    results, _ = metric.compute(
        references=[["pass"]] * len(programs),
        predictions=programs,
        timeout=timeout,
        k=[1],
    )

    # results["pass@1"] is a float in [0, 1] from the metric
    pass_at_1_fraction: float = results.get("pass@1", 0.0)
    pass_at_1_pct = 100.0 * pass_at_1_fraction

    # Per-example bool: rerun to get individual results
    per_example: List[bool] = []
    for prog_list in programs:
        r, _ = metric.compute(
            references=[["pass"]],
            predictions=[prog_list],
            timeout=timeout,
            k=[1],
        )
        per_example.append(bool(r.get("pass@1", 0.0) >= 1.0))

    return pass_at_1_pct, per_example
