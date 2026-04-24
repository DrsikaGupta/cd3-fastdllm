#!/usr/bin/env python3
"""
HumanEval 0-shot pass@1 evaluation script for CD³-FastdLLM.

Usage:
    python scripts/run_humaneval.py [--config configs/default.yaml] [OPTIONS]

WARNING: This script executes model-generated Python code.
Set the environment variable HF_ALLOW_CODE_EVAL=1 to enable execution.
Always run inside a sandboxed environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval 0-shot pass@1 evaluation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-id", default=None)
    p.add_argument("--gen-len", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--mode", default=None, choices=["confidence", "cd3"])
    p.add_argument("--variant", default=None, choices=["basic", "prefix_cache", "dual_cache"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-code-eval", action="store_true",
                   help="Set HF_ALLOW_CODE_EVAL=1 automatically")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.allow_code_eval:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    import yaml  # type: ignore

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.model_id:
        cfg["model"]["id"] = args.model_id
    if args.gen_len:
        cfg["generation"]["gen_len"] = args.gen_len
    if args.num_steps:
        cfg["generation"]["num_steps"] = args.num_steps
    if args.mode:
        cfg.setdefault("run", {})["mode"] = args.mode
    if args.variant:
        cfg.setdefault("run", {})["variant"] = args.variant
    if args.max_examples:
        cfg["eval"]["humaneval"]["max_examples"] = args.max_examples
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    mode = cfg.get("run", {}).get("mode", "cd3")
    variant = cfg.get("run", {}).get("variant", "dual_cache")
    gen_len = cfg["generation"]["gen_len"]
    num_steps = cfg["generation"]["num_steps"]
    block_length = cfg["generation"]["block_length"]
    temperature = cfg["generation"]["temperature"]
    cd3_kwargs = cfg["cd3"]
    max_examples: Optional[int] = cfg["eval"]["humaneval"]["max_examples"]
    timeout: float = cfg["eval"]["humaneval"]["timeout"]
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    from cd3_fastdllm.models import load_model

    device_map = args.device if args.device else cfg["model"].get("device_map", "auto")
    model, tokenizer = load_model(
        model_id=cfg["model"]["id"],
        device_map=device_map,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    # ── Load dataset ────────────────────────────────────────────────────────
    from cd3_fastdllm.data import load_humaneval, format_humaneval_0shot
    from cd3_fastdllm.eval import humaneval_pass_at_1

    logger.info("Loading HumanEval …")
    ds = load_humaneval(split=cfg["eval"]["humaneval"]["split"])
    examples = list(ds)
    if max_examples:
        examples = examples[:max_examples]

    logger.info("Evaluating %d examples (mode=%s, variant=%s, gen_len=%d) …",
                len(examples), mode, variant, gen_len)

    # ── Generation loop ─────────────────────────────────────────────────────
    from cd3_fastdllm.decoding.generation import generate_text

    raw_completions: List[str] = []
    prompts: List[str] = []
    tests: List[str] = []
    entry_points: List[str] = []
    tok_counts: List[int] = []
    wall_times: List[float] = []

    for i, ex in enumerate(examples):
        prompt = format_humaneval_0shot(ex["prompt"])

        t0 = time.perf_counter()
        completion, _info = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gen_len=gen_len,
            num_steps=num_steps,
            mode=mode,
            variant=variant,
            block_length=block_length,
            cd3_kwargs=cd3_kwargs if mode == "cd3" else {},
            temperature=temperature,
        )
        t1 = time.perf_counter()

        raw_completions.append(completion)
        prompts.append(prompt)
        tests.append(ex["test"])
        entry_points.append(ex["entry_point"])
        tok_counts.append(gen_len)
        wall_times.append(t1 - t0)

        if (i + 1) % 20 == 0:
            logger.info("  Generated %d/%d examples", i + 1, len(examples))

    # ── Evaluate ────────────────────────────────────────────────────────────
    logger.info("Running code evaluation …")
    pass_at_1, per_example = humaneval_pass_at_1(
        prompts=prompts,
        completions=raw_completions,
        tests=tests,
        entry_points=entry_points,
        timeout=timeout,
    )

    total_tokens = sum(tok_counts)
    total_time = sum(wall_times)
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    result = {
        "model": cfg["model"]["id"],
        "mode": mode,
        "variant": variant,
        "gen_len": gen_len,
        "num_steps": num_steps,
        "pass_at_1_pct": pass_at_1,
        "num_examples": len(examples),
        "tokens_per_sec": tok_per_sec,
        "wall_time_sec": total_time,
        "per_example_pass": per_example,
    }

    out_file = output_dir / f"humaneval_{mode}_{variant}_genlen{gen_len}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info("HumanEval pass@1: %.2f%%  |  %.1f tok/s", pass_at_1, tok_per_sec)
    logger.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()
