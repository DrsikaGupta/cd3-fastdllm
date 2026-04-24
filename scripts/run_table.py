#!/usr/bin/env python3
"""
Benchmark table script: reproduces the CD³ paper Table comparing modes at
generation lengths 256 and 512.

Columns: mode | gen_len | GSM8K acc% | HumanEval pass@1% | tok/s | speedup vs baseline

Usage:
    python scripts/run_table.py [--config configs/default.yaml] [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce CD³ benchmark table")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-id", default=None)
    p.add_argument("--max-gsm8k", type=int, default=None,
                   help="Max GSM8K examples per mode (None=all 1319)")
    p.add_argument("--max-humaneval", type=int, default=None,
                   help="Max HumanEval examples per mode (None=all 164)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-code-eval", action="store_true")
    p.add_argument("--gen-lengths", nargs="+", type=int, default=None,
                   help="Generation lengths to evaluate (default: [256, 512])")
    return p.parse_args()


def evaluate_mode(
    model,
    tokenizer,
    mode: str,
    variant: str,
    gen_len: int,
    num_steps: int,
    block_length: int,
    temperature: float,
    cd3_kwargs: Dict,
    gsm8k_examples: List,
    humaneval_examples: List,
    fewshot: List,
    timeout: float,
) -> Dict:
    """Run both GSM8K and HumanEval for a single (mode, variant, gen_len) combo."""
    from cd3_fastdllm.decoding.generation import generate_text
    from cd3_fastdllm.data import format_gsm8k_5shot, format_humaneval_0shot
    from cd3_fastdllm.eval import (
        gsm8k_accuracy,
        humaneval_pass_at_1,
        extract_final_number,
    )

    # ── GSM8K ──────────────────────────────────────────────────────────────
    gsm8k_preds: List[str] = []
    gsm8k_golds: List[str] = []
    gsm8k_tokens = 0
    gsm8k_time = 0.0

    for ex in gsm8k_examples:
        prompt = format_gsm8k_5shot(ex["question"], fewshot)
        t0 = time.perf_counter()
        pred, _ = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt,
            gen_len=gen_len, num_steps=num_steps, mode=mode,
            variant=variant, block_length=block_length,
            cd3_kwargs=cd3_kwargs if mode == "cd3" else {},
            temperature=temperature,
        )
        gsm8k_time += time.perf_counter() - t0
        gsm8k_preds.append(pred)
        gsm8k_golds.append(ex["answer"])
        gsm8k_tokens += gen_len

    gsm8k_acc = gsm8k_accuracy(gsm8k_preds, gsm8k_golds)
    gsm8k_tps = gsm8k_tokens / gsm8k_time if gsm8k_time > 0 else 0.0

    # ── HumanEval ──────────────────────────────────────────────────────────
    he_completions: List[str] = []
    he_prompts: List[str] = []
    he_tests: List[str] = []
    he_eps: List[str] = []
    he_tokens = 0
    he_time = 0.0

    for ex in humaneval_examples:
        prompt = format_humaneval_0shot(ex["prompt"])
        t0 = time.perf_counter()
        completion, _ = generate_text(
            model=model, tokenizer=tokenizer, prompt=prompt,
            gen_len=gen_len, num_steps=num_steps, mode=mode,
            variant=variant, block_length=block_length,
            cd3_kwargs=cd3_kwargs if mode == "cd3" else {},
            temperature=temperature,
        )
        he_time += time.perf_counter() - t0
        he_completions.append(completion)
        he_prompts.append(prompt)
        he_tests.append(ex["test"])
        he_eps.append(ex["entry_point"])
        he_tokens += gen_len

    he_pass1 = 0.0
    if he_completions:
        he_pass1, _ = humaneval_pass_at_1(
            he_prompts, he_completions, he_tests, he_eps, timeout=timeout
        )

    he_tps = he_tokens / he_time if he_time > 0 else 0.0
    avg_tps = (gsm8k_tokens + he_tokens) / (gsm8k_time + he_time) if (gsm8k_time + he_time) > 0 else 0.0

    return {
        "mode": mode,
        "variant": variant,
        "gen_len": gen_len,
        "gsm8k_acc": gsm8k_acc,
        "humaneval_pass1": he_pass1,
        "gsm8k_tps": gsm8k_tps,
        "humaneval_tps": he_tps,
        "avg_tps": avg_tps,
    }


def print_table(rows: List[Dict], gen_len: int) -> None:
    """Pretty-print the results table for a given gen_len."""
    baseline_tps = next(
        (r["avg_tps"] for r in rows if r["mode"] == "confidence" and r["variant"] == "basic"),
        None,
    )

    header = f"\n{'Mode':<25} {'gen_len':>8} {'GSM8K%':>8} {'HE pass@1%':>12} {'tok/s':>8} {'speedup':>8}"
    print("=" * len(header))
    print(f"Generation length: {gen_len}")
    print(header)
    print("-" * len(header))

    for r in rows:
        if r["gen_len"] != gen_len:
            continue
        speedup = r["avg_tps"] / baseline_tps if baseline_tps else float("nan")
        name = f"{r['mode']}({r['variant']})"
        print(
            f"{name:<25} {r['gen_len']:>8} {r['gsm8k_acc']:>7.2f}% "
            f"{r['humaneval_pass1']:>11.2f}% {r['avg_tps']:>8.1f} {speedup:>7.2f}x"
        )
    print("=" * len(header))


def main() -> None:
    args = parse_args()

    if args.allow_code_eval:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    import yaml  # type: ignore

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.model_id:
        cfg["model"]["id"] = args.model_id
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_lengths = args.gen_lengths or cfg["eval"].get("gen_lengths", [256, 512])
    table_modes = cfg["eval"]["table_modes"]
    cd3_kwargs = cfg["cd3"]
    block_length = cfg["generation"]["block_length"]
    temperature = cfg["generation"]["temperature"]
    timeout = cfg["eval"]["humaneval"]["timeout"]

    max_gsm8k: Optional[int] = args.max_gsm8k or cfg["eval"]["gsm8k"].get("max_examples")
    max_he: Optional[int] = args.max_humaneval or cfg["eval"]["humaneval"].get("max_examples")

    # ── Load model once ─────────────────────────────────────────────────────
    from cd3_fastdllm.models import load_model

    device_map = args.device if args.device else cfg["model"].get("device_map", "auto")
    logger.info("Loading model %s …", cfg["model"]["id"])
    model, tokenizer = load_model(
        model_id=cfg["model"]["id"],
        device_map=device_map,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    # ── Load datasets once ──────────────────────────────────────────────────
    from cd3_fastdllm.data import load_gsm8k, gsm8k_fewshot_examples, load_humaneval

    logger.info("Loading datasets …")
    gsm8k_ds = load_gsm8k(split=cfg["eval"]["gsm8k"]["split"])
    fewshot = gsm8k_fewshot_examples(n=cfg["eval"]["gsm8k"]["num_fewshot"])
    gsm8k_examples = list(gsm8k_ds)[:max_gsm8k] if max_gsm8k else list(gsm8k_ds)

    he_ds = load_humaneval(split=cfg["eval"]["humaneval"]["split"])
    he_examples = list(he_ds)[:max_he] if max_he else list(he_ds)

    # ── Run all modes × gen_lengths ─────────────────────────────────────────
    all_rows: List[Dict] = []

    for gen_len in gen_lengths:
        num_steps = cfg["generation"]["num_steps"]

        for m_cfg in table_modes:
            name = m_cfg["name"]
            mode = m_cfg["mode"]
            variant = m_cfg["variant"]

            logger.info("Running: %s  gen_len=%d …", name, gen_len)

            row = evaluate_mode(
                model=model,
                tokenizer=tokenizer,
                mode=mode,
                variant=variant,
                gen_len=gen_len,
                num_steps=num_steps,
                block_length=block_length,
                temperature=temperature,
                cd3_kwargs=cd3_kwargs,
                gsm8k_examples=gsm8k_examples,
                humaneval_examples=he_examples,
                fewshot=fewshot,
                timeout=timeout,
            )
            row["display_name"] = name
            all_rows.append(row)

            # Save intermediate result
            out_file = output_dir / f"table_{name.replace(' ', '_')}_genlen{gen_len}.json"
            with open(out_file, "w") as f:
                json.dump(row, f, indent=2)

    # ── Print summary table ──────────────────────────────────────────────────
    for gl in gen_lengths:
        print_table(all_rows, gl)

    # Save all rows
    all_file = output_dir / "table_all.json"
    with open(all_file, "w") as f:
        json.dump(all_rows, f, indent=2)
    logger.info("Full table saved to %s", all_file)


if __name__ == "__main__":
    main()
