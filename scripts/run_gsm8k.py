#!/usr/bin/env python3
"""
GSM8K 5-shot evaluation script for CD³-FastdLLM.

Usage:
    python scripts/run_gsm8k.py [--config configs/default.yaml] [OPTIONS]

Options are merged on top of the YAML config. CLI overrides have highest
priority.
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
    p = argparse.ArgumentParser(description="GSM8K 5-shot evaluation")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--model-id", default=None, help="Override model.id")
    p.add_argument("--gen-len", type=int, default=None, help="Override generation.gen_len")
    p.add_argument("--num-steps", type=int, default=None, help="Override generation.num_steps")
    p.add_argument("--mode", default=None, choices=["confidence", "cd3"], help="Gating mode")
    p.add_argument("--variant", default=None, choices=["basic", "prefix_cache", "dual_cache"])
    p.add_argument("--max-examples", type=int, default=None, help="Limit number of test examples")
    p.add_argument("--output-dir", default=None, help="Override output.dir")
    p.add_argument("--device", default=None, help="Force device (e.g. 'cpu', 'cuda:0')")
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import yaml  # type: ignore

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
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
        cfg["eval"]["gsm8k"]["max_examples"] = args.max_examples
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    mode = cfg.get("run", {}).get("mode", "cd3")
    variant = cfg.get("run", {}).get("variant", "dual_cache")
    gen_len = cfg["generation"]["gen_len"]
    num_steps = cfg["generation"]["num_steps"]
    block_length = cfg["generation"]["block_length"]
    temperature = cfg["generation"]["temperature"]
    cd3_kwargs = cfg["cd3"]
    max_examples: Optional[int] = cfg["eval"]["gsm8k"]["max_examples"]
    num_fewshot: int = cfg["eval"]["gsm8k"]["num_fewshot"]
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
    from cd3_fastdllm.data import load_gsm8k, gsm8k_fewshot_examples, format_gsm8k_5shot
    from cd3_fastdllm.eval import gsm8k_accuracy

    logger.info("Loading GSM8K …")
    test_ds = load_gsm8k(split=cfg["eval"]["gsm8k"]["split"])
    fewshot = gsm8k_fewshot_examples(n=num_fewshot)

    examples = list(test_ds)
    if max_examples:
        examples = examples[:max_examples]

    logger.info("Evaluating %d examples (mode=%s, variant=%s, gen_len=%d) …",
                len(examples), mode, variant, gen_len)

    # ── Generation loop ─────────────────────────────────────────────────────
    from cd3_fastdllm.decoding.generation import generate_text

    preds: List[str] = []
    golds: List[str] = []
    tok_counts: List[int] = []
    wall_times: List[float] = []

    for i, ex in enumerate(examples):
        prompt = format_gsm8k_5shot(ex["question"], fewshot)
        gold = ex["answer"]

        t0 = time.perf_counter()
        pred, _info = generate_text(
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

        preds.append(pred)
        golds.append(gold)
        tok_counts.append(gen_len)
        wall_times.append(t1 - t0)

        if (i + 1) % 50 == 0:
            running_acc = gsm8k_accuracy(preds, golds)
            logger.info("  [%d/%d] running accuracy: %.2f%%", i + 1, len(examples), running_acc)

    acc = gsm8k_accuracy(preds, golds)
    total_tokens = sum(tok_counts)
    total_time = sum(wall_times)
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    result = {
        "model": cfg["model"]["id"],
        "mode": mode,
        "variant": variant,
        "gen_len": gen_len,
        "num_steps": num_steps,
        "accuracy_pct": acc,
        "num_examples": len(examples),
        "tokens_per_sec": tok_per_sec,
        "wall_time_sec": total_time,
    }

    out_file = output_dir / f"gsm8k_{mode}_{variant}_genlen{gen_len}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info("GSM8K accuracy: %.2f%%  |  %.1f tok/s", acc, tok_per_sec)
    logger.info("Results saved to %s", out_file)


if __name__ == "__main__":
    main()
