"""
Baseline: run the untouched base model on your held-out test set.
Writes metrics_baseline.json — the bar your fine-tune must beat.

Usage:
    python 01_baseline.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --test data/test.jsonl \
        --out runs/baseline/

The test file is JSONL with one object per line:
    {"messages": [...], "expected": "<gold answer>", "task": "<task_name>"}

`expected` and `task` are used by the metric. `messages` is the chat
history fed to the model (system + user, no assistant turn yet).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="0=all")
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
        if torch.cuda.is_available() else None,
    )
    model.eval()

    rows = list(load_jsonl(args.test))
    if args.limit:
        rows = rows[: args.limit]

    preds = []
    correct = 0
    total_in = 0
    total_out = 0
    t0 = time.time()

    for i, ex in enumerate(rows):
        prompt = tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-6),
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        em = exact_match(gen, ex["expected"])
        correct += em
        total_in += inputs["input_ids"].shape[1]
        total_out += out.shape[1] - inputs["input_ids"].shape[1]
        preds.append({"i": i, "pred": gen, "expected": ex["expected"], "match": em})

        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(rows)}] running acc={correct/(i+1):.3f}")

    wall = time.time() - t0
    metrics = {
        "model": args.model,
        "n": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "tokens_in": total_in,
        "tokens_out": total_out,
        "wall_seconds": round(wall, 2),
        "tokens_per_sec": round(total_out / max(wall, 1e-6), 2),
    }
    (out_dir / "preds.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds)
    )
    (out_dir / "metrics_baseline.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
