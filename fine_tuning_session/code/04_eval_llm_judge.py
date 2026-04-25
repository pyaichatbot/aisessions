"""
LLM-as-judge evaluation. Cited: Zheng et al., 2023, arXiv:2306.05685.

Two modes:
  - pointwise: judge scores 1..10 on a rubric.
  - pairwise:  judge picks A vs B; runs both orders, counts a win
               only if both orderings agree (controls for position
               bias documented in the MT-Bench paper).
  - faithfulness: yes/no against ground_truth (hallucination metric).

Usage:
    python 04_eval_llm_judge.py pointwise --preds runs/ft1/preds.jsonl \
        --judge claude-sonnet-4-6 --out runs/ft1/judge_pointwise.json

    python 04_eval_llm_judge.py pairwise --a runs/baseline/preds.jsonl \
        --b runs/ft1/preds.jsonl --judge claude-sonnet-4-6 \
        --out runs/ft1/judge_pairwise.json

    python 04_eval_llm_judge.py faithful --preds runs/ft1/preds.jsonl \
        --grounded data/grounded.jsonl --judge claude-sonnet-4-6 \
        --out runs/ft1/judge_faithful.json

Each preds.jsonl row: {"i": int, "pred": str, "expected": str, ...}
Set ANTHROPIC_API_KEY env var.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Lazy import — only one judge family used per run.
def _client():
    import anthropic
    return anthropic.Anthropic()


JUDGE_MODEL_DEFAULT = "claude-sonnet-4-6"


# ---------- prompts ----------
POINTWISE_RUBRIC = """You are a strict evaluator. Score the answer 1-10
on three criteria, then average. Brevity is not penalized.
- Correctness: matches the expected answer in meaning.
- Completeness: covers what was asked, no missing key facts.
- Format: follows any structural requirements (JSON, code, etc.).

Reply with a single line:
SCORE: <number 1-10>
Then one sentence of justification.

QUESTION:
{question}

EXPECTED:
{expected}

ANSWER:
{answer}
"""

PAIRWISE = """You compare two answers to the same question. Same length
budget for both. Reply EXACTLY one of: A, B, TIE. Then one sentence why.

QUESTION:
{question}

EXPECTED (reference, not shown to either model):
{expected}

ANSWER A:
{a}

ANSWER B:
{b}
"""

FAITHFUL = """Decide whether the answer contains ONLY facts that are
entailed by the ground truth. Extra correct facts are fine. Any fact
that contradicts or is not supported by the ground truth = NO.

Reply EXACTLY one of: YES, NO. Then one sentence why.

QUESTION:
{question}

GROUND TRUTH:
{truth}

ANSWER:
{answer}
"""


def _ask(client, model: str, prompt: str, max_tokens: int = 256) -> str:
    for attempt in range(4):
        try:
            r = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text.strip()
        except Exception as e:                  # network / 5xx — back off
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return ""


def _question_of(row: dict) -> str:
    """Reconstruct the user question from chat messages, falling back
    to a string field if present."""
    if "question" in row:
        return row["question"]
    msgs = row.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    return user


# ---------- modes ----------
def pointwise(args):
    client = _client()
    rows = [json.loads(l) for l in open(args.preds) if l.strip()]
    scores = []
    for row in rows:
        prompt = POINTWISE_RUBRIC.format(
            question=_question_of(row),
            expected=row.get("expected", ""),
            answer=row.get("pred", ""),
        )
        out = _ask(client, args.judge, prompt)
        m = re.search(r"SCORE:\s*([\d.]+)", out)
        score = float(m.group(1)) if m else None
        scores.append({"i": row.get("i"), "score": score, "raw": out})
    valid = [s["score"] for s in scores if s["score"] is not None]
    summary = {
        "judge": args.judge,
        "n": len(scores),
        "mean_score": round(sum(valid) / max(len(valid), 1), 3),
        "missing": len(scores) - len(valid),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": scores}, f, indent=2)
    print(json.dumps(summary, indent=2))


def pairwise(args):
    client = _client()
    a_rows = {r["i"]: r for r in (json.loads(l) for l in open(args.a) if l.strip())}
    b_rows = {r["i"]: r for r in (json.loads(l) for l in open(args.b) if l.strip())}
    common = sorted(set(a_rows) & set(b_rows))

    wins_a = wins_b = ties = 0
    detail = []
    for i in common:
        ra, rb = a_rows[i], b_rows[i]
        question = _question_of(ra) or _question_of(rb)
        # Run both orderings to control position bias.
        v1 = _ask(client, args.judge, PAIRWISE.format(
            question=question, expected=ra.get("expected", ""),
            a=ra.get("pred", ""), b=rb.get("pred", "")))
        v2 = _ask(client, args.judge, PAIRWISE.format(
            question=question, expected=ra.get("expected", ""),
            a=rb.get("pred", ""), b=ra.get("pred", "")))
        c1 = _verdict(v1)
        c2 = _verdict(v2)
        # Map v2's perspective back: A in v2 == B in v1.
        if c1 == "A" and c2 == "B":
            verdict = "A"; wins_a += 1
        elif c1 == "B" and c2 == "A":
            verdict = "B"; wins_b += 1
        else:
            verdict = "TIE"; ties += 1
        detail.append({"i": i, "verdict": verdict, "v1": v1, "v2": v2})

    n = len(common)
    summary = {
        "judge": args.judge,
        "n": n,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "win_rate_a": round(wins_a / max(n, 1), 3),
        "win_rate_b": round(wins_b / max(n, 1), 3),
        "tie_rate": round(ties / max(n, 1), 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": detail}, f, indent=2)
    print(json.dumps(summary, indent=2))


def _verdict(text: str) -> str:
    head = text.strip().splitlines()[0].upper()
    for tag in ("A", "B", "TIE"):
        if tag in head.split():
            return tag
    return "TIE"


def faithful(args):
    client = _client()
    preds = [json.loads(l) for l in open(args.preds) if l.strip()]
    grounded = {}
    for line in open(args.grounded):
        if not line.strip():
            continue
        g = json.loads(line)
        grounded[g["i"]] = g

    yes = no = missing = 0
    detail = []
    for row in preds:
        g = grounded.get(row.get("i"))
        if g is None:
            missing += 1
            continue
        out = _ask(client, args.judge, FAITHFUL.format(
            question=_question_of(g),
            truth=g["truth"],
            answer=row.get("pred", ""),
        ))
        first = out.strip().splitlines()[0].upper()
        if first.startswith("YES"):
            yes += 1; v = "YES"
        elif first.startswith("NO"):
            no += 1; v = "NO"
        else:
            missing += 1; v = "?"
        detail.append({"i": row.get("i"), "verdict": v, "raw": out})

    total = yes + no
    summary = {
        "judge": args.judge,
        "n": total,
        "faithful": yes,
        "hallucinated": no,
        "hallucination_rate": round(no / max(total, 1), 3),
        "missing": missing,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": detail}, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pointwise")
    p.add_argument("--preds", required=True)
    p.add_argument("--judge", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--out", required=True)
    p.set_defaults(func=pointwise)

    p = sub.add_parser("pairwise")
    p.add_argument("--a", required=True, help="baseline preds.jsonl")
    p.add_argument("--b", required=True, help="finetuned preds.jsonl")
    p.add_argument("--judge", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--out", required=True)
    p.set_defaults(func=pairwise)

    p = sub.add_parser("faithful")
    p.add_argument("--preds", required=True)
    p.add_argument("--grounded", required=True,
                   help="JSONL with {i, question, truth}")
    p.add_argument("--judge", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--out", required=True)
    p.set_defaults(func=faithful)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
