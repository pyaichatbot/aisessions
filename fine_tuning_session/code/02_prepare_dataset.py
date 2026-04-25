"""
Prepare and validate a chat-format dataset for QLoRA fine-tuning.

Input: raw.jsonl (one object per line) with `messages` and `expected`.
Output: data/{train,val,test}.jsonl with a stratified split.

Usage:
    python 02_prepare_dataset.py --raw raw.jsonl --out data/ \
        --train 0.8 --val 0.1 --test 0.1 --seed 42 \
        --stratify-key task

The script:
- Validates each row's schema.
- Strips trailing whitespace, normalizes quotes.
- Stratifies by `--stratify-key` (label or task name).
- Refuses to overlap train/val/test on `messages` content hash.
- Prints class-balance summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify-key", default="task")
    return p.parse_args()


SMART_QUOTES = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-",
}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def validate(row: dict, i: int) -> None:
    if "messages" not in row or not isinstance(row["messages"], list):
        raise ValueError(f"row {i}: missing or bad 'messages'")
    for m in row["messages"]:
        if m.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"row {i}: bad role {m.get('role')!r}")
        if "content" not in m:
            raise ValueError(f"row {i}: message missing content")
    if "expected" not in row:
        raise ValueError(f"row {i}: missing 'expected'")


def content_hash(row: dict) -> str:
    h = hashlib.sha256()
    for m in row["messages"]:
        h.update(m["role"].encode())
        h.update(b"\x00")
        h.update(m["content"].encode())
        h.update(b"\x01")
    return h.hexdigest()


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    with open(args.raw) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            validate(row, i)
            for m in row["messages"]:
                m["content"] = normalize(m["content"])
            row["expected"] = normalize(row["expected"])
            h = content_hash(row)
            if h in seen:
                continue            # drop dupes
            seen.add(h)
            row["_hash"] = h
            rows.append(row)

    rng = random.Random(args.seed)
    by_strat = defaultdict(list)
    for r in rows:
        by_strat[r.get(args.stratify_key, "_")].append(r)

    train, val, test = [], [], []
    for k, group in by_strat.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # Sanity: ensure no hash overlap (defensive — stratified split shouldn't
    # produce overlap, but a bug in stratify would).
    h_train = {r["_hash"] for r in train}
    h_val = {r["_hash"] for r in val}
    h_test = {r["_hash"] for r in test}
    assert not (h_train & h_val), "train/val overlap"
    assert not (h_train & h_test), "train/test overlap"
    assert not (h_val & h_test), "val/test overlap"

    for split, items in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{split}.jsonl"
        with open(path, "w") as f:
            for r in items:
                r = {k: v for k, v in r.items() if k != "_hash"}
                f.write(json.dumps(r) + "\n")
        labels = Counter(r.get(args.stratify_key, "_") for r in items)
        print(f"{split}: n={len(items):5d}  classes={dict(labels)}")

    print(f"wrote {out}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    main()
