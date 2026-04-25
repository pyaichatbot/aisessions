"""
Compare baseline and fine-tuned runs. Produces report.md — the
artifact you bring to interviews.

Usage:
    python 06_compare.py \
        --baseline runs/baseline/metrics_baseline.json \
        --finetuned runs/ft1/metrics_finetuned.json \
        --hallu-base runs/baseline/judge_faithful.json \
        --hallu-ft   runs/ft1/judge_faithful.json \
        --bench-bf16 runs/ft1/bench_bf16.json \
        --bench-int4 runs/ft1/bench_int4.json \
        --api-cost-per-1m 3.00 \
        --gpu-cost-per-hour 1.00 \
        --out runs/ft1/report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text()) if p else None


def pct(new, old):
    if old in (None, 0):
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


def cost_per_1m(gpu_per_hr, tokens_per_sec):
    if not tokens_per_sec:
        return None
    return gpu_per_hr / (tokens_per_sec * 3600 / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--hallu-base")
    ap.add_argument("--hallu-ft")
    ap.add_argument("--bench-bf16")
    ap.add_argument("--bench-int4")
    ap.add_argument("--api-cost-per-1m", type=float, default=None,
                    help="API $/1M output tokens for the baseline model")
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None,
                    help="$/hr for the GPU you serve on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load(args.baseline)
    ft = load(args.finetuned)
    hb = load(args.hallu_base)
    hf = load(args.hallu_ft)
    bbf = load(args.bench_bf16)
    bint = load(args.bench_int4)

    base_acc = base.get("accuracy")
    ft_acc = ft.get("accuracy")

    base_hallu = hb["summary"]["hallucination_rate"] if hb else None
    ft_hallu = hf["summary"]["hallucination_rate"] if hf else None

    bf16_tps = bbf.get("tokens_per_sec") if bbf else None
    int4_tps = bint.get("tokens_per_sec") if bint else None

    api_cost = args.api_cost_per_1m
    self_bf16 = cost_per_1m(args.gpu_cost_per_hour, bf16_tps) if args.gpu_cost_per_hour else None
    self_int4 = cost_per_1m(args.gpu_cost_per_hour, int4_tps) if args.gpu_cost_per_hour else None

    lines = []
    lines.append("# Fine-tune Report\n")
    lines.append(f"- Base model: `{base.get('model')}`")
    lines.append(f"- Fine-tuned: `{ft.get('model')}`")
    lines.append(f"- Test set: n = {ft.get('n', '?')}\n")

    lines.append("## Quality\n")
    lines.append("| Metric | Baseline | Fine-tuned | Δ (relative) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Accuracy | {base_acc:.3f} | {ft_acc:.3f} | {pct(ft_acc, base_acc)} |"
    )
    if base_hallu is not None and ft_hallu is not None:
        lines.append(
            f"| Hallucination rate | {base_hallu:.3f} | {ft_hallu:.3f} | "
            f"{pct(ft_hallu, base_hallu)} |"
        )
    lines.append("")

    if bf16_tps or int4_tps:
        lines.append("## Throughput (vLLM)\n")
        lines.append("| Mode | tokens/sec |")
        lines.append("|---|---|")
        if bf16_tps:
            lines.append(f"| bf16 | {bf16_tps:.1f} |")
        if int4_tps:
            lines.append(f"| INT4 (AWQ) | {int4_tps:.1f} |")
        if bf16_tps and int4_tps:
            lines.append(
                f"| Δ INT4 vs bf16 | {pct(int4_tps, bf16_tps)} |"
            )
        lines.append("")

    if api_cost or self_bf16 or self_int4:
        lines.append("## Cost ($/1M output tokens)\n")
        lines.append("| Channel | $/1M |")
        lines.append("|---|---|")
        if api_cost is not None:
            lines.append(f"| Base API | ${api_cost:.2f} |")
        if self_bf16 is not None:
            lines.append(f"| Self-host bf16 | ${self_bf16:.3f} |")
        if self_int4 is not None:
            lines.append(f"| Self-host INT4 | ${self_int4:.3f} |")
        if api_cost and self_int4:
            lines.append(
                f"| Δ self-host INT4 vs API | {pct(self_int4, api_cost)} |"
            )
        lines.append("")

    lines.append("## Interview-ready claims\n")
    lines.append(
        f"- Fine-tuned with QLoRA on `{ft.get('model')}`, single GPU."
    )
    if base_acc is not None and ft_acc is not None:
        lines.append(
            f"- Accuracy: {base_acc:.2f} → {ft_acc:.2f} "
            f"({pct(ft_acc, base_acc)})."
        )
    if base_hallu is not None and ft_hallu is not None:
        lines.append(
            f"- Hallucination rate: {base_hallu:.2f} → {ft_hallu:.2f} "
            f"({pct(ft_hallu, base_hallu)})."
        )
    if api_cost and self_int4:
        lines.append(
            f"- Inference cost: ${api_cost:.2f} → ${self_int4:.3f} per 1M "
            f"tokens ({pct(self_int4, api_cost)}) using INT4 AWQ on vLLM."
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
