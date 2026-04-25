# Fine-Tuning Session — From Zero to Hireable

A lean learning track. Three phases. Build first, study on demand.

## The bet

You will be hired when you can honestly say:

- "I fine-tuned a model using QLoRA"
- "Accuracy improved from X → Y on my held-out set"
- "Hallucination reduced by N% measured by an LLM judge"
- "Inference cost reduced by Z% via INT4 quantization on vLLM"

This session gives you the **methodology and code** to fill in your
real X, Y, N, Z. Numbers come from your actual run, not from me.

## The three phases

| Phase | Time | Goal |
|-------|------|------|
| 1. Foundations | 3–5 days | Just enough theory to not be lost |
| 2. Build the project | 2–3 weeks | One end-to-end QLoRA fine-tune with measurable wins |
| 3. On-demand depth | forever | Learn PyTorch internals only when a problem forces you |

## Files in this folder

| File | What it gives you |
|------|------------------|
| `phase1_foundations.md` | Transformers, tokenization, fine-tune vs prompt, LoRA/QLoRA — citation-dense |
| `phase2_project.md` | The 6-step project, with code references |
| `phase3_on_demand.md` | Debug cards: OOM, instability, slow inference |
| `citations.md` | Bibliography. Every claim has an arXiv ID |
| `cheatsheet.md` | Interview statements + the evidence that backs each one |
| `code/` | Runnable Python skeletons: baseline, dataset, QLoRA train, judge eval, vLLM serve, compare |

## How to use this

1. Read `phase1_foundations.md` over 3–5 sessions of 1–2 hours.
2. Pick a domain task (see `phase2_project.md` § Choose your task).
3. Run `code/01_baseline.py` → `code/06_compare.py` in order.
4. Open `phase3_on_demand.md` only when you hit a wall.
5. Write up your numbers in `cheatsheet.md` § Your numbers.

## Hardware

QLoRA-7B fine-tune fits in 16 GB VRAM (T4, A10, RTX 3090/4090).
QLoRA-13B fits in 24 GB. QLoRA-65B fits in 48 GB (Dettmers et al.,
2023, [arXiv:2305.14314]). Colab T4 / Modal / Lambda / RunPod all
work. CPU-only does not work.
