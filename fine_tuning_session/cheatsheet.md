# Interview Cheat-Sheet

The four claims this project lets you make. Each comes with the
evidence you must be ready to show.

---

## Claim 1 — "I fine-tuned a model using QLoRA"

**What it means**: 4-bit NF4 quantization of the base + LoRA adapters
on top, paged AdamW 8-bit optimizer, bf16 compute dtype.

**Evidence to keep**:

- `code/03_qlora_train.py` — the trainer.
- `adapter_config.json` from your run (shows r, alpha, target modules).
- W&B / TensorBoard run with loss curves.
- One screenshot: `nvidia-smi` during training showing < 24 GB VRAM
  on a single GPU.

**Likely follow-up questions and answers**:

- *Why QLoRA over full fine-tuning?* "Full fine-tune of a 7B in fp16
  needs ~80 GB just for weights+grads+optimizer. QLoRA needs <16 GB.
  On 65B the gap is 780 GB → 48 GB. Same task perf in the paper."
  (Dettmers et al., 2023, [arXiv:2305.14314].)
- *Why r=16?* "Empirically a sweet spot for 7B on small datasets;
  doubled if I see underfitting, halved if overfitting. The QLoRA
  paper used r=64 on 65B; lower works for smaller bases."
- *What target modules?* "All linear projections in attention and
  FFN: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
  Some authors only target `q_proj, v_proj` — that's faster but
  weaker."
- *Compute dtype?* "bf16 on Ampere+ — wider exponent range than fp16,
  no loss-scaling needed."

---

## Claim 2 — "Accuracy improved from X → Y"

**What it means**: same held-out test set, same prompt template, same
decoding config, baseline vs fine-tuned, computed automatically.

**Evidence to keep**:

- `data/test.jsonl` — never seen during training.
- `metrics_baseline.json` and `metrics_finetuned.json`.
- `report.md` from `code/06_compare.py`.

**Numbers**: write your real ones in
`Your numbers` § below. The 0.68 → 0.89 in the project README is a
template, not a claim.

**Likely follow-ups**:

- *How big was the test set?* Aim for ≥ 200 samples for stable F1.
- *Statistical significance?* Bootstrap-resample (1000×) the test set;
  report 95% CI on accuracy. If CIs overlap, the improvement is
  noise.
- *Did the test set leak?* "I generated training data and test data
  from disjoint sources. I hash-checked overlap. Test labels were
  human-audited."

---

## Claim 3 — "Hallucination reduced by N%"

**What it means**: on a `(question, ground_truth)` set, an LLM judge
marked answers YES/NO for faithfulness. Compute rate before/after.

**Evidence to keep**:

- `data/grounded_qa.jsonl` — questions with reference answers.
- `judge_prompts.md` — exact rubric you used.
- Both runs of the judge with order shuffled.
- The delta calculation.

**Likely follow-ups**:

- *Which judge?* "Claude Sonnet 4.x. Different family from base, so
  no self-preference bias. MT-Bench paper shows >80% agreement with
  humans for GPT-4 judges; same family of result holds for modern
  Claude judges." (Zheng et al., 2023, [arXiv:2306.05685].)
- *Position bias?* "Each pair judged twice with order flipped. Win
  counted only if both orderings agree. About 8% of pairs become
  ties, which is fine."
- *Verbosity bias?* "I length-controlled both outputs — same
  `max_new_tokens`, same stop tokens. Judge rubric explicitly says
  brevity is not penalized."

---

## Claim 4 — "Inference cost reduced by Z% via INT4 quantization"

**What it means**: same fine-tuned model, served two ways. bf16 via
vLLM as one baseline, INT4 (AWQ or GPTQ) via vLLM as the optimized
case. Measure tokens/sec on a fixed batch and a fixed prompt mix.

**Evidence to keep**:

- `bench_bf16.json`, `bench_int4.json` — output of `code/05_serve_vllm.sh`
  + a short benchmark client.
- Same eval set re-run on quantized model to verify quality didn't
  collapse.

**Cost arithmetic** (memorize this formula):

```
$/1M tokens = (GPU $/hr) / (tokens/sec * 3600 / 1e6)
```

Example: A10 at $1/hr, 1500 tok/s INT4 → $0.18 per 1M tokens.

**Likely follow-ups**:

- *Why AWQ over GPTQ or bnb-4bit?* "AWQ has better throughput on
  vLLM and competitive accuracy. GPTQ is fine too. bnb-4bit at
  inference is slower — it's optimized for training memory, not
  inference speed." (Lin et al., 2023, [arXiv:2306.00978]; Frantar
  et al., 2022, [arXiv:2210.17323].)
- *Why vLLM over `transformers.generate`?* "PagedAttention manages
  the KV cache like virtual memory — no fragmentation, much higher
  effective batch size. The paper reports 2–4× throughput at same
  latency vs FasterTransformer/Orca." (Kwon et al., 2023,
  [arXiv:2309.06180].)
- *What about quality loss?* "Re-ran the test set on the INT4 model.
  Accuracy dropped by 0.X — within tolerance." Always check this.

---

## "How would you do this differently for production?" — bonus

A senior interviewer will ask. Have answers:

1. **Eval harness gates the deploy**: PR to model registry runs the
   test set and judge eval; merge blocked if accuracy regresses
   beyond a threshold.
2. **Shadow / canary**: 5% traffic to new model, log judge scores in
   real time, auto-rollback if they tank.
3. **Continuous data flywheel**: capture prod inputs, label the bad
   cases, fold into next training batch monthly.
4. **Adapters not merged**: keep adapters separate when you serve
   multiple variants per tenant.
5. **Cost SLO**: $/1M tokens and p95 latency are first-class metrics,
   tracked alongside accuracy. The team that ships cheap fast useful
   models wins.

---

## Your numbers

Fill in after your run. Bring this to interviews.

```
Task: ___________________________________________________
Base model: _____________________________________________
Dataset size: train=____ / val=____ / test=____
Training: r=____ alpha=____ epochs=____ effective_batch=____
GPU: ____________ wall_clock=____ hours

Baseline accuracy:        ______
Fine-tuned accuracy:      ______
Δ accuracy:               ______ (relative %)

Baseline hallucination:   ______
Fine-tuned hallucination: ______
Δ hallucination:          ______ (relative %)

bf16 tokens/sec (vLLM):   ______
INT4 tokens/sec (vLLM):   ______
$/1M tokens, base API:    $______
$/1M tokens, self-host:   $______
Δ cost:                   ______ (relative %)
```
