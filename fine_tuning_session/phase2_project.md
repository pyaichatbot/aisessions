# Phase 2 — Build the Project

The project that makes you hireable. End-to-end QLoRA fine-tune with
measured improvement. 2–3 weeks part-time.

---

## Choose your task

The task must satisfy three criteria:

1. **Programmatic eval**: you can compute accuracy, F1, exact-match,
   or judge-score automatically. No "looks good to me."
2. **Clear input/output format**: a base model can be prompted into the
   right shape, but with high error rate.
3. **Achievable data**: 500–2000 high-quality samples.

### Good candidates

| Task | Eval metric | Why |
|------|------------|-----|
| NL → SQL on your DB schema | Execution match (run query, compare rows) | Hard constraint, easy to score |
| Customer support intent classification | Macro-F1 vs labeled set | Simple, very common |
| Domain Q&A (legal/medical/finance) with grounded answers | LLM judge: faithful + correct | Cite-able hallucination metric |
| Function-call / tool-call generation | Schema match + arg correctness | JSON-shaped, deterministic |
| Code review comment generation | Pairwise LLM judge vs base | Nuanced quality, judge works well |

### Bad candidates

- "General chatbot personality" — no objective metric.
- "Make it smarter" — undefined.
- Tasks needing fresh knowledge (use RAG, not fine-tuning).

---

## Step 1 — Baseline

Before you fine-tune anything, run the **untouched base model** on
your eval set. This number is the bar you must beat.

`code/01_baseline.py` does this. Pick a strong open base (e.g.
`mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct`,
`meta-llama/Meta-Llama-3.1-8B-Instruct`) or a frontier API.

Record `metrics_baseline.json`:

```json
{
  "model": "Mistral-7B-Instruct-v0.3",
  "n": 200,
  "accuracy": 0.68,
  "hallucination_rate": 0.31,
  "tokens_in": 11240,
  "tokens_out": 4180,
  "p50_latency_ms": 850,
  "cost_per_1k": 0.00
}
```

> Until you have a real baseline number, you cannot claim "improved
> from X → Y". Most candidates skip this and lose credibility.

---

## Step 2 — Dataset

500–2000 `(instruction, response)` pairs. Quality > quantity.

### Format

JSONL, one example per line, in OpenAI/HF chat format:

```jsonl
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### Sources

- **Hand-curated** from your domain (best signal).
- **LLM-distilled**: prompt a frontier model with your system prompt
  and seed examples; have it generate more. Filter by hand. (Stanford
  Alpaca approach, 2023.)
- **Existing labeled data** reformatted into chat turns.

### Splits

80% train / 10% val / 10% test. Keep `test.jsonl` **untouched until
the final eval**. Do not look at it. Do not tune on it.

### Common dataset bugs

- Inconsistent system prompt across examples → model learns drift.
- Train and test answers come from the same source LLM → inflated
  metrics. Have humans audit the test set.
- Class imbalance (90% one label) → model learns to predict majority.
  Stratify your splits.
- Trailing whitespace, BOM, smart quotes → tokenization shifts.

`code/02_prepare_dataset.py` does the split, schema check, and chat-
template-aware tokenization.

---

## Step 3 — QLoRA Fine-Tune

This is the actual fine-tune. Citation: Dettmers et al., 2023,
[arXiv:2305.14314]; Hu et al., 2021, [arXiv:2106.09685].

`code/03_qlora_train.py` is a complete trainer. Key knobs:

```python
# Quantization (the Q in QLoRA)
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat-4
    bnb_4bit_use_double_quant=True,       # double quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA
LoraConfig(
    r=16,                                  # 8–64 typical
    lora_alpha=32,                         # rule of thumb: 2 * r
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training (HF TrainingArguments)
TrainingArguments(
    learning_rate=2e-4,                    # higher than full FT (LoRA tolerates)
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,         # effective batch 16
    gradient_checkpointing=True,
    bf16=True,
    optim="paged_adamw_8bit",              # paged optimizer (the QLoRA paper)
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    load_best_model_at_end=True,
)
```

### What you watch during training

- **train_loss** dropping smoothly. If flat → LR too low. If NaN → LR
  too high or no gradient clipping.
- **eval_loss** should track train_loss for a while, then diverge.
  When eval_loss starts climbing, you are overfitting — stop earlier.
- **GPU mem**: should be stable. Spikes mean optimizer is paging.
- **Tokens/sec**: regression here means something is wrong (e.g.
  gradient_checkpointing got disabled).

### Output

Adapter weights — small (~50–200 MB for a 7B base). The base model is
unchanged on disk. You merge or serve adapters on top.

---

## Step 4 — Evaluate (LLM-as-judge)

Citation: Zheng et al., 2023, *Judging LLM-as-a-Judge with MT-Bench
and Chatbot Arena* ([arXiv:2306.05685]). The paper shows that
GPT-4-class judges achieve **>80% agreement with human preferences**,
making them a viable proxy.

### Two judging modes

**Pointwise**: judge scores each answer 1–10 against a rubric.
Simple, but score scale drifts.

**Pairwise**: judge picks A vs B (or tie). More robust. Use this for
"baseline vs fine-tuned" comparisons.

### Critical pitfalls (from the MT-Bench paper)

1. **Position bias**: judges prefer the first option. Mitigation: run
   each pair twice with order flipped, count win only if both agree.
2. **Verbosity bias**: judges prefer longer answers. Mitigation:
   length-control your generations; include in the rubric.
3. **Self-preference bias**: a judge prefers outputs from its own
   family. Mitigation: use a different model family as judge than as
   base.

### Faithfulness / hallucination eval

For grounded Q&A, build a small set of `(question, ground_truth)`
pairs. Ask the judge:

> Does the answer contain only facts that are entailed by the
> ground truth? Reply EXACTLY YES or NO.

`hallucination_rate = (# of NO) / (# total)`.

Run this on baseline and fine-tuned. The delta is your "hallucination
reduced by N%" claim.

`code/04_eval_llm_judge.py` runs both pointwise and pairwise modes,
shuffles order, and writes `metrics_finetuned.json`.

---

## Step 5 — Deploy

Two outputs from this step:

### A. Merge adapters into base

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model = model.merge_and_unload()
model.save_pretrained("./merged")
tokenizer.save_pretrained("./merged")
```

Now `./merged` is a normal HF model. No PEFT runtime dependency.

### B. Serve with vLLM

vLLM (Kwon et al., 2023, [arXiv:2309.06180]) introduces
**PagedAttention** — managing the KV cache like virtual memory. The
paper reports **2–4× throughput** at the same latency vs FasterTransformer
and Orca, and much higher vs naive HuggingFace `generate`.

```bash
# bf16 baseline
vllm serve ./merged --port 8000

# INT4 quantization (AWQ format, prepared offline)
vllm serve ./merged-awq --quantization awq --port 8000
```

### Quantization for inference

- **AWQ** — Activation-aware Weight Quantization (Lin et al., 2023,
  [arXiv:2306.00978]). 4-bit weights, faster than GPTQ.
- **GPTQ** (Frantar et al., 2022, [arXiv:2210.17323]). 4-bit, popular,
  well supported.
- **bitsandbytes 4-bit** at inference is also possible but slower than
  AWQ/GPTQ on vLLM.

Quantization at inference reduces VRAM ~4× (fp16 → int4) and typically
improves throughput because memory bandwidth is the bottleneck. Some
accuracy loss — measure on **your** eval set, not on a generic bench.

---

## Step 6 — Compare

`code/06_compare.py` reads both `metrics_*.json` files and writes
`report.md`:

```
                           BASELINE       FINE-TUNED      Δ
accuracy                   0.68           0.89            +30.9%
hallucination_rate         0.31           0.18            -41.9%
p50_latency_ms (vLLM bf16) 220            230             +4.5%
p50_latency_ms (vLLM int4) —              140             -36.4% vs baseline
$/1M tokens (API base)     $3.00          —
$/1M tokens (self-host)    —              $1.20           -60.0%
```

(Numbers above are illustrative templates. Yours will differ.)

These four rows are **the four interview claims**, with evidence.

---

## Honesty checklist

Before quoting numbers in an interview, verify:

- [ ] Test set was held out. Not seen during training or LLM judging
      prompt design.
- [ ] Same eval set, same prompt template, same decoding settings
      across baseline and fine-tuned.
- [ ] LLM judge was a **different** model family from base, with
      shuffled order and length-controlled generations.
- [ ] Hallucination eval used ground-truth references, not vibes.
- [ ] Cost numbers compare apples to apples (per 1M tokens, same
      throughput target).
- [ ] You can produce the artifact (`report.md`, repo, tensorboard,
      W&B run) on demand in a screen-share interview.

If you can check all six, your claims survive scrutiny.

---

[arXiv:2106.09685]: https://arxiv.org/abs/2106.09685
[arXiv:2305.14314]: https://arxiv.org/abs/2305.14314
[arXiv:2306.05685]: https://arxiv.org/abs/2306.05685
[arXiv:2309.06180]: https://arxiv.org/abs/2309.06180
[arXiv:2306.00978]: https://arxiv.org/abs/2306.00978
[arXiv:2210.17323]: https://arxiv.org/abs/2210.17323
