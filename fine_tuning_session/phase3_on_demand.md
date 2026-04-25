# Phase 3 — Learn on Demand

You do not study PyTorch upfront. You hit a problem, then learn the
exact thing that solves it. Each card below maps a symptom to root
causes and fixes, with citations.

---

## Card A — `CUDA out of memory` (OOM)

**Where it happens**: model loading, training step, or generation.

### Diagnosis

Print the deltas:

```python
import torch
torch.cuda.reset_peak_memory_stats()
# ... do the operation ...
print(torch.cuda.max_memory_allocated() / 1e9, "GB")
```

### Memory budget for QLoRA-7B fine-tune (bf16 compute, NF4 base)

| Component | Approx VRAM |
|-----------|-------------|
| Base weights (4-bit NF4) | ~3.5 GB |
| LoRA adapters + grads (bf16) | ~0.3 GB |
| Optimizer state (paged AdamW 8-bit) | ~0.3 GB |
| Activations (depends on `seq_len`, `batch`) | 4–10 GB |
| KV cache (eval + checkpoint) | 1–3 GB |

Total: 10–18 GB. Fits a T4 (16) or A10/3090 (24).

### Fixes, in order of impact

1. **Enable gradient checkpointing**:
   `model.gradient_checkpointing_enable()` or
   `TrainingArguments(gradient_checkpointing=True)`. Trades ~30% step
   time for ~3–4× activation memory savings.
2. **Reduce `max_seq_length`**: activations scale with seq². Trim from
   2048 → 1024 cuts a lot.
3. **Lower `per_device_train_batch_size` + raise
   `gradient_accumulation_steps`** to keep the same effective batch.
4. **Use paged optimizer**: `optim="paged_adamw_8bit"` (Dettmers et
   al., 2023, [arXiv:2305.14314]). Keeps optimizer state in CPU,
   pages on demand.
5. **Lower LoRA rank**: r=8 instead of r=16 halves adapter memory
   (small effect on a 7B model).
6. **Disable attention dropout** during fine-tune (it adds activations).
7. **Use FlashAttention 2** (Dao, 2023, [arXiv:2307.08691]):
   `attn_implementation="flash_attention_2"` in `from_pretrained`.
   Reduces attention activation memory and speeds up.

### Inference-time OOM

KV cache is the usual culprit at long context. Switch to a model with
**Grouped-Query Attention** (LLaMA-3, Mistral 7B+) which shrinks KV
cache by `n_q_heads / n_kv_heads`. Or use vLLM, whose PagedAttention
(Kwon et al., 2023, [arXiv:2309.06180]) eliminates KV-cache fragmentation.

---

## Card B — Training is unstable (loss spikes, NaN, no convergence)

### Diagnosis

Plot `train_loss` vs steps. What does it look like?

| Symptom | Likely cause |
|---------|-------------|
| Flat from step 0 | LR too low, or LoRA targeting wrong modules, or chat template mismatch |
| Spikes to NaN | LR too high, no grad clipping, mixed precision underflow |
| Drops then plateaus high | Underfitting — train longer or raise LR |
| Train loss drops, eval loss rises | Overfitting — stop earlier, more data, more dropout |
| Loss = 0 immediately | You are computing loss on the prompt as well as the response, *and* the prompt is identical across examples. Mask the prompt. |

### Fixes

1. **Learning rate**: typical QLoRA LR is `1e-4` to `3e-4`. Full
   fine-tune is `1e-5` to `5e-5`. Way different scales. Don't reuse FT
   defaults for LoRA.
2. **Warmup**: `warmup_ratio=0.03` to `0.1`. Linear or cosine. Without
   warmup, early Adam steps can blow up.
3. **Gradient clipping**: `max_grad_norm=1.0`. Catches any rare spike.
4. **Loss masking**: when training on chat data, set `labels = -100`
   on all tokens that belong to the prompt; only the assistant's reply
   contributes to the loss. Most data collators (`DataCollatorForCompletionOnlyLM`)
   do this. Verify by printing one batch.
5. **Mixed precision**: prefer `bf16=True` on Ampere+ (A10, A100,
   3090, 4090, H100). `fp16` underflows more easily; needs loss
   scaling.
6. **Verify chat template**: print the **exact tokenized string** the
   model sees. The string at fine-tune must equal the string at
   inference. Use `tokenizer.apply_chat_template(msgs, tokenize=False)`
   and eyeball it.

---

## Card C — Inference is too slow / too expensive

### Diagnosis

Three numbers determine throughput:

- **Tokens / second** (decode rate, decoder-only).
- **Concurrent requests** the server can batch.
- **Memory bandwidth** of the GPU (the actual bottleneck for 7B+
  decode).

### Fixes, in order of impact

1. **Switch from `transformers.generate()` to vLLM**. PagedAttention
   gives 2–4× throughput at same latency vs FT/Orca (Kwon et al.,
   2023, [arXiv:2309.06180]). For HF baseline the gap is much larger.
2. **Continuous batching**: vLLM and TGI do this automatically. Don't
   roll your own.
3. **INT4 weight quantization** (AWQ — Lin et al., 2023,
   [arXiv:2306.00978]; or GPTQ — Frantar et al., 2022,
   [arXiv:2210.17323]). Cuts weight memory ~4× → less bandwidth →
   faster decode + larger batch fits.
4. **Speculative decoding** (Leviathan et al., 2022,
   [arXiv:2211.17192]; Chen et al., 2023, [arXiv:2302.01318]).
   A small "draft" model proposes K tokens; the big model verifies in
   one forward pass. 2–3× speedup, lossless. vLLM supports it.
5. **FlashAttention 2** (Dao, 2023, [arXiv:2307.08691]) — already on
   by default in modern vLLM, but verify.
6. **Tensor parallelism** for big models (`--tensor-parallel-size 2`
   in vLLM). Splits weights across GPUs, reduces per-GPU bandwidth.
7. **Right-size the model**: a 7B fine-tune that beats a 70B base on
   your task is the cheapest win. Distillation > scaling.

### Cost calculation

```
$/1M tokens = (GPU $/hr) / (tokens/sec * 3600 / 1e6)
```

A10 ≈ $0.60–$1.00 / hr. If your fine-tuned 7B does 1500 tokens/sec on
INT4 vLLM: `$1.0 / (1500 * 3.6 / 1) = $0.18 / 1M tokens`. Compare to
GPT-4o at $2.50 / 1M input + $10 / 1M output. That is your "60% cost
reduction" claim, with arithmetic.

---

## Card D — Evals look great, production is bad

### Diagnosis

Eval/prod distribution drift. Check:

1. The exact **prompt template** the model sees in prod vs eval.
2. **Sampling settings**: `temperature`, `top_p`, `max_tokens`,
   `repetition_penalty`. Deterministic eval (`temperature=0`) followed
   by sampled prod (`temperature=0.7`) gives different behavior.
3. **System prompt** drift. Prod system prompt added a sentence the
   model never saw during fine-tune. It will react unpredictably.
4. **Long-tail inputs**: your eval set was sanitized. Prod has typos,
   non-English, emoji, code. Add adversarial samples.
5. **Adapter merge sanity**: did you actually merge the right
   adapter? Hash the merged weights and compare.

### Fixes

- Capture 100 real prod inputs, label them, add to eval. Re-run.
- Use the **same generation config** in eval and prod. Pin it.
- Canary deploy: 5% of traffic to fine-tuned model with side-by-side
  judge eval. Promote when judge prefers it on > 60% of cases with
  `p < 0.05`.

---

## Card E — Catastrophic forgetting

You fine-tune for SQL generation. Now the model can't answer "hi how
are you" coherently.

### Cause

Full or aggressive LoRA fine-tune overwrites general capability.

### Fixes

1. **Mix in 5–20% general instruction data** (e.g. samples from a
   public mixture like OpenHermes, Tulu, or your own held-out set).
   Citation: Wei et al., 2021, FLAN ([arXiv:2109.01652]) showed
   instruction-tuning generalizes when the mix is diverse.
2. **Lower LoRA rank** — less capacity to overwrite.
3. **Train fewer epochs**.
4. **Use adapters as adapters**: keep them separate, hot-swap per
   request, do not merge into base.

---

## When to stop reading and just run

Each of these cards points to a specific action. Try the action.
If it works, move on. If it doesn't, read the cited paper.

You will not learn PyTorch by reading. You will learn it by debugging.

---

[arXiv:2305.14314]: https://arxiv.org/abs/2305.14314
[arXiv:2307.08691]: https://arxiv.org/abs/2307.08691
[arXiv:2309.06180]: https://arxiv.org/abs/2309.06180
[arXiv:2306.00978]: https://arxiv.org/abs/2306.00978
[arXiv:2210.17323]: https://arxiv.org/abs/2210.17323
[arXiv:2211.17192]: https://arxiv.org/abs/2211.17192
[arXiv:2302.01318]: https://arxiv.org/abs/2302.01318
[arXiv:2109.01652]: https://arxiv.org/abs/2109.01652
