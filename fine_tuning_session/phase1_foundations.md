# Phase 1 — Minimal Foundations

3–5 days. No deep PyTorch theory. Just enough to build.

---

## 1.1 What is a Transformer (high level)

The Transformer is the architecture behind every modern LLM. Introduced
in *Attention Is All You Need* (Vaswani et al., 2017,
[arXiv:1706.03762]). Replaces RNN recurrence with **self-attention**.

### Core components

- **Self-attention**: each token computes a weighted average over all
  other tokens. Weights come from `softmax(QKᵀ/√d) V`. Q, K, V are
  linear projections of the input.
- **Multi-head**: run h attention heads in parallel, concat, project.
- **FFN**: a 2-layer MLP per position (e.g., 4× hidden dim).
- **Residual + LayerNorm**: stabilizes deep stacks.
- **Stack of N blocks**: GPT-3 has 96, LLaMA-3-8B has 32.

### Decoder-only LLMs

Most modern LLMs (GPT, LLaMA, Mistral, Qwen, Claude) are decoder-only:
each token attends only to itself and prior tokens (causal mask). The
output is a probability distribution over the vocabulary. You sample
the next token, append, repeat.

### Modern variants you will encounter

- **Pre-LayerNorm** (LayerNorm before sub-block) — more stable training
  than the original Post-LN (Xiong et al., 2020, [arXiv:2002.04745]).
- **RMSNorm** instead of LayerNorm (Zhang & Sennrich, 2019,
  [arXiv:1910.07467]) — used in LLaMA.
- **RoPE** rotary position embedding (Su et al., 2021,
  [arXiv:2104.09864]) — used in LLaMA, Mistral, Qwen.
- **SwiGLU** activation in FFN (Shazeer, 2020, [arXiv:2002.05202]) —
  used in LLaMA.
- **GQA** grouped-query attention (Ainslie et al., 2023,
  [arXiv:2305.13245]) — used in LLaMA-2 70B+, LLaMA-3, Mistral 7B+.
  Reduces KV-cache memory by sharing K/V across query heads.
- **FlashAttention** (Dao et al., 2022, [arXiv:2205.14135]) —
  IO-aware exact attention, lower memory and faster.
- **KV cache**: during autoregressive decoding, cache K and V for past
  tokens so each new token costs O(seq_len) not O(seq_len²).

### What you need to remember

> A decoder-only transformer is a stack of (self-attention + FFN)
> blocks. Self-attention lets each token mix information from prior
> tokens. The output projects to a vocabulary, you sample, repeat.

You do **not** need to derive backprop or implement attention from
scratch to fine-tune. You will revisit internals on-demand in Phase 3.

---

## 1.2 What is Tokenization

Models do not see text. They see token IDs. The tokenizer is a
deterministic function `text ↔ list[int]`.

### Algorithms

- **BPE — Byte-Pair Encoding** (Sennrich et al., 2016,
  [arXiv:1508.07909]). Start with characters. Iteratively merge the
  most frequent adjacent pair into a single token. Stop at vocab size.
- **WordPiece** (Schuster & Nakajima, 2012; used by BERT). Like BPE
  but merges based on likelihood, not frequency.
- **SentencePiece** (Kudo & Richardson, 2018, [arXiv:1808.06226]).
  Language-agnostic. Treats whitespace as a normal character (`▁`).
  Used by LLaMA-1, T5.
- **Byte-level BPE** (GPT-2, GPT-3, GPT-4): operates on bytes, so any
  Unicode is representable. No `<unk>` token.

### What current models use

| Model | Tokenizer | Vocab |
|-------|-----------|-------|
| GPT-4 | tiktoken `cl100k_base` (byte BPE) | 100,277 |
| GPT-4o, o1, o3 | tiktoken `o200k_base` | 200,019 |
| LLaMA-1, LLaMA-2 | SentencePiece BPE | 32,000 |
| LLaMA-3, LLaMA-3.1, LLaMA-3.2 | tiktoken-style BPE | 128,000 |
| Mistral | SentencePiece BPE | 32,000 |
| Qwen2.5 | tiktoken-style BPE | 151,936 |

(Sources: model cards on HuggingFace; OpenAI tiktoken repo.)

### Practical implications

- 1 token ≈ 0.75 English words on average (OpenAI tokenizer docs).
- Code, JSON, non-English text use **more** tokens per character.
- Cost and context length are measured in tokens, not characters.
- Special tokens (`<|im_start|>`, `<s>`, `[INST]`) frame the prompt.
  Use the **chat template** the model was trained with —
  `tokenizer.apply_chat_template(...)` in HuggingFace handles this.
- Mismatched template at fine-tune vs inference is the #1 silent bug.

---

## 1.3 Fine-tuning vs Prompting

Four levers, in order of cost and commitment.

### 1. Prompting (zero/few-shot, in-context learning)

GPT-3 paper (Brown et al., 2020, [arXiv:2005.14165]) showed that a
sufficiently large model can learn from examples in the prompt with no
weight updates. Zero cost beyond inference. First thing to try.

### 2. RAG (Retrieval-Augmented Generation)

Lewis et al., 2020, [arXiv:2005.11401]. Retrieve relevant docs at
inference, stuff into the prompt, generate. Best for **knowledge** that
changes (docs, policies, manuals). No fine-tune needed.

### 3. SFT — Supervised Fine-Tuning

Update weights on `(instruction, response)` pairs. Best for **stable
behavior or format** the base model can't reliably do via prompt
(specific JSON shape, domain jargon, tone, refusal style).

- Instruction tuning generalizes across tasks (Wei et al., 2021, FLAN,
  [arXiv:2109.01652]; Stanford Alpaca, 2023).
- Full fine-tune updates every parameter — expensive.
- LoRA / QLoRA fine-tune updates < 1% of parameters — cheap. See § 1.4.

### 4. Preference tuning (RLHF / DPO)

Goes beyond SFT by training on **preferences**, not just demonstrations.

- **RLHF** (Ouyang et al., 2022, InstructGPT, [arXiv:2203.02155]):
  train a reward model from human comparisons, then PPO against it.
- **DPO** (Rafailov et al., 2023, [arXiv:2305.18290]): closed-form
  alternative — no reward model, no RL. Trains directly on chosen vs
  rejected pairs. Now the default for most open-weight finetunes.

### When to fine-tune (decision rule)

| You need... | Try first |
|-------------|-----------|
| Better factual answers from your docs | RAG |
| A different tone, format, or refusal pattern | Few-shot, then SFT |
| A specialized output (SQL, function calls, medical codes) | SFT/QLoRA |
| Smaller cheaper model that matches a frontier model on your task | QLoRA + distillation |
| Capability the base model lacks entirely | Probably not fine-tuning. Use a better base model. |

### Anti-pattern

Fine-tuning to inject **knowledge** that is in your documents.
RAG is almost always better, cheaper, and easier to update.

---

## 1.4 What is LoRA (and QLoRA)

You will fine-tune a 7B-parameter model on a single consumer GPU.
This is only possible because of LoRA + 4-bit quantization.

### LoRA — Low-Rank Adaptation

Hu et al., 2021, [arXiv:2106.09685].

For each weight matrix `W ∈ ℝ^{d×k}` you want to adapt, **freeze W**
and learn a low-rank update:

```
W_new = W + ΔW    where  ΔW = B · A
                          B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r ≪ d
```

Typical `r = 8, 16, 32, 64`. Trainable params per matrix: `r·(d+k)`,
which is tiny vs `d·k`.

**Cited result**: "Compared to GPT-3 175B fine-tuned with Adam, LoRA
can reduce the number of trainable parameters by 10,000× and the GPU
memory requirement by 3×" (Hu et al., 2021).

At inference you can **merge** `B·A` into `W` and serve a single
matrix — zero overhead vs the base model. You can also keep adapters
separate and hot-swap them per request.

Standard target modules for transformer LLMs: `q_proj, k_proj, v_proj,
o_proj, gate_proj, up_proj, down_proj` (LLaMA-style names).

### QLoRA — Quantized LoRA

Dettmers et al., 2023, [arXiv:2305.14314]. Three innovations:

1. **4-bit NormalFloat (NF4)**: quantizes the frozen base weights to
   4 bits using a data-type optimal for normally distributed weights.
2. **Double Quantization**: quantizes the quantization constants
   themselves, saving ~0.37 bits per parameter.
3. **Paged Optimizers**: uses NVIDIA unified memory to page optimizer
   states between CPU and GPU, avoiding OOM on memory spikes.

**Cited results from the QLoRA paper**:

- "QLoRA reduces the average memory requirements of finetuning a 65B
  parameter model from > 780 GB of GPU memory to < 48 GB without
  degrading the runtime or predictive performance compared to a 16-bit
  fully finetuned baseline."
- "Our best model family, which we name Guanaco, outperforms all
  previous openly released models on the Vicuna benchmark, reaching
  99.3% of the performance level of ChatGPT while only requiring 24
  hours of finetuning on a single GPU."

### What you actually do in practice

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # < 1% of total
```

That is the entire conceptual surface area you need. The library
(`peft` + `transformers` + `bitsandbytes`) does the rest. See
`code/03_qlora_train.py` for the full training loop.

---

## What you should be able to say after Phase 1

- "Decoder-only transformer is a stack of self-attention + FFN blocks."
- "Tokenization is BPE-family; mismatched chat templates break training."
- "Prompting < RAG < SFT < preference tuning, in that order of effort."
- "LoRA learns a low-rank ΔW; QLoRA also quantizes the frozen base to
   4-bit NF4 with double-quant + paged optimizers."

If you can say these four sentences and explain each in two minutes,
move to Phase 2.

---

[arXiv:1706.03762]: https://arxiv.org/abs/1706.03762
[arXiv:2002.04745]: https://arxiv.org/abs/2002.04745
[arXiv:1910.07467]: https://arxiv.org/abs/1910.07467
[arXiv:2104.09864]: https://arxiv.org/abs/2104.09864
[arXiv:2002.05202]: https://arxiv.org/abs/2002.05202
[arXiv:2305.13245]: https://arxiv.org/abs/2305.13245
[arXiv:2205.14135]: https://arxiv.org/abs/2205.14135
[arXiv:1508.07909]: https://arxiv.org/abs/1508.07909
[arXiv:1808.06226]: https://arxiv.org/abs/1808.06226
[arXiv:2005.14165]: https://arxiv.org/abs/2005.14165
[arXiv:2005.11401]: https://arxiv.org/abs/2005.11401
[arXiv:2109.01652]: https://arxiv.org/abs/2109.01652
[arXiv:2203.02155]: https://arxiv.org/abs/2203.02155
[arXiv:2305.18290]: https://arxiv.org/abs/2305.18290
[arXiv:2106.09685]: https://arxiv.org/abs/2106.09685
[arXiv:2305.14314]: https://arxiv.org/abs/2305.14314
