# Citations

Every claim in this session traces to one of these. Read the abstract
of each at minimum. Read the full paper of QLoRA, LoRA, and MT-Bench.

## Architecture

- **Vaswani et al., 2017** — *Attention Is All You Need.*
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
  The original Transformer.

- **Xiong et al., 2020** — *On Layer Normalization in the Transformer
  Architecture.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745).
  Pre-LN vs Post-LN; Pre-LN is more stable.

- **Zhang & Sennrich, 2019** — *Root Mean Square Layer Normalization.*
  [arXiv:1910.07467](https://arxiv.org/abs/1910.07467). RMSNorm
  (LLaMA uses this).

- **Su et al., 2021** — *RoFormer: Enhanced Transformer with Rotary
  Position Embedding.*
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864). RoPE.

- **Shazeer, 2020** — *GLU Variants Improve Transformer.*
  [arXiv:2002.05202](https://arxiv.org/abs/2002.05202). SwiGLU.

- **Ainslie et al., 2023** — *GQA: Training Generalized Multi-Query
  Transformer Models from Multi-Head Checkpoints.*
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). Grouped-query
  attention.

- **Dao et al., 2022** — *FlashAttention: Fast and Memory-Efficient
  Exact Attention with IO-Awareness.*
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

- **Dao, 2023** — *FlashAttention-2: Faster Attention with Better
  Parallelism and Work Partitioning.*
  [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).

## Tokenization

- **Sennrich et al., 2016** — *Neural Machine Translation of Rare
  Words with Subword Units.*
  [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). BPE.

- **Kudo & Richardson, 2018** — *SentencePiece: A simple and language
  independent subword tokenizer and detokenizer for Neural Text
  Processing.*
  [arXiv:1808.06226](https://arxiv.org/abs/1808.06226).

- **OpenAI tiktoken** — https://github.com/openai/tiktoken.
  Reference BPE for GPT-3.5 / 4 / 4o.

## Adaptation methods

- **Brown et al., 2020** — *Language Models are Few-Shot Learners.*
  [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). GPT-3,
  in-context learning.

- **Lewis et al., 2020** — *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.*
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401). RAG.

- **Wei et al., 2021** — *Finetuned Language Models Are Zero-Shot
  Learners.* [arXiv:2109.01652](https://arxiv.org/abs/2109.01652).
  FLAN; instruction tuning generalizes.

- **Ouyang et al., 2022** — *Training language models to follow
  instructions with human feedback.*
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). InstructGPT,
  RLHF.

- **Rafailov et al., 2023** — *Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model.*
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290). DPO.

- **Hu et al., 2021** — *LoRA: Low-Rank Adaptation of Large Language
  Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

- **Dettmers et al., 2023** — *QLoRA: Efficient Finetuning of
  Quantized LLMs.*
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). The paper
  this entire session orbits.

## Evaluation

- **Zheng et al., 2023** — *Judging LLM-as-a-Judge with MT-Bench
  and Chatbot Arena.*
  [arXiv:2306.05685](https://arxiv.org/abs/2306.05685). >80% judge-
  human agreement; documents position bias and verbosity bias.

- **Lin et al., 2022** — *TruthfulQA: Measuring How Models Mimic
  Human Falsehoods.*
  [arXiv:2109.07958](https://arxiv.org/abs/2109.07958).

- **Li et al., 2023** — *HaluEval: A Large-Scale Hallucination
  Evaluation Benchmark for Large Language Models.*
  [arXiv:2305.11747](https://arxiv.org/abs/2305.11747).

## Inference / serving

- **Kwon et al., 2023** — *Efficient Memory Management for Large
  Language Model Serving with PagedAttention.*
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). vLLM. 2–4×
  throughput vs FasterTransformer / Orca at same latency.

- **Frantar et al., 2022** — *GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers.*
  [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).

- **Lin et al., 2023** — *AWQ: Activation-aware Weight Quantization
  for LLM Compression and Acceleration.*
  [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).

- **Dettmers et al., 2022** — *LLM.int8(): 8-bit Matrix Multiplication
  for Transformers at Scale.*
  [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).

- **Leviathan et al., 2022** — *Fast Inference from Transformers via
  Speculative Decoding.*
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).

- **Chen et al., 2023** — *Accelerating Large Language Model Decoding
  with Speculative Sampling.*
  [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).

## Tools and libraries

- **HuggingFace `transformers`** — https://github.com/huggingface/transformers
- **HuggingFace `peft`** — https://github.com/huggingface/peft (LoRA, QLoRA)
- **HuggingFace `trl`** — https://github.com/huggingface/trl (SFTTrainer, DPO)
- **`bitsandbytes`** — https://github.com/TimDettmers/bitsandbytes (NF4, paged optimizers)
- **`vllm`** — https://github.com/vllm-project/vllm
- **`autoawq`** — https://github.com/casper-hansen/AutoAWQ (AWQ quantization)
