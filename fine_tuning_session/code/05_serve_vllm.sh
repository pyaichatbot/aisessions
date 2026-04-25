#!/usr/bin/env bash
# Serve the fine-tuned model with vLLM in two configurations:
# bf16 (correctness baseline) and INT4 AWQ (cost/throughput target).
# Cited: Kwon et al., 2023, arXiv:2309.06180; Lin et al., 2023, arXiv:2306.00978.
#
# Usage:
#   05_serve_vllm.sh merge runs/ft1 ./merged
#   05_serve_vllm.sh quantize ./merged ./merged-awq
#   05_serve_vllm.sh serve ./merged                # bf16
#   05_serve_vllm.sh serve ./merged-awq awq        # INT4 AWQ
#   05_serve_vllm.sh bench http://localhost:8000   # quick throughput check
set -euo pipefail

CMD="${1:?cmd required: merge|quantize|serve|bench}"

case "$CMD" in
  merge)
    ADAPTER="${2:?adapter dir}"
    OUT="${3:?merged out dir}"
    python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json, pathlib

cfg = json.loads(pathlib.Path("$ADAPTER/lora_args.json").read_text())
base = cfg["base"]
print(f"merging adapter from $ADAPTER on top of {base}")

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="cpu"
)
model = PeftModel.from_pretrained(model, "$ADAPTER")
model = model.merge_and_unload()
model.save_pretrained("$OUT", safe_serialization=True)
tok.save_pretrained("$OUT")
print("merged →", "$OUT")
PY
    ;;

  quantize)
    SRC="${2:?merged dir}"
    OUT="${3:?awq out dir}"
    # Requires `pip install autoawq`
    python - <<PY
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_config = {
  "zero_point": True, "q_group_size": 128,
  "w_bit": 4, "version": "GEMM",
}
print("loading FP16 model from $SRC")
model = AutoAWQForCausalLM.from_pretrained("$SRC", safetensors=True)
tok = AutoTokenizer.from_pretrained("$SRC", trust_remote_code=True)
print("quantizing (calibration ~5–15 min on 7B)")
model.quantize(tok, quant_config=quant_config)
model.save_quantized("$OUT")
tok.save_pretrained("$OUT")
print("AWQ-INT4 saved →", "$OUT")
PY
    ;;

  serve)
    MODEL="${2:?model dir}"
    QUANT="${3:-}"
    if [ -n "$QUANT" ]; then
      exec vllm serve "$MODEL" --quantization "$QUANT" \
        --port 8000 --max-model-len 4096
    else
      exec vllm serve "$MODEL" \
        --port 8000 --max-model-len 4096
    fi
    ;;

  bench)
    BASE="${2:-http://localhost:8000}"
    PROMPTS="${3:-bench_prompts.txt}"
    [ -f "$PROMPTS" ] || cat > "$PROMPTS" <<'EOF'
Summarize the second law of thermodynamics in two sentences.
Write a SQL query to find the top 5 customers by revenue this quarter.
List three differences between BPE and SentencePiece tokenization.
Translate to French: "The rain in Spain falls mainly on the plain."
Explain the difference between L1 and L2 regularization.
EOF
    python - <<PY
import json, time, urllib.request, statistics
prompts = [l.strip() for l in open("$PROMPTS") if l.strip()]
t_total = 0.0; tokens_total = 0
for p in prompts * 4:
    body = json.dumps({
        "model": "served",
        "messages": [{"role":"user","content": p}],
        "max_tokens": 128, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "$BASE/v1/chat/completions",
        data=body, headers={"Content-Type":"application/json"},
    )
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req).read())
    dt = time.time() - t0
    out = r["choices"][0]["message"]["content"]
    n = r["usage"]["completion_tokens"]
    tokens_total += n; t_total += dt
print(json.dumps({
    "tokens": tokens_total, "wall_seconds": round(t_total,2),
    "tokens_per_sec": round(tokens_total / max(t_total,1e-6), 2),
}, indent=2))
PY
    ;;

  *)
    echo "usage: $0 {merge|quantize|serve|bench}" >&2
    exit 2
    ;;
esac
