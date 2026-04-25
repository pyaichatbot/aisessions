"""
QLoRA fine-tune. Cited: Dettmers et al., 2023, arXiv:2305.14314;
Hu et al., 2021, arXiv:2106.09685.

Usage:
    python 03_qlora_train.py \
        --base mistralai/Mistral-7B-Instruct-v0.3 \
        --train data/train.jsonl --val data/val.jsonl \
        --out runs/ft1 --epochs 3 --r 16

Outputs:
    runs/ft1/                   adapter weights + tokenizer
    runs/ft1/training_args.bin  exact config used
    runs/ft1/trainer_state.json loss curves

Designed to fit in <= 16 GB VRAM for a 7B base on a single GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Default LLaMA-style projection names. Override per-model if needed.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds_train = load_dataset("json", data_files=args.train, split="train")
    ds_val = load_dataset("json", data_files=args.val, split="train")

    def to_text(example):
        # Render messages with the model's chat template. The assistant
        # turn is included so the trainer learns to produce it. The
        # SFTTrainer + DataCollatorForCompletionOnlyLM will mask the
        # prompt portion of the loss when `response_template` is set.
        msgs = list(example["messages"])
        msgs.append({"role": "assistant", "content": example["expected"]})
        text = tok.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    ds_train = ds_train.map(to_text, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(to_text, remove_columns=ds_val.column_names)

    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))

    # Persist adapter config explicitly for reproducibility.
    with open(out / "lora_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
