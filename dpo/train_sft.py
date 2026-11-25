"""Completion-only LoRA SFT trainer in native bfloat16 PyTorch + PEFT.

Trains either:
  - sft_validity : matched prompt distribution, uniform valid target (Coupon-Collector ablation)
  - sft_history  : matched prompt distribution, uncovered valid target (MLE history baseline)
with identical architecture, LoRA rank, learning rate, batch size, and step count.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import argparse
import json
import math
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo.common import format_chat_prompt


class CompletionSFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 2560):
        self.samples = []
        rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        eos = tokenizer.eos_token or ""
        skipped = 0
        for r in rows:
            p_str = format_chat_prompt(tokenizer, r["prompt"])
            c_str = r["completion"] + eos
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            c_ids = tokenizer.encode(c_str, add_special_tokens=False)
            if len(p_ids) + len(c_ids) > max_length:
                # Truncate oldest prompt tokens from the middle only if ever exceeding max_length
                keep_p = max_length - len(c_ids)
                if keep_p < 64:
                    skipped += 1
                    continue
                p_ids = p_ids[:keep_p]
            input_ids = p_ids + c_ids
            labels = [-100] * len(p_ids) + c_ids
            self.samples.append((input_ids, labels))
        print(f"[CompletionSFTDataset] loaded={len(self.samples)} skipped={skipped} from {jsonl_path.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_completion_batch(batch, pad_token_id: int):
    max_len = max(len(x[0]) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, lbs) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(lbs, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def completion_only_ce_loss(peft_model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Project hidden states through lm_head ONLY at non-masked completion positions.
    Eliminates the 152k-vocab logit allocation on ~2,000 prompt tokens (saves >10 GB VRAM)."""
    causal_lm = peft_model.get_base_model()
    backbone_out = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_hidden = backbone_out.last_hidden_state[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    active_hidden = shift_hidden[mask]
    active_labels = shift_labels[mask]
    logits = causal_lm.lm_head(active_hidden).float()
    return F.cross_entropy(logits, active_labels)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bucketed_batches(samples: list, batch_size: int, mega_mult: int, rng: random.Random) -> list:
    """Shuffle dataset, group into mega-batches, sort within mega-batch by sequence length to
    eliminate padding waste, then shuffle the resulting micro-batches."""
    pool = list(samples)
    rng.shuffle(pool)
    mega = batch_size * mega_mult
    batches = []
    for i in range(0, len(pool), mega):
        chunk = sorted(pool[i : i + mega], key=lambda x: len(x[0]))
        for j in range(0, len(chunk), batch_size):
            batches.append(chunk[j : j + batch_size])
    rng.shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing if <12GB VRAM")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU is required for SFT training"
    torch.cuda.set_per_process_memory_fraction(0.78, 0)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = CompletionSFTDataset(args.data, tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    num_micro_per_epoch = math.ceil(len(ds) / args.batch_size)
    steps_per_epoch = math.ceil(num_micro_per_epoch / args.grad_accum)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(5, int(0.06 * total_steps))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    opt_step = 0
    micro = 0
    running_loss = 0.0
    log_history: List[dict] = []
    full_epochs = int(math.ceil(args.epochs))

    for epoch in range(full_epochs):
        if opt_step >= total_steps:
            break
        raw_batches = make_bucketed_batches(ds.samples, args.batch_size, mega_mult=16, rng=rng)
        for raw_b in raw_batches:
            if opt_step >= total_steps:
                break
            batch = collate_completion_batch(raw_b, tokenizer.pad_token_id)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                raw_loss = completion_only_ce_loss(
                    model, batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                loss = raw_loss / args.grad_accum
            loss.backward()
            running_loss += float(raw_loss.detach().cpu())
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.memory_reserved() > 9 * (1024 ** 3):
                    torch.cuda.empty_cache()
                opt_step += 1

                if opt_step % 20 == 0 or opt_step == 1 or opt_step == total_steps:
                    avg_l = running_loss / micro
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    cur_lr = scheduler.get_last_lr()[0]
                    rec = {
                        "step": opt_step,
                        "total_steps": total_steps,
                        "loss": round(avg_l, 5),
                        "lr": round(cur_lr, 7),
                        "vram_gb": round(vram_gb, 2),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    log_history.append(rec)
                    print(
                        f"[SFT] step {opt_step:4d}/{total_steps} | loss={avg_l:.4f} | "
                        f"lr={cur_lr:.2e} | vram={vram_gb:.2f}GB | t={rec['elapsed_s']:.0f}s",
                        flush=True,
                    )
                    running_loss = 0.0
                    micro = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.base_model,
        "data": str(args.data),
        "n_samples": len(ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_steps": opt_step,
        "elapsed_s": round(time.time() - t0, 1),
        "log_history": log_history,
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[SFT] saved adapter to {args.output_dir} in {meta['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
