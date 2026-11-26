"""Anchored Coverage-Weighted DPO trainer initialized from sft_history.

Objective per preference tuple (x, y+, y-, w):
  L(theta) = -w * log sigma( beta * [(log pi_theta(y+|x) - log pi_ref(y+|x))
                                   - (log pi_theta(y-|x) - log pi_ref(y-|x))] )
             - rpo_alpha * (log pi_theta(y+|x) / |y+|)

Reference log-probabilities under pi_ref = pi_{sft_history} are precomputed exactly in float32
before updating the LoRA parameters, requiring zero extra GPU memory for a second reference copy.
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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo.common import format_chat_prompt
from dpo.train_sft import collate_completion_batch, set_seed


def compute_sequence_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sum_logp, mean_logp) over non-masked completion tokens in float32,
    applying lm_head ONLY on non-masked completion positions to avoid prompt logit allocation."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone_out = causal_lm.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_hidden = backbone_out.last_hidden_state[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100

    bsz = input_ids.shape[0]
    sum_logps = []
    mean_logps = []
    for i in range(bsz):
        m_i = mask[i]
        h_i = shift_hidden[i][m_i]
        y_i = shift_labels[i][m_i]
        logits_i = causal_lm.lm_head(h_i).float()
        lp_i = F.log_softmax(logits_i, dim=-1).gather(-1, y_i.unsqueeze(-1)).squeeze(-1)
        s_i = lp_i.sum()
        sum_logps.append(s_i)
        mean_logps.append(s_i / max(1, int(m_i.sum().item())))
    return torch.stack(sum_logps), torch.stack(mean_logps)


class CoverageDPODataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 2560):
        self.items = []
        rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        eos = tokenizer.eos_token or ""
        skipped = 0
        for r in rows:
            if r["chosen"] == r["rejected"]:
                skipped += 1
                continue
            p_str = format_chat_prompt(tokenizer, r["prompt"])
            c_ids = tokenizer.encode(r["chosen"] + eos, add_special_tokens=False)
            r_ids = tokenizer.encode(r["rejected"] + eos, add_special_tokens=False)
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            max_comp = max(len(c_ids), len(r_ids))
            if len(p_ids) + max_comp > max_length:
                keep_p = max_length - max_comp
                if keep_p < 64:
                    skipped += 1
                    continue
                p_ids = p_ids[:keep_p]
            self.items.append({
                "c_input": p_ids + c_ids,
                "c_label": [-100] * len(p_ids) + c_ids,
                "r_input": p_ids + r_ids,
                "r_label": [-100] * len(p_ids) + r_ids,
                "weight": float(r.get("weight", 1.0)),
                "ref_c": 0.0,
                "ref_r": 0.0,
            })
        print(f"[CoverageDPODataset] loaded={len(self.items)} skipped={skipped} from {jsonl_path.name}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_dpo(batch, pad_id: int):
    c_batch = collate_completion_batch([(x["c_input"], x["c_label"]) for x in batch], pad_id)
    r_batch = collate_completion_batch([(x["r_input"], x["r_label"]) for x in batch], pad_id)
    return {
        "c": c_batch,
        "r": r_batch,
        "weight": torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
        "ref_c": torch.tensor([x["ref_c"] for x in batch], dtype=torch.float32),
        "ref_r": torch.tensor([x["ref_r"] for x in batch], dtype=torch.float32),
    }


def disable_lora_dropout(model):
    """Set LoRA dropout to 0 during DPO so pi_{theta_0} matches pi_ref deterministically."""
    for mod in model.modules():
        if hasattr(mod, "lora_dropout"):
            ld = getattr(mod, "lora_dropout")
            if isinstance(ld, torch.nn.ModuleDict):
                for k in ld.keys():
                    ld[k] = torch.nn.Identity()


@torch.no_grad()
def precompute_ref_logps(model, ds: CoverageDPODataset, pad_id: int, device: torch.device, batch_size: int = 2):
    model.eval()
    t0 = time.time()
    n = len(ds)
    for start in range(0, n, batch_size):
        chunk = [ds.items[i] for i in range(start, min(n, start + batch_size))]
        c_b = collate_completion_batch([(x["c_input"], x["c_label"]) for x in chunk], pad_id)
        r_b = collate_completion_batch([(x["r_input"], x["r_label"]) for x in chunk], pad_id)
        c_b = {k: v.to(device, non_blocking=True) for k, v in c_b.items()}
        r_b = {k: v.to(device, non_blocking=True) for k, v in r_b.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref_c, _ = compute_sequence_logps(model, c_b["input_ids"], c_b["attention_mask"], c_b["labels"])
            ref_r, _ = compute_sequence_logps(model, r_b["input_ids"], r_b["attention_mask"], r_b["labels"])
        for j, item in enumerate(chunk):
            item["ref_c"] = float(ref_c[j].cpu())
            item["ref_r"] = float(ref_r[j].cpu())
        if start % 32 == 0 and torch.cuda.memory_reserved() > 8 * (1024 ** 3):
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    print(f"[DPO] precomputed exact reference logps for {n} pairs in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--sft-adapter", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--rpo-alpha", type=float, default=0.25, help="NLL anchor weight on chosen completion")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=110)
    ap.add_argument("--lr", type=float, default=3.0e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2560)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU is required for DPO training"
    torch.cuda.set_per_process_memory_fraction(0.78, 0)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = CoverageDPODataset(args.data, tokenizer, max_length=args.max_length)
    if args.max_steps and args.max_steps > 0:
        needed = args.max_steps * args.batch_size * args.grad_accum
        if len(ds.items) > needed:
            rng_sub = random.Random(args.seed)
            rng_sub.shuffle(ds.items)
            ds.items = ds.items[:needed]
            print(f"[DPO] subsampled exact {len(ds.items)} pairs for {args.max_steps} optimizer steps", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    base.config.use_cache = False
    if args.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = PeftModel.from_pretrained(base, str(args.sft_adapter), is_trainable=True)
    disable_lora_dropout(model)
    precompute_ref_logps(model, ds, tokenizer.pad_token_id, device, batch_size=2)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dpo(b, tokenizer.pad_token_id),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(5, int(0.08 * total_steps))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    opt_step = 0
    micro = 0
    sum_loss, sum_acc, sum_margin, sum_nll = 0.0, 0.0, 0.0, 0.0
    log_history: List[dict] = []
    full_epochs = int(math.ceil(args.epochs))

    for epoch in range(full_epochs):
        for batch in loader:
            if opt_step >= total_steps:
                break
            c_b = {k: v.to(device, non_blocking=True) for k, v in batch["c"].items()}
            r_b = {k: v.to(device, non_blocking=True) for k, v in batch["r"].items()}
            w = batch["weight"].to(device, non_blocking=True)
            ref_c = batch["ref_c"].to(device, non_blocking=True)
            ref_r = batch["ref_r"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pol_c_sum, pol_c_mean = compute_sequence_logps(
                    model, c_b["input_ids"], c_b["attention_mask"], c_b["labels"]
                )
                pol_r_sum, _ = compute_sequence_logps(
                    model, r_b["input_ids"], r_b["attention_mask"], r_b["labels"]
                )
                c_rew = args.beta * (pol_c_sum - ref_c)
                r_rew = args.beta * (pol_r_sum - ref_r)
                logits_diff = c_rew - r_rew
                dpo_loss = -(w * F.logsigmoid(logits_diff)).mean()
                nll_loss = -pol_c_mean.mean()
                total_loss = (dpo_loss + args.rpo_alpha * nll_loss) / args.grad_accum

            total_loss.backward()

            sum_loss += float((dpo_loss + args.rpo_alpha * nll_loss).detach().cpu())
            sum_acc += float((logits_diff > 0).float().mean().detach().cpu())
            sum_margin += float(logits_diff.mean().detach().cpu())
            sum_nll += float(nll_loss.detach().cpu())
            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.memory_reserved() > 9 * (1024 ** 3):
                    torch.cuda.empty_cache()
                opt_step += 1

                if opt_step % 15 == 0 or opt_step == 1 or opt_step == total_steps:
                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    rec = {
                        "step": opt_step,
                        "total_steps": total_steps,
                        "loss": round(sum_loss / micro, 5),
                        "reward_acc": round(sum_acc / micro, 4),
                        "margin": round(sum_margin / micro, 4),
                        "nll": round(sum_nll / micro, 4),
                        "lr": round(scheduler.get_last_lr()[0], 7),
                        "vram_gb": round(vram_gb, 2),
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    log_history.append(rec)
                    print(
                        f"[Cov-DPO] step {opt_step:4d}/{total_steps} | loss={rec['loss']:.4f} | "
                        f"acc={rec['reward_acc']:.3f} | margin={rec['margin']:+.3f} | "
                        f"nll={rec['nll']:.4f} | vram={vram_gb:.2f}GB | t={rec['elapsed_s']:.0f}s",
                        flush=True,
                    )
                    sum_loss = sum_acc = sum_margin = sum_nll = 0.0
                    micro = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.base_model,
        "sft_adapter": str(args.sft_adapter),
        "data": str(args.data),
        "n_pairs": len(ds),
        "beta": args.beta,
        "rpo_alpha": args.rpo_alpha,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "total_steps": opt_step,
        "elapsed_s": round(time.time() - t0, 1),
        "log_history": log_history,
    }
    (args.output_dir / "train_log.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Cov-DPO] saved adapter to {args.output_dir} in {meta['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
