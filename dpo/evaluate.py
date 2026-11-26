"""Batched GPU evaluator under the STRICT frozen official HypoSpace 3D protocol:
  - Unchanged official prompt: Benchmark3D.create_prompt(obs, prior_structures=unique_structures)
  - Official layered voxel output parsed by Benchmark3D.parse_llm_response()
  - Budget N = |H_O| (m = 1 completion per slot; failed/dup/invalid attempts consume budget)
  - No validator in the generation loop (any parsed novel structure enters prior_structures)
  - Scored by official score_predictions.py
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import argparse
import json
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo.common import (
    QuestionSpec,
    Structure3D,
    count_blocks,
    format_chat_prompt,
    load_eval_questions,
    parse_and_normalize,
    tier_of,
)
from dpo.train_sft import set_seed


@torch.no_grad()
def run_split_evaluation(
    model,
    tokenizer,
    questions: List[QuestionSpec],
    out_dir: Path,
    split: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_new_tokens: int = 512,
    batch_size: int = 24,
    seed: int = 1,
) -> dict:
    set_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Per-question rollout state (strictly no ground-truth / no validator access!)
    states: Dict[str, dict] = {}
    for q in questions:
        states[q.qid] = {
            "q": q,
            "unique_structures": [],
            "seen_hashes": set(),
            "traces": [],
        }

    max_budget = max(q.budget for q in questions)
    t0 = time.time()

    for slot in range(1, max_budget + 1):
        active = [q for q in questions if slot <= q.budget]
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            raw_prompts = [
                q.bench.create_prompt(q.obs, prior_structures=states[q.qid]["unique_structures"])
                for q in chunk
            ]
            chat_prompts = [format_chat_prompt(tokenizer, p) for p in raw_prompts]
            enc = tokenizer(chat_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            in_len = enc["input_ids"].shape[1]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            for idx, q in enumerate(chunk):
                gen_ids = gen[idx, in_len:]
                # Strip trailing pad tokens
                eos_id = tokenizer.eos_token_id
                valid_toks = []
                hit_eos = False
                for tid in gen_ids.tolist():
                    if tid == eos_id:
                        hit_eos = True
                        break
                    if tid == tokenizer.pad_token_id and hit_eos:
                        break
                    valid_toks.append(tid)
                comp_tokens = len(valid_toks)
                truncated = (comp_tokens >= max_new_tokens) and (not hit_eos)
                text = tokenizer.decode(valid_toks, skip_special_tokens=True)
                prompt_tokens = int(enc["attention_mask"][idx].sum().item())

                st_obj = states[q.qid]
                parsed = parse_and_normalize(q.bench, text)
                is_dup = False
                h_val = None
                blocks = None
                tier = None
                if parsed is not None:
                    h_val = parsed.get_hash()
                    blocks = count_blocks(parsed)
                    tier = tier_of(blocks, q.k)
                    if h_val in st_obj["seen_hashes"]:
                        is_dup = True
                    else:
                        st_obj["seen_hashes"].add(h_val)
                        st_obj["unique_structures"].append(parsed)

                st_obj["traces"].append({
                    "question": q.qid,
                    "k": q.k,
                    "slot": slot,
                    "budget": q.budget,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "finish_reason": "length" if truncated else "stop",
                    "parsed": parsed is not None,
                    "hash": h_val,
                    "is_dup_any": is_dup,
                    "blocks": blocks,
                    "tier": tier,
                    "scorer_text": text,
                })

        if torch.cuda.memory_reserved() > 8 * (1024 ** 3):
            torch.cuda.empty_cache()
        if slot % 9 == 0 or slot == max_budget:
            print(f"[eval:{split}] finished slot {slot:2d}/{max_budget} ({time.time() - t0:.1f}s)", flush=True)

    # Write traces and predictions.jsonl
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as pf:
        for q in questions:
            t_rows = states[q.qid]["traces"]
            with (traces_dir / f"{q.qid}.jsonl").open("w", encoding="utf-8") as tf:
                for r in t_rows:
                    tf.write(json.dumps(r, ensure_ascii=False) + "\n")
                    pf.write(
                        json.dumps(
                            {"question": r["question"], "slot": r["slot"], "scorer_text": r["scorer_text"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    # Run official unmodified score_predictions.py if evaluating a full split
    all_split_qs = load_eval_questions(split)
    if len(questions) < len(all_split_qs):
        total_s = round(time.time() - t0, 2)
        print(f"[eval:{split}] saved subset traces ({len(questions)} questions) in {total_s}s", flush=True)
        return {"questions": len(questions)}

    score_path = out_dir / "score.json"
    if score_path.exists():
        score_path.unlink()
    cmd = [sys.executable, str(ROOT / "score_predictions.py"), split, str(pred_path), "--output", str(score_path)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    score_data = json.loads(score_path.read_text(encoding="utf-8"))
    summary = score_data["summary"]
    total_s = round(time.time() - t0, 2)
    print(
        f"[eval:{split}] OFFICIAL SCORE -> "
        f"parse={summary['parse_success_rate']*100:.2f}% | "
        f"VR={summary['valid_rate']*100:.2f}% | "
        f"NR={summary['novelty_rate']*100:.2f}% | "
        f"RR={summary['recovery_rate']*100:.2f}% | ({total_s}s)",
        flush=True,
    )
    return score_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir; omit for base model")
    ap.add_argument("--arm", required=True, help="Name of arm, e.g. base, sft_validity, sft_history, coverage_dpo")
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--repeats", type=int, nargs="+", default=[1])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--limit-per-group", type=int, default=0, help="Optional per-k question cap for fast train rollouts")
    ap.add_argument("--out-root", type=Path, default=ROOT / "dpo" / "runs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU required"
    torch.cuda.set_per_process_memory_fraction(0.78, 0)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    questions = load_eval_questions(args.split)
    if args.limit_per_group > 0:
        k2 = [q for q in questions if q.k == 2][: args.limit_per_group]
        k3 = [q for q in questions if q.k == 3][: args.limit_per_group * 2]
        questions = k2 + k3
    for rep in args.repeats:
        rep_dir = args.out_root / args.split / args.arm / f"rep{rep}"
        if (rep_dir / "score.json").exists():
            print(f"[eval] skipping existing {rep_dir / 'score.json'}", flush=True)
            continue
        run_split_evaluation(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            out_dir=rep_dir,
            split=args.split,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=1000 * rep + 17,
        )


if __name__ == "__main__":
    main()
