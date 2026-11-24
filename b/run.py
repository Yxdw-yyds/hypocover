"""Run role-B guided height-vector generation on the frozen 3D benchmark.

Example:
  python b/run.py --split validation --policy coord_fixed
  python b/run.py --split validation --policy free --limit 2 --dry-run

Writes predictions-<split>-<policy>.jsonl for score_predictions.py and
b/traces/<policy>/<question>.jsonl with every prompt/answer/usage record.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from .policy import build_prompt
from .representation import Question, load_questions, parse_vec, vec_to_scorer_text

ROOT = Path(__file__).resolve().parents[1]
MODEL = "deepseek/deepseek-v4-flash-0731"


def read_key(env_file: str | None) -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    candidates = [Path(env_file)] if env_file else []
    candidates += [ROOT.parent / "hypocover" / ".env", ROOT / ".env"]
    for path in candidates:
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"\s*OPENROUTER_API_KEY\s*=\s*(\S+)", line)
                if m:
                    return m.group(1)
    raise SystemExit("OPENROUTER_API_KEY not found (env var or .env)")


def ask(client: OpenAI, prompt: str, args) -> dict:
    t0 = time.time()
    r = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        extra_body={"reasoning": {"enabled": False}},
    )
    choice = r.choices[0]
    u = r.usage
    cost = getattr(u, "cost", None) or (u.model_extra or {}).get("cost", 0.0) or 0.0
    return {
        "text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "cost": float(cost),
        "provider": getattr(r, "provider", None) or (r.model_extra or {}).get("provider"),
        "served_model": getattr(r, "model", None),
        "seconds": round(time.time() - t0, 2),
    }


def run_question(q: Question, client: OpenAI, args) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed + zlib.crc32(q.id.encode()))
    found: list[tuple[int, ...]] = []
    preds, traces = [], []
    for slot in range(1, q.budget + 1):
        prompt, action = build_prompt(q, found, args.policy, slot - 1, rng)
        if args.dry_run:
            print(f"\n=== {q.id} slot {slot} [{action}] ===\n{prompt}")
            preds.append({"question": q.id, "slot": slot, "scorer_text": ""})
            continue
        rec, err = None, None
        for attempt in range(args.retries):
            try:
                rec = ask(client, prompt, args)
                break
            except Exception as e:  # noqa: BLE001 - network/API errors
                err = str(e)
                time.sleep(2 * (attempt + 1))
        if rec is None:
            preds.append({"question": q.id, "slot": slot, "scorer_text": ""})
            traces.append({"question": q.id, "slot": slot, "action": action,
                           "prompt": prompt, "error": err, "vec": None, "new": False})
            continue
        vec = parse_vec(rec["text"], q.k)
        text = vec_to_scorer_text(vec, q.positions) if vec is not None else ""
        new = vec is not None and vec not in found
        if vec is not None and vec not in found:
            found.append(vec)
        preds.append({"question": q.id, "slot": slot, "scorer_text": text})
        traces.append({"question": q.id, "slot": slot, "action": action, "prompt": prompt,
                       "response": rec["text"], "vec": vec, "new": new,
                       "finish_reason": rec["finish_reason"], "usage": {
                           "prompt_tokens": rec["prompt_tokens"],
                           "completion_tokens": rec["completion_tokens"],
                           "cost": rec["cost"]},
                       "provider": rec["provider"], "served_model": rec["served_model"],
                       "seconds": rec["seconds"]})
        if args.verbose:
            print(f"{q.id} slot {slot:2d} [{action:6s}] vec={vec} new={new} "
                  f"{rec['seconds']}s ${rec['cost']:.6f}", flush=True)
    return preds, traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--policy", default="coord_fixed", choices=["free", "coord_fixed", "coord_random"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--retries", type=int, default=1, help="API attempts per slot")
    ap.add_argument("--seed", type=int, default=33550336)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    questions = load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]
    out = Path(args.out or f"predictions-{args.split}-{args.policy}.jsonl")
    traces_dir = Path(args.traces or f"b/traces/{args.policy}")
    traces_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for q in questions:
            run_question(q, None, args)
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_key(args.env_file),
                    timeout=120)
    all_preds, calls, cost = [], 0, 0.0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_question, q, client, args): q for q in questions}
        for fu in as_completed(futs):
            q = futs[fu]
            preds, traces = fu.result()
            all_preds += preds
            calls += sum(1 for t in traces if "usage" in t)
            cost += sum(t["usage"]["cost"] for t in traces if "usage" in t)
            with (traces_dir / f"{q.id}.jsonl").open("w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"done {q.id}: {len(preds)} slots", flush=True)

    all_preds.sort(key=lambda r: (r["question"], r["slot"]))
    with out.open("w", encoding="utf-8") as f:
        for r in all_preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parsed = sum(1 for r in all_preds if r["scorer_text"])
    print(f"\n{args.split}/{args.policy}: {len(questions)} questions, {len(all_preds)} slots, "
          f"parsed={parsed} ({parsed / max(len(all_preds), 1):.1%}), calls={calls}, cost=${cost:.4f}")
    print(f"predictions -> {out}")


if __name__ == "__main__":
    main()
