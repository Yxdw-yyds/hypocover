"""k=4 runner (role C pilot).  Reuses b.policy / b.representation unchanged; B's decoding.
Key is read ONLY from the OPENROUTER_API_KEY environment variable.
  python -m c4.run4 --split train --policy free --ids k4_tr_00 --dry-run
  python -m c4.run4 --split train --policy free --limit 6
"""
import argparse, json, os, random, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from b.policy import build_prompt
from b.representation import Question, parse_vec

HERE = Path(__file__).resolve().parent
MODEL = "deepseek/deepseek-v4-flash-0731"
URL = "https://openrouter.ai/api/v1/chat/completions"

def load(split, ids=None):
    d = json.loads((HERE / "data" / "k4-questions.json").read_text())
    out = []
    for q in d["questions"]:
        if q["split"] != split or (ids and q["id"] not in ids): continue
        out.append(Question(id=q["id"], top_view=tuple(tuple(r) for r in q["top_view"]),
                            positions=tuple(tuple(p) for p in q["positions"]),
                            budget=q["n_solutions"], split=split, group="4"))
    return out

def ask(prompt, key, a):
    t0 = time.time()
    r = requests.post(URL, timeout=120, headers={"Authorization": f"Bearer {key}"}, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, "top_p": 0.8, "max_tokens": 512,
        "reasoning": {"enabled": False}, "usage": {"include": True}})
    r.raise_for_status()
    j = r.json(); ch = j["choices"][0]; u = j.get("usage", {})
    return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cost": float(u.get("cost") or 0.0), "provider": j.get("provider"),
            "served_model": j.get("model"), "seconds": round(time.time() - t0, 2)}

_COST = {"spent": 0.0}
_LOCK = threading.Lock()


def run_question(q, key, a):
    rng = random.Random(a.seed + zlib.crc32(q.id.encode()))
    found, traces = [], []
    for slot in range(1, a.budget + 1):
        with _LOCK:
            if _COST["spent"] > a.max_cost:
                return q, None  # cap reached: no trace is written, rerun resumes this question
        prompt, action = build_prompt(q, found, a.policy, slot - 1, rng)
        if a.dry_run:
            print(f"=== {q.id} slot {slot} [{action}] ===\n{prompt}\n"); 
            if slot >= 2: break
            found.append(tuple([1] * q.k)); continue
        rec, err = None, None
        for attempt in range(3):
            try: rec = ask(prompt, key, a); break
            except Exception as e: err = type(e).__name__ + ": " + str(e)[:200]; time.sleep(3 * (attempt + 1))
        if rec is None:
            traces.append({"question": q.id, "slot": slot, "action": action, "transport_error": err,
                           "vec": None, "new": False}); continue
        vec = parse_vec(rec["text"], q.k)
        new = vec is not None and vec not in found
        if new: found.append(vec)
        with _LOCK:
            _COST["spent"] += rec["cost"] or 0.0
        traces.append({"question": q.id, "slot": slot, "action": action, "response": rec["text"],
                       "vec": vec, "new": new, "finish_reason": rec["finish_reason"],
                       "usage": {k: rec[k] for k in ("prompt_tokens", "completion_tokens", "cost")},
                       "provider": rec["provider"], "served_model": rec["served_model"], "seconds": rec["seconds"]})
    return q, traces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train"); ap.add_argument("--policy", default="free",
                    choices=["free", "coord_fixed", "coord_random"])
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--limit", type=int)
    ap.add_argument("--budget", type=int, default=81); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=33550336); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=0.30, help="stop launching when exceeded")
    a = ap.parse_args()
    qs = load(a.split, a.ids)[: a.limit]
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and not a.dry_run: sys.exit("OPENROUTER_API_KEY not set in this shell")
    outdir = HERE / "traces" / a.policy; outdir.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = []
        for q in qs:
            if a.dry_run or not (outdir / f"{q.id}.jsonl").exists():
                futs.append(pool.submit(run_question, q, key, a))
        for fu in as_completed(futs):
            q, tr = fu.result()
            if a.dry_run: continue
            if tr is None:
                print("max-cost reached; skipped", q.id); continue
            (outdir / f"{q.id}.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tr), encoding="utf-8")
            c = sum(t.get("usage", {}).get("cost", 0) for t in tr); spent += c
            print(f"done {q.id} found={sum(t['new'] for t in tr)}/{q.budget} "
                  f"transport_err={sum('transport_error' in t for t in tr)} cost=${c:.4f} total=${spent:.4f}", flush=True)
    print("TOTAL", f"${spent:.4f}")
if __name__ == "__main__": main()
