"""Strategy-layer runner on the OFFICIAL layered 3D prompt / parser / validator.

  python -m c4.run_layered --split train --policy random_action --backend mock --limit 3
  python -m c4.run_layered --split validation --policy learned --model-file c4/policy.json --repeat 1

Per (split, policy, repeat) it writes  c4/layered/<split>/<policy>/rep<R>/
  traces/<question>.jsonl   one record per slot (action, outcome, blocks, tier, compliance, usage)
  predictions.jsonl         {question, slot, scorer_text}  -> official score_predictions.py
The key is read ONLY from the OPENROUTER_API_KEY environment variable or, if unset, from the
private key file written by c4/save_key.py (~/.hypocover_openrouter_key); it is never printed.
Every finished question appends its cost to c4/spend_ledger.jsonl; a question is not started
once the ledger total exceeds --total-cap. A trace file that
already exists is skipped (resume); each repeat has its own directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "3d"))
sys.path.insert(0, str(ROOT))
from run_3d_benchmark import Benchmark3D  # noqa: E402

from c4 import layered_actions as LA  # noqa: E402

MODEL = "deepseek/deepseek-v4-flash-0731"
URL = "https://openrouter.ai/api/v1/chat/completions"
POLICIES = ("free_official", "random_action", "random_level", "tier_cycle", "tier_static", "paper_stratified", "learned")
_COST = {"spent": 0.0}
_LOCK = threading.Lock()
_FATAL = threading.Event()  # set on 401/402/403: stop all slots without writing partial results
DS_KEY_FILE = Path(os.environ.get("HYPOCOVER_DS_KEY_FILE", str(Path.home() / ".hypocover_deepseek_key")))
KEY_FILE = Path(os.environ.get("HYPOCOVER_KEY_FILE", str(Path.home() / ".hypocover_openrouter_key")))
LEDGER = ROOT / "c4" / "spend_ledger.jsonl"


def read_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
    return k


def ledger_total() -> float:
    if not LEDGER.exists():
        return 0.0
    tot = 0.0
    for l in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            tot += float(json.loads(l)["cost"])
        except Exception:  # noqa: BLE001
            pass
    return tot


def ledger_add(label: str, cost: float):
    with _LOCK, LEDGER.open("a", encoding="utf-8") as f:  # append-only: safe across processes
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "cost": round(cost, 6)}) + chr(10))


# ------------------------------------------------------------------ data
def load_questions(split: str, ids=None):
    groups = json.loads((ROOT / "data" / "manifest.json").read_text())["groups"]
    out = []
    for gid, g in groups.items():
        bench = Benchmark3D(str(ROOT / "data" / g["file"]))
        items = {i["observation_id"]: i for i in bench.all_observation_sets}
        for q in g[split]:
            if ids and q not in ids:
                continue
            obs = items[q]["observation"]
            out.append({"id": q, "obs": obs, "k": LA.k_of(obs), "budget": int(g["budget"]), "bench": bench})
    return out


# ------------------------------------------------------------------ backends
class OpenRouter:
    def __init__(self, max_tokens: int, provider: str | None = None):
        import requests
        self.requests = requests
        self.provider = provider  # None = OpenRouter auto-routing (same regime as A's and B's runs)
        self.key = read_key()
        if not self.key:
            sys.exit("no API key: set OPENROUTER_API_KEY or run  python -B c4/save_key.py  once")
        self.max_tokens = max_tokens

    def query(self, prompt, ctx):
        t0 = time.time()
        body = {
            "model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7,
            "top_p": 0.8, "max_tokens": self.max_tokens, "reasoning": {"enabled": False},
            "usage": {"include": True}}
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        r = self.requests.post(URL, timeout=180, headers={"Authorization": f"Bearer {self.key}"}, json=body)
        r.raise_for_status()
        j = r.json(); ch = j["choices"][0]; u = j.get("usage", {})
        return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "cost": float(u.get("cost") or 0.0), "provider": j.get("provider"),
                "served_model": j.get("model"), "seconds": round(time.time() - t0, 2)}


class DeepSeek:
    """DeepSeek official API (one provider, no routing). Cost is ESTIMATED from token counts and the
    --price-in/--price-out flags (USD per 1M tokens); the real limit is the account balance."""
    def __init__(self, max_tokens, model, thinking, price_in, price_out, base_url, price_hit=0.0):
        import requests
        self.requests = requests
        self.key = os.environ.get("DEEPSEEK_API_KEY") or (
            DS_KEY_FILE.read_text(encoding="utf-8").strip() if DS_KEY_FILE.exists() else None)
        if not self.key:
            sys.exit("no DeepSeek key: run  python -B c4/save_key.py --deepseek  once")
        if not model:
            sys.exit("--model is required for the deepseek backend")
        self.max_tokens, self.model, self.thinking = max_tokens, model, thinking
        self.price_hit = price_hit
        self.price_in, self.price_out, self.url = price_in, price_out, base_url.rstrip("/") + "/chat/completions"

    def _cost(self, u, pt, ct):
        hit = u.get("prompt_cache_hit_tokens"); miss = u.get("prompt_cache_miss_tokens")
        if hit is None or miss is None:  # no cache fields: treat every prompt token as a miss (upper bound)
            hit, miss = 0, pt
        return (miss * self.price_in + hit * self.price_hit + ct * self.price_out) / 1e6

    def query(self, prompt, ctx):
        t0 = time.time()
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7,
                "top_p": 0.8, "max_tokens": self.max_tokens}
        if self.thinking in ("disabled", "enabled"):
            body["thinking"] = {"type": self.thinking}
        r = self.requests.post(self.url, timeout=180, headers={"Authorization": f"Bearer {self.key}"}, json=body)
        r.raise_for_status()
        j = r.json(); ch = j["choices"][0]; u = j.get("usage", {})
        pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        return {"text": ch["message"].get("content") or "", "finish_reason": ch.get("finish_reason"),
                "prompt_tokens": pt, "completion_tokens": ct,
                "cost": self._cost(u, pt, ct), "provider": "deepseek-official",
                "served_model": j.get("model"), "seconds": round(time.time() - t0, 2),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}


class Mock:
    """Offline stand-in: obeys hints ~85% of the time, repeats ~12%, breaks support ~10%."""
    def __init__(self, seed: int, cost: float = 0.0):
        self.seed = seed
        self.cost = cost

    def query(self, prompt, ctx):
        rng = ctx["rng"]; obs = ctx["obs"]; k = ctx["k"]; action = ctx["action"]
        cells = [(i // 3, i % 3) for i, ch in enumerate(obs) if ch == "1"]
        past = ctx["mock_past"]
        if past and rng.random() < 0.12:
            heights = past[int(rng.integers(len(past)))]
        else:
            for _ in range(200):
                heights = [int(rng.integers(1, 4)) for _ in cells]
                if action == "none" or rng.random() < 0.15 or LA.tier_of(sum(heights), k) == action:
                    break
        past.append(heights)
        layers = []
        for l in range(1, max(heights) + 1):
            grid = [[0] * 3 for _ in range(3)]
            for (r, c), h in zip(cells, heights):
                grid[r][c] = 1 if h >= l else 0
            layers.append(grid)
        if rng.random() < 0.10:  # invalid: a block on an empty top-view cell
            empty = [(r, c) for r in range(3) for c in range(3) if (r, c) not in cells]
            if empty:
                r, c = empty[int(rng.integers(len(empty)))]; layers[0][r][c] = 1
        text = "Structure:\n" + "\n".join(
            f"Layer {i + 1}:\n" + "\n".join(" ".join(map(str, row)) for row in g) for i, g in enumerate(layers))
        return {"text": text, "finish_reason": "stop", "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(text) // 4, "cost": self.cost, "provider": "mock",
                "served_model": "mock", "seconds": 0.0}


# ------------------------------------------------------------------ one question
def choose(policy, step, q, history, rng, model):
    if policy == "free_official":
        return "none"
    if policy.startswith("fixed_"):  # pilot: always the same action
        return policy[6:]
    if policy == "random_action":
        return LA.choose_random_action(rng)
    if policy == "random_level":
        return LA.choose_random_level(rng)
    if policy == "tier_cycle":
        return LA.choose_tier_cycle(step)
    if policy == "tier_static":
        return LA.choose_tier_static(step, q["k"])
    if policy == "paper_stratified":
        return LA.choose_paper_stratified(step, q["budget"], q["k"])
    if policy == "learned":
        return LA.choose_learned(model, history, step, q["budget"], q["k"])
    raise ValueError(policy)


def run_question(q, backend, a, model):
    bench, obs, k, N = q["bench"], q["obs"], q["k"], q["budget"]
    rng = np.random.default_rng(a.seed + 1000 * a.repeat + zlib.crc32(q["id"].encode()))
    unique, seen = [], set()
    history, traces, preds, mock_past = [], [], [], []
    if not a.dry_run and a.backend != "mock" and ledger_total() > a.total_cap:
        return q["id"], None, None  # lifetime cap reached: nothing is written, nothing is spent
    for slot in range(1, N + 1):
        with _LOCK:
            over = _COST["spent"] > a.max_cost
        if over or _FATAL.is_set():  # cap reached or payment/auth failure: abandon this question WITHOUT writing a trace, so resume re-runs it
            return q["id"], None, None
        action = choose(a.policy, slot - 1, q, history, rng, model)
        prompt = LA.add_hint(bench.create_prompt(obs, prior_structures=unique), action, k)
        if a.dry_run:
            if slot <= 2:
                print(f"=== {q['id']} slot {slot} [{action}] ===\n{prompt[-1200:]}\n")
            history.append({"action": action, "outcome": "new", "tier": None})
            continue
        rec, err = None, None
        for attempt in range(3):
            try:
                rec = backend.query(prompt, {"rng": rng, "obs": obs, "k": k, "action": action, "mock_past": mock_past})
                break
            except Exception as e:  # transport/API failure: counted separately from model errors
                err = type(e).__name__ + ": " + str(e)[:200]
                if any(f" {c} " in err for c in (401, 402, 403)):  # key/credit problem: stop everything, write nothing
                    _FATAL.set()
                    break
                time.sleep(3 * (attempt + 1))
        if rec is None and _FATAL.is_set():
            return q["id"], None, None
        if rec is None:
            preds.append({"question": q["id"], "slot": slot, "scorer_text": ""})
            traces.append({"question": q["id"], "slot": slot, "action": action, "transport_error": err})
            continue
        outcome, blocks, tier, comp = "unparsed", None, None, None
        try:  # an odd response must never abort the run; the official scorer still judges the raw text
            st = bench.parse_llm_response(rec["text"])
            if st is not None:
                st = st.normalize()
                blocks = LA.count_blocks(st); tier = LA.tier_of(blocks, k); comp = LA.complies(action, k, blocks)
                h = st.get_hash()
                valid = bench.validate_structure_matches_observations(st, obs)
                if h in seen:
                    outcome = "dup"
                else:
                    outcome = "new" if valid else "invalid"
                    seen.add(h); unique.append(st)
        except Exception:  # noqa: BLE001
            outcome, blocks, tier, comp = "unparsed", None, None, None
        with _LOCK:
            _COST["spent"] += rec["cost"] or 0.0
        history.append({"action": action, "outcome": outcome, "tier": tier})
        preds.append({"question": q["id"], "slot": slot, "scorer_text": rec["text"]})
        traces.append({"question": q["id"], "slot": slot, "k": k, "budget": N, "action": action,
                       "outcome": outcome, "blocks": blocks,
                       "tier": tier, "complies": comp, "finish_reason": rec["finish_reason"],
                       "prompt_tokens": rec["prompt_tokens"], "completion_tokens": rec["completion_tokens"],
                       "cost": rec["cost"], "provider": rec["provider"], "served_model": rec["served_model"],
                       "seconds": rec["seconds"], "response": rec["text"]})
    return q["id"], traces, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--policy", default="free_official")  # POLICIES or fixed_<action> (pilot)
    ap.add_argument("--model-file", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--limit", type=int)
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "deepseek", "mock"])
    ap.add_argument("--model", default=None, help="deepseek backend: model name")
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled", "omit"])
    ap.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens (estimate for the ledger)")
    ap.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    ap.add_argument("--price-hit", type=float, default=0.0, help="USD per 1M cache-hit prompt tokens")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--ledger", default=None, help="ledger file (default c4/spend_ledger.jsonl)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider (no fallback); default: auto-routing")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=33550336)
    ap.add_argument("--max-cost", type=float, default=1.0, help="per-process spend cap (USD)")
    ap.add_argument("--total-cap", type=float, default=3.5, help="lifetime cap over c4/spend_ledger.jsonl (USD)")
    ap.add_argument("--out-root", default=str(ROOT / "c4" / "layered"))
    ap.add_argument("--mock-cost", type=float, default=0.0, help="per-call cost reported by the mock backend (tests the cap)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global LEDGER
    if a.ledger:
        LEDGER = Path(a.ledger)

    model = None
    if a.policy == "learned":
        if not a.model_file:
            sys.exit("--policy learned needs --model-file")
        model = json.loads(Path(a.model_file).read_text(encoding="utf-8"))
    qs = load_questions(a.split, a.ids)[: a.limit]
    backend = None if a.dry_run else (Mock(a.seed, a.mock_cost) if a.backend == "mock" else DeepSeek(a.max_tokens, a.model, a.thinking, a.price_in, a.price_out, a.base_url, a.price_hit) if a.backend == "deepseek" else OpenRouter(a.max_tokens, a.provider))
    out = Path(a.out_root) / a.split / a.policy / f"rep{a.repeat}"
    if not a.dry_run:
        (out / "traces").mkdir(parents=True, exist_ok=True)
    todo = [q for q in qs if a.dry_run or not (out / "traces" / f"{q['id']}.jsonl").exists()]
    all_preds = []
    # prior predictions of already-finished questions are re-read so predictions.jsonl stays complete
    for q in qs:
        f = out / "traces" / f"{q['id']}.jsonl"
        if f.exists() and q not in todo:
            for l in f.read_text(encoding="utf-8").splitlines():
                t = json.loads(l)
                all_preds.append({"question": q["id"], "slot": t["slot"], "scorer_text": t.get("response", "")})
    with ThreadPoolExecutor(1 if a.dry_run else a.workers) as pool:
        futs = {pool.submit(run_question, q, backend, a, model): q for q in todo}
        skipped = []
        for fu in as_completed(futs):
            qid, traces, preds = fu.result()
            if a.dry_run:
                continue
            if traces is None:
                skipped.append(qid); continue
            (out / "traces" / f"{qid}.jsonl").write_text(
                "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces), encoding="utf-8")
            all_preds += preds
            c = sum(t.get("cost", 0) or 0 for t in traces); spent = _COST["spent"]
            if a.backend != "mock":
                ledger_add(f"{a.split}/{a.policy}/rep{a.repeat}/{qid}", c)
            n_new = sum(t.get("outcome") == "new" for t in traces)
            n_err = sum("transport_error" in t for t in traces)
            print(f"done {qid}: new={n_new}/{len(traces)} transport_err={n_err} cost=${c:.4f} total=${spent:.4f}", flush=True)
    if not a.dry_run and skipped:
        sys.exit(f"max-cost ${a.max_cost} reached; {len(skipped)} question(s) NOT run (rerun to resume): {skipped[:5]}...")
    if not a.dry_run:
        spent = _COST["spent"]
        all_preds.sort(key=lambda r: (r["question"], r["slot"]))
        (out / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_preds), encoding="utf-8")
        print(f"TOTAL ${spent:.4f}  ->  {out}")


if __name__ == "__main__":
    main()
