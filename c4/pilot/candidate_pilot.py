"""PILOT: value-guided candidate selection on the official layered 3D prompt (train split only).

Per step: sample m candidates from the model with the unchanged official prompt, classify each with the
official parser/validator (+ hash for novelty), submit ONE according to --select:
  first         candidates[0]                      (= free_official, m extra calls wasted; control)
  valid_new     first candidate that is valid & new, else candidates[0]   (filter only, NOT learned)
  tier_balance  among valid & new candidates, the one whose complexity tier is most under-covered
                relative to the tier capacities implied by k (heuristic stand-in for a learned value)
Writes c4/pilot/cand/<select>/<qid>.jsonl.  Not part of the frozen protocol.
"""
import argparse, json, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from c4 import run_layered as RL  # noqa: E402
from c4 import layered_actions as LA  # noqa: E402

LOCK = threading.Lock()
SPENT = {"v": 0.0}


def classify(bench, obs, k, text, seen):
    out = {"outcome": "unparsed", "tier": None, "blocks": None, "hash": None, "st": None}
    try:
        st = bench.parse_llm_response(text)
        if st is not None:
            st = st.normalize()
            b = LA.count_blocks(st); h = st.get_hash(); valid = bench.validate_structure_matches_observations(st, obs)
            out.update(blocks=b, tier=LA.tier_of(b, k), hash=h, st=st,
                       outcome="dup" if h in seen else ("new" if valid else "invalid"))
    except Exception:  # noqa: BLE001
        pass
    return out


def run_q(q, backend, a):
    bench, obs, k, N = q["bench"], q["obs"], q["k"], q["budget"]
    seen, unique, found = set(), [], {"low": 0, "mid": 0, "high": 0}
    cap = LA.gt_tier_counts(k)
    rows = []
    for slot in range(1, N + 1):
        with LOCK:
            if SPENT["v"] > a.max_cost:
                return q["id"], None
        prompt = bench.create_prompt(obs, prior_structures=unique)

        def one(_):
            for attempt in range(3):
                try:
                    return backend.query(prompt, {})
                except Exception as e:  # noqa: BLE001
                    if any(f" {c} " in str(e) for c in (401, 402, 403)):
                        raise
                    time.sleep(3 * (attempt + 1))
            return None
        with ThreadPoolExecutor(a.m) as ex:
            recs = list(ex.map(one, range(a.m)))
        if any(r is None for r in recs):
            return q["id"], None
        with LOCK:
            SPENT["v"] += sum(r["cost"] or 0 for r in recs)
        cands = [classify(bench, obs, k, r["text"], seen) for r in recs]
        # candidates that repeat an earlier candidate of this same step are not new twice
        s2 = set()
        for c in cands:
            if c["outcome"] == "new":
                if c["hash"] in s2:
                    c["outcome"] = "dup"
                s2.add(c["hash"])
        pick = 0
        if a.select in ("valid_new", "tier_balance"):
            news = [i for i, c in enumerate(cands) if c["outcome"] == "new"]
            if news:
                pick = news[0]
                if a.select == "tier_balance":
                    best = max(news, key=lambda i: (cap[cands[i]["tier"]] - found[cands[i]["tier"]], -i))
                    pick = best
        ch = cands[pick]
        if ch["outcome"] in ("new", "invalid"):
            seen.add(ch["hash"]); unique.append(ch["st"])
        if ch["outcome"] == "new":
            found[ch["tier"]] += 1
        rows.append({"slot": slot, "pick": pick, "chosen": ch["outcome"], "chosen_tier": ch["tier"],
                     "cands": [{"o": c["outcome"], "t": c["tier"]} for c in cands],
                     "trunc": [r["finish_reason"] == "length" for r in recs],
                     "cost": sum(r["cost"] or 0 for r in recs),
                     "response": recs[pick]["text"]})
    return q["id"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True, choices=["first", "valid_new", "tier_balance"])
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--max-cost", type=float, default=0.5)
    ap.add_argument("--ledger", default=str(ROOT / "c4" / "spend_ledger_pilot.jsonl"))
    a = ap.parse_args()
    RL.LEDGER = Path(a.ledger)
    backend = RL.DeepSeek(1024, "deepseek-flash", "disabled", 0.30, 1.20, "https://api.deepseek.com", 0.006)
    qs = [q for q in RL.load_questions("train", a.ids)]
    out = ROOT / "c4" / "pilot" / "cand" / a.select; out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(qs)) as ex:
        for qid, rows in ex.map(lambda q: run_q(q, backend, a), qs):
            if rows is None:
                print("abandoned", qid, flush=True); continue
            (out / f"{qid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            c = sum(r["cost"] for r in rows); RL.ledger_add(f"pilotcand/{a.select}/{qid}", c)
            print(f"done {qid}: new={sum(r['chosen']=='new' for r in rows)}/{len(rows)} cost=${c:.4f}", flush=True)
    print(f"TOTAL ${SPENT['v']:.4f}")


if __name__ == "__main__":
    main()
