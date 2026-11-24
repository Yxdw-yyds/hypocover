"""Reconcile the 'mean blocks' number under several plausible conventions."""
import itertools
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import parse_vec

root = Path(__file__).resolve().parents[1]
recs = []
for p in sorted((root / "b/traces/free").glob("*.jsonl")):
    rec = [json.loads(line) for line in p.open(encoding="utf-8")]
    if len(rec[0]["vec"]) == 3:
        recs.append(rec)

print("k=3 questions:", len(recs), "budget:", len(recs[0]))


def vecs(rec, slots):
    out = []
    for i in slots:
        v = parse_vec(rec[i].get("response"), 3)
        if v is not None:
            out.append(v)
    return out


def stats(name, fn):
    vals = [fn(rec) for rec in recs]
    print(f"{name:38s} mean={mean(vals):.3f}")


stats("first 9, unique", lambda r: mean(sum(v) for v in dict.fromkeys(vecs(r, range(9)))))
stats("first 9, all rows (dups kept)", lambda r: mean(sum(v) for v in vecs(r, range(9))))
stats("first 9, trace vec field", lambda r: mean(sum(r[i]["vec"]) for i in range(9)))
stats("last 9 (slots 19-27), unique", lambda r: mean(sum(v) for v in dict.fromkeys(vecs(r, range(18, 27)))))
stats("slots 10-18, unique", lambda r: mean(sum(v) for v in dict.fromkeys(vecs(r, range(9, 18)))))
stats("slots 5-13, unique", lambda r: mean(sum(v) for v in dict.fromkeys(vecs(r, range(4, 13)))))
stats("all 27, unique", lambda r: mean(sum(v) for v in dict.fromkeys(vecs(r, range(27)))))
stats("all 27, all rows", lambda r: mean(sum(v) for v in vecs(r, range(27))))
stats("mean of per-slot cumulative mean", lambda r: mean(
    mean(sum(v) for v in dict.fromkeys(vecs(r, range(i + 1)))) for i in range(9)))
