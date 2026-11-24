"""Analyze where the free policy repeats or misses (re-parses with the fixed parser)."""
import itertools
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import parse_vec

root = Path(__file__).resolve().parents[1]
rows = []
for p in sorted((root / "b/traces/free").glob("*.jsonl")):
    rec = [json.loads(line) for line in p.open(encoding="utf-8")]
    rows.append((p.stem, rec))

print("questions:", len(rows))
dup_total = 0
for qid, rec in rows:
    k = len(rec[0]["vec"])
    found, dups = [], []
    for r in rec:
        v = parse_vec(r.get("response"), k)
        if v is None:
            dups.append((r["slot"], None))
        elif v in found:
            dups.append((r["slot"], v))
        else:
            found.append(v)
    missing = sorted(set(itertools.product((1, 2, 3), repeat=k)) - set(found))
    if dups or missing:
        print(f"{qid}: dups={[d[0] for d in dups]} missing={missing}")
    dup_total += len(dups)
print("total duplicates:", dup_total, "of", sum(len(r) for _, r in rows))

# RR@k curves
print("\nRR@k (mean over questions):")
for k in (2, 3):
    recs = [r for _, r in rows if len(r[0]["vec"]) == k]
    curve = []
    for slot in range(1, len(recs[0]) + 1):
        vals = []
        for r in recs:
            seen = set()
            for x in r[:slot]:
                v = parse_vec(x.get("response"), k)
                if v is not None:
                    seen.add(v)
            vals.append(len(seen) / (3 ** k))
        curve.append(round(mean(vals), 3))
    print(f"  k={k}: " + " ".join(f"{i+1}:{v}" for i, v in enumerate(curve)))
