"""Does free cover the complex hypotheses early? (budget-limited bias check)"""
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

for k in (2, 3):
    recs = [r for _, r in rows if len(r[0]["vec"]) == k]
    allv = list(itertools.product((1, 2, 3), repeat=k))
    comp = {v: sum(v) for v in allv}
    order = sorted(allv, key=lambda v: comp[v])
    low = set(order[:len(order) // 3])
    high = set(order[-len(order) // 3:])
    print(f"k={k}: |H_O|={len(allv)} mean blocks={mean(comp.values()):.2f} "
          f"low-third mean={mean(comp[v] for v in low):.2f} high-third mean={mean(comp[v] for v in high):.2f}")
    for budget in (max(1, len(allv) // 3), max(2, len(allv) // 2), len(allv) - 1):
        stats = []
        for rec in recs:
            found = []
            for r in rec[:budget]:
                v = parse_vec(r.get("response"), k)
                if v is not None and v not in found:
                    found.append(v)
            stats.append((mean(comp[v] for v in found), len(set(found) & low) / len(low),
                          len(set(found) & high) / len(high)))
        print(f"  budget={budget:2d}: mean blocks of found={stats[0][0]:.2f} "
              f"low-third covered={mean(s[1] for s in stats):.2f} "
              f"high-third covered={mean(s[2] for s in stats):.2f}")
