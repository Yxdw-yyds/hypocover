"""Gate-1 report for k=4.  PASS criterion (fixed BEFORE running): free wastes >=2 slots
(duplicate/unparseable) on >=2 pilot questions.  Transport errors are counted separately.
Scoring = number of distinct parsed vectors (== RR*81 by the bijection; official scorer not used)."""
import json, sys
from pathlib import Path
policy = sys.argv[1] if len(sys.argv) > 1 else "free"
d = Path(__file__).resolve().parent / "traces" / policy
BUD = [8, 16, 24, 32, 40, 48, 56, 64, 72, 81]
rows, cost, perq = [], 0.0, {}
for f in sorted(d.glob("*.jsonl")):
    tr = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    cum, n = [], 0
    for t in tr: n += bool(t.get("new")); cum.append(n)
    waste = [t["slot"] for t in tr if not t.get("new") and "transport_error" not in t]
    terr = sum("transport_error" in t for t in tr)
    cost += sum(t.get("usage", {}).get("cost", 0) or 0 for t in tr)
    perq[f.stem] = (cum, waste, terr)
    print(f"{f.stem}: found {cum[-1]}/81  waste={len(waste)} at slots {waste}  transport_err={terr}")
if perq:
    print("\nfound@N 平均 (上限=N):")
    print("N     " + " ".join(f"{b:>6}" for b in BUD))
    print("mean  " + " ".join(f"{sum(c[min(b,len(c))-1] for c,_,_ in perq.values())/len(perq):6.2f}" for b in BUD))
    bad = sum(1 for _, w, _ in perq.values() if len(w) >= 2)
    print(f"\n题数={len(perq)} 浪费>=2 的题={bad}  总费用=${cost:.4f}  -> Gate1 {'PASS' if bad >= 2 else 'FAIL'}")
