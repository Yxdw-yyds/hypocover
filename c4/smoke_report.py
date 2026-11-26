"""Checklist report for the 2-question real smoke run (before the full collection).
  python -B c4/smoke_report.py c4/layered_smoke/train/random_action/rep9
Checks: length-truncation rate (should be ~0 at max_tokens 1024), hint compliance on real
responses (incl. ones with reasoning before 'Structure:'), unparsed rate, provider mix, cost."""
import collections
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) / "traces"
rows = [json.loads(l) for f in sorted(d.glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines()]
ok = [r for r in rows if "outcome" in r]
n = len(ok)
print(f"slots={len(rows)} usable={n} transport_errors={len(rows) - n}")
print("length-truncated:", f"{sum(r['finish_reason'] == 'length' for r in ok) / max(n, 1):.1%}", "(want ~0%)")
print("outcomes:", {k: f"{v / n:.0%}" for k, v in collections.Counter(r['outcome'] for r in ok).most_common()})
hinted = [r for r in ok if r["action"] != "none" and r["complies"] is not None]
print("compliance when hinted:", f"{sum(r['complies'] for r in hinted) / max(len(hinted), 1):.0%}", f"(n={len(hinted)})")
by = collections.defaultdict(list)
for r in hinted:
    by[r["action"]].append(r["complies"])
print("  by action:", {a: f"{sum(v) / len(v):.0%} (n={len(v)})" for a, v in sorted(by.items())})
pre = [r for r in ok if not r["response"].lstrip().startswith(("Structure", "Layer"))]
print("responses with text before the structure:", f"{len(pre) / max(n, 1):.0%}")
print("providers:", dict(collections.Counter(r["provider"] for r in ok)))
print("cost:", f"${sum(r['cost'] or 0 for r in ok):.4f}")
