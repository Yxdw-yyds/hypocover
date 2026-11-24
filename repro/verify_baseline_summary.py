"""Verify our recomputation against the branch's summary.json (per group)."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
manifest = json.loads((root / "data" / "manifest.json").read_text())["groups"]
summary = json.loads((root / "summary.json").read_text())["splits"]
metrics = ["parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate"]

for split in ("validation", "test"):
    cases = json.loads((root / f"{split}-score.json").read_text())["cases"]
    by_q = {c["observation_set_id"]: c for c in cases}
    for group in ("2", "3", "0"):
        if group == "0":
            qs = [q for g in manifest.values() for q in g[split]]
        else:
            qs = manifest[group][split]
        got = [mean(by_q[q][m] for q in qs) for m in metrics]
        want = summary[split][group]["metrics"]
        diff = max(abs(a - b) for a, b in zip([x * 100 for x in got], want))
        print(f"{split:10s} group={group}  n={len(qs):2d}  "
              f"got={[round(x * 100, 2) for x in got]}  want={want}  maxdiff={diff:.4f}")
