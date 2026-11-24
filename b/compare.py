"""Compare baseline vs role-B policies on the frozen splits."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data" / "manifest.json").read_text())["groups"]
baseline = json.loads((root / "summary.json").read_text())["splits"]
mets = ["parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate"]

rows = []
for split in ("validation", "test"):
    for policy, fname in (("official prompt", None), ("B: free", f"{split}-score-free.json"),
                          ("B: coord_fixed", f"{split}-score-coord_fixed.json")):
        for group in ("2", "3", "0"):
            if policy == "official prompt":
                vals = baseline[split][group]["metrics"]
                n = baseline[split][group]["questions"]
                cost = baseline[split][group]["cost"]
            else:
                cases = json.loads((root / fname).read_text())["cases"]
                by_q = {c["observation_set_id"]: c for c in cases}
                qs = ([q for g in manifest.values() for q in g[split]] if group == "0"
                      else manifest[group][split])
                vals = [mean(by_q[q][m] for q in qs) * 100 for m in mets]
                n = len(qs)
                cost = None
            rows.append((split, policy, group, n, vals, cost))

print("| split | method | cols | n | parse | VR | NR | RR |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for split, policy, group, n, vals, cost in rows:
    cols = {"0": "all", "2": "2", "3": "3"}[group]
    print(f"| {split} | {policy} | {cols} | {n} | " +
          " | ".join(f"{v:.2f}" for v in vals) + " |")

# per-question misses for the free policy
print("\nmisses (free policy):")
for split in ("validation", "test"):
    cases = json.loads((root / f"{split}-score-free.json").read_text())["cases"]
    for c in cases:
        if c["recovery_rate"] < 1:
            print(f"  {split} {c['observation_set_id']}: RR={c['recovery_rate']:.3f} "
                  f"({c['n_recovered_gts']}/{c['n_ground_truths']})")
