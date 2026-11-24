"""Free diagnostics on the committed traces: truncation, parse, cost, RR spread."""
import json
from pathlib import Path
from collections import Counter
from statistics import mean

root = Path(__file__).resolve().parent
manifest = json.loads((root / "data/manifest.json").read_text())["groups"]
split_of = {}
for split in ("train", "validation", "test"):
    for g in manifest.values():
        for q in g[split]:
            split_of[q] = split

rows = []
for p in sorted((root / "results").glob("*.jsonl")):
    q = p.stem
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        r["question"] = q
        r["split"] = split_of.get(q, "?")
        rows.append(r)

print("traces:", len(rows), "questions:", len({r['question'] for r in rows}))
print("cost $%.4f" % sum(r["cost"] for r in rows),
      "prompt", sum(r["usage"]["prompt_tokens"] for r in rows),
      "completion", sum(r["usage"]["completion_tokens"] for r in rows))
print("finish_reason:", Counter(r["finish_reason"] for r in rows))
print("providers:", Counter(r["provider"] for r in rows))

cases = {}
for split in ("validation", "test"):
    for c in json.loads((root / f"{split}-score.json").read_text())["cases"]:
        cases[c["observation_set_id"]] = c
rr = [c["recovery_rate"] for c in cases.values()]
print("cases:", len(cases), "RR mean %.3f min %.3f max %.3f" % (mean(rr), min(rr), max(rr)))
print("perfect RR cases:", sum(x >= 0.999 for x in rr),
      " zero RR cases:", sum(x <= 1e-9 for x in rr))
print("questions with parse<100%:", sum(c["parse_success_rate"] < 0.999 for c in cases.values()))
trunc = Counter(r["question"] for r in rows if r["finish_reason"] == "length")
print("questions with truncated traces:", len(trunc), "total truncated:", sum(trunc.values()))
