"""Turn the committed DeepSeek traces into the scorer's predictions JSONL.

For each question in a split, results/<question>.jsonl holds one row per slot
with the raw model `response`. score_predictions.py wants
{question, slot, scorer_text}; we map response -> scorer_text.
"""
from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent
groups = json.loads((root / "data/manifest.json").read_text())["groups"]

for split in ("validation", "test"):
    budgets = {q: g["budget"] for g in groups.values() for q in g[split]}
    rows = []
    for q, budget in budgets.items():
        path = root / "results" / f"{q}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing trace: {path}")
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        recs.sort(key=lambda r: r["slot"])
        got = [r["slot"] for r in recs]
        if got != list(range(1, budget + 1)):
            raise SystemExit(f"{q}: slots {got} != 1..{budget}")
        rows += [{"question": q, "slot": r["slot"], "scorer_text": r["response"]}
                 for r in recs]
    out = root / f"predictions-{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}: {len(budgets)} questions, {len(rows)} predictions -> {out.name}")
