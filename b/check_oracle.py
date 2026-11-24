"""End-to-end format check: feed all valid vectors as predictions -> score should be 1.0."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import all_vectors, load_questions, vec_to_scorer_text

split = "validation"
rows = []
for q in load_questions(split):
    vs = all_vectors(q.k)
    assert len(vs) == q.budget, (q.id, len(vs), q.budget)
    for slot, v in enumerate(vs, start=1):
        rows.append({"question": q.id, "slot": slot,
                     "scorer_text": vec_to_scorer_text(v, q.positions)})

out = Path(f"predictions-{split}-oracle.jsonl")
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", out, len(rows), "rows")
