"""One-off: re-parse trace vec/new fields with the fixed (last-vector) parser.

Keeps the raw `response` untouched; when a derived field changes, the old value
is preserved as `vec_original` / `new_original`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import load_questions, parse_vec

root = Path(__file__).resolve().parents[1]
kmap = {q.id: q.k for split in ("validation", "test") for q in load_questions(split)}

changed = 0
for policy in ("free", "coord_fixed"):
    for path in sorted((root / "b/traces" / policy).glob("*.jsonl")):
        qid = path.stem
        k = kmap[qid]
        recs = [json.loads(line) for line in path.open(encoding="utf-8")]
        found = set()
        for r in recs:
            old_vec = tuple(r["vec"]) if r.get("vec") else None
            new_vec = parse_vec(r.get("response"), k)
            is_new = new_vec is not None and new_vec not in found
            if new_vec is not None:
                found.add(new_vec)
            if new_vec != old_vec or is_new != bool(r.get("new")):
                r.setdefault("vec_original", r.get("vec"))
                r.setdefault("new_original", r.get("new"))
                r["vec"] = list(new_vec) if new_vec is not None else None
                r["new"] = is_new
                r["parse_note"] = "re-parsed 2026-11-24: last H=(...) is the final answer"
                changed += 1
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("changed slots:", changed)
