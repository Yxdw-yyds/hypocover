"""Rebuild predictions from committed traces with the fixed (last-vector) parser."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import load_questions, parse_vec, vec_to_scorer_text

root = Path(__file__).resolve().parents[1]
questions = {split: {q.id: q for q in load_questions(split)}
             for split in ("validation", "test")}

changed = 0
for policy in ("free", "coord_fixed"):
    for split, qmap in questions.items():
        rows = []
        for qid, q in qmap.items():
            path = root / "b" / "traces" / policy / f"{qid}.jsonl"
            recs = [json.loads(line) for line in path.open(encoding="utf-8")]
            recs.sort(key=lambda r: r["slot"])
            assert [r["slot"] for r in recs] == list(range(1, q.budget + 1)), (qid, policy)
            for r in recs:
                vec = parse_vec(r.get("response"), q.k)
                if (vec is None) != (r.get("vec") is None) or (
                        vec is not None and tuple(r["vec"]) != vec):
                    changed += 1
                    print(f"changed {policy} {qid} slot {r['slot']}: {r.get('vec')} -> {vec}")
                rows.append({"question": qid, "slot": r["slot"],
                             "scorer_text": vec_to_scorer_text(vec, q.positions) if vec else ""})
        rows.sort(key=lambda r: (r["question"], r["slot"]))
        out = root / f"predictions-{split}-{policy}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}/{policy}: {len(rows)} slots -> {out.name}")

print("changed slots:", changed)
