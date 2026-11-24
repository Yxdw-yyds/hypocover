"""Verify the free-policy traces re-parse to 0 duplicates / full coverage.

Read-only. Prints the corrected duplicate/missing counts and the raw answer of
the one slot that the old parser mis-read.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b.representation import parse_vec

root = Path(__file__).resolve().parents[1]
total_slots = dup_slots = 0
missing_any = []
for path in sorted((root / "b/traces/free").glob("*.jsonl")):
    recs = [json.loads(line) for line in path.open(encoding="utf-8")]
    k = len(recs[0]["vec"]) if recs[0]["vec"] else 3
    found = []
    for r in recs:
        total_slots += 1
        v = parse_vec(r.get("response"), k)
        if v is None or v in found:
            dup_slots += 1
        elif v not in found:
            found.append(v)
    missing = sorted(set(itertools.product((1, 2, 3), repeat=k)) - set(found))
    if missing:
        missing_any.append((path.stem, missing))

print(f"free traces: {len(list((root/'b/traces/free').glob('*.jsonl')))} questions, "
      f"{total_slots} slots, duplicates/parse-fail={dup_slots}, missing-questions={len(missing_any)}")
for qid, miss in missing_any:
    print("  missing", qid, miss)

path = root / "b/traces/free/g3_h3_b3_079.jsonl"
recs = [json.loads(line) for line in path.open(encoding="utf-8")]
print("\ng3_h3_b3_079 slot 27 raw response:")
print(repr(recs[-1]["response"]))
print("parsed:", parse_vec(recs[-1]["response"], 3))
