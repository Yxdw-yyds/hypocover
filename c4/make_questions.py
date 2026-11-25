"""Build the frozen k=4 question bank: 4 occupied cells of the 3x3 grid, max height 3
(81 hypotheses).  Layouts are deduplicated under the 8 rotations/mirrors, then split
deterministically.  No API calls, no answers needed: H_O = {1,2,3}^4 by the bijection."""
import json, random
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "k4-questions.json"

def transforms(cells):
    def rot(c): r, k = c; return (k, 2 - r)
    def mir(c): r, k = c; return (r, 2 - k)
    out, cur = [], list(cells)
    for _ in range(4):
        out.append(tuple(sorted(cur))); out.append(tuple(sorted(mir(c) for c in cur)))
        cur = [rot(c) for c in cur]
    return out

grid = [(r, c) for r in range(3) for c in range(3)]
orbits = {}
for cells in combinations(grid, 4):
    orbits.setdefault(min(transforms(cells)), []).append(cells)
reps = sorted(orbits)
assert sum(len(v) for v in orbits.values()) == 126
print("layouts 126 -> orbits", len(reps))
rng = random.Random(33550336)
order = reps[:]; rng.shuffle(order)
split = {"train": order[:11], "validation": order[11:15], "test": order[15:]}
qs = []
for s, lst in split.items():
    for i, cells in enumerate(sorted(lst)):
        top = [[1 if (r, c) in cells else 0 for c in range(3)] for r in range(3)]
        qs.append({"id": f"k4_{s[:2]}_{i:02d}", "split": s, "top_view": top,
                   "positions": [list(c) for c in cells], "n_solutions": 81})
OUT.write_text(json.dumps({"seed": 33550336, "k": 4, "max_height": 3, "questions": qs}, indent=1))
print({s: len(v) for s, v in split.items()}, "->", OUT)
