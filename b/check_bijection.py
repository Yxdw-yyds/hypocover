"""Check the height-vector bijection: every valid structure = one vector in {1..3}^k."""
import json
import itertools
from pathlib import Path

root = Path(r"D:\Machines Learning Final\hypocover-deepseek")


def layers_to_vec(layers, positions):
    """Height of each occupied column from a layer-string structure."""
    heights = {p: 0 for p in positions}
    for l, row in enumerate(layers, start=1):
        for r in range(3):
            for c in range(3):
                if row[r * 3 + c] == "1" and (r, c) in heights:
                    heights[(r, c)] = l
    return tuple(heights[p] for p in positions)


for name, k in (("columns-2.json", 2), ("columns-3.json", 3)):
    d = json.loads((root / "data" / name).read_text())
    print(name, "sets:", len(d["observation_sets"]))
    bad = 0
    for s in d["observation_sets"]:
        obs = s["observation"]
        positions = [(i // 3, i % 3) for i, ch in enumerate(obs) if ch == "1"]
        assert len(positions) == k, (s["observation_id"], positions)
        gts = {layers_to_vec(g["layers"], positions) for g in s["ground_truth_structures"]}
        expect = set(itertools.product(range(1, 4), repeat=k))
        if gts != expect:
            bad += 1
            print("  MISMATCH", s["observation_id"], "missing", expect - gts, "extra", gts - expect)
    print("  |H_O| =", 3 ** k, " mismatches:", bad)
