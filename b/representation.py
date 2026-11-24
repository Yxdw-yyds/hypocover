"""Height-vector representation for the frozen 3D benchmark (role B).

Each question shows a 3x3 top view.  Since blocks obey gravity, a valid
structure is exactly a choice of height in {1,2,3} for every occupied column.
So the whole hypothesis set H_O is {1,2,3}^k (k = 2 or 3 occupied columns),
and one hypothesis is a k-tuple of heights.  This module converts between that
tuple, the official "Layer n:" text the scorer expects, and the model output.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Question:
    id: str
    top_view: tuple[tuple[int, ...], ...]
    positions: tuple[tuple[int, int], ...]
    budget: int
    split: str
    group: str

    @property
    def k(self) -> int:
        return len(self.positions)


def _load_split_lists() -> dict:
    return json.loads((ROOT / "data" / "manifest.json").read_text())["groups"]


def load_questions(split: str) -> list[Question]:
    """Questions of one split, in manifest order (group 2 then 3)."""
    groups = _load_split_lists()
    out = []
    for gid, g in groups.items():
        file = json.loads((ROOT / "data" / g["file"]).read_text())
        by_id = {s["observation_id"]: s for s in file["observation_sets"]}
        for qid in g[split]:
            s = by_id[qid]
            obs = s["observation"]
            positions = tuple((i // 3, i % 3) for i, ch in enumerate(obs) if ch == "1")
            top = tuple(tuple(int(obs[r * 3 + c]) for c in range(3)) for r in range(3))
            out.append(Question(id=qid, top_view=top, positions=positions,
                                budget=int(g["budget"]), split=split, group=gid))
    return out


def all_vectors(k: int) -> list[tuple[int, ...]]:
    return list(itertools.product((1, 2, 3), repeat=k))


def vec_to_scorer_text(vec: tuple[int, ...], positions: tuple[tuple[int, int], ...]) -> str:
    """Height vector -> the 'Layer 1:\\n...' text the official scorer parses."""
    height = max(vec)
    lines = []
    for layer in range(1, height + 1):
        lines.append(f"Layer {layer}:")
        for r in range(3):
            row = []
            for c in range(3):
                if (r, c) in positions:
                    h = vec[positions.index((r, c))]
                    row.append("1" if h >= layer else "0")
                else:
                    row.append("0")
            lines.append(" ".join(row))
    return "\n".join(lines)


_VEC_RE = re.compile(r"H\s*=\s*\(([^)]*)\)", re.I)
_INT_RE = re.compile(r"(?<![0-9])[1-3](?![0-9])")


def parse_vec(text: str | None, k: int) -> tuple[int, ...] | None:
    """Pull a k-tuple of heights in {1,2,3} out of a model answer."""
    if not text:
        return None
    m = _VEC_RE.search(text)
    if m:
        nums = [int(x) for x in re.findall(r"[1-3]", m.group(1))]
        if len(nums) == k:
            return tuple(nums)
    nums = [int(x) for x in _INT_RE.findall(text.splitlines()[0] if text else "")]
    if len(nums) == k:
        return tuple(nums)
    nums = [int(x) for x in _INT_RE.findall(text)]
    if len(nums) == k:
        return tuple(nums)
    return None


def render_history(found: list[tuple[int, ...]]) -> str:
    return ", ".join("H=(" + ",".join(map(str, v)) + ")" for v in found)


def column_names(q: Question) -> list[str]:
    return [f"(row {r + 1}, col {c + 1})" for r, c in q.positions]
