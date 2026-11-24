"""Fixed / random guidance policies over the height-vector representation.

The action is a *coordinate constraint*: "your new vector must differ from
every vector found so far in coordinate j" (1-based).  This guides exploration
along one column at a time without revealing the answer.  `fixed` cycles the
coordinate, `random` picks one uniformly, `free` applies no coordinate
constraint.  C can later learn this choice; here we only provide the fixed and
random baselines on the same action set.
"""
from __future__ import annotations

import random

from .representation import Question, column_names, render_history


def choose_action(policy: str, step: int, k: int, rng: random.Random | None = None) -> int | None:
    """Return a 0-based coordinate index, or None for 'no coordinate constraint'."""
    if policy == "free" or step == 0:
        return None
    if policy == "coord_fixed":
        return (step - 1) % k
    if policy == "coord_random":
        return (rng or random).randrange(k)
    raise ValueError(f"unknown policy {policy!r}")


def build_prompt(q: Question, found: list[tuple[int, ...]], policy: str,
                 step: int, rng: random.Random | None = None) -> tuple[str, str]:
    """Return (prompt, action label) for one query."""
    k = q.k
    action = choose_action(policy, step, k, rng)
    rows = "\n".join(" ".join(map(str, row)) for row in q.top_view)
    cols = column_names(q)

    lines = [
        "You are given the top view of a 3x3 stack of unit cubes (maximum height 3).",
        "A 1 means at least one cube is present in that column; a 0 means the column is empty.",
        "",
        "Top view (rows top to bottom):",
        rows,
        "",
        "Occupied columns, in this fixed order:",
        "; ".join(f"{i + 1}. {name}" for i, name in enumerate(cols)),
        "",
        f"Because of gravity, a valid structure is fully described by the height (1, 2 or 3) of each occupied column, "
        f"in the order above: H=(h1,...,h{k}). Any such vector is physically valid.",
    ]
    if found:
        lines += [
            "",
            "Height vectors already found:",
            render_history(found),
            "",
            "Your answer must be a vector different from every vector above.",
        ]
        if action is not None:
            lines.append(
                f"In particular it must differ in coordinate {action + 1} "
                f"(the height of column {action + 1} at {cols[action]}).")
    lines += [
        "",
        f"Reply with exactly one line and nothing else: H=(h1,...,h{k}) where every h is 1, 2 or 3.",
    ]
    return "\n".join(lines), ("free" if action is None else f"coord{action + 1}")
