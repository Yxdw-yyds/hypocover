"""Actions, complexity tiers, compliance check and history-only features for the strategy
layer on the OFFICIAL layered 3D format.

The official prompt is never changed; an action only appends one sentence to its end.
Nothing here uses the ground-truth answer set, except `gt_tier_counts`, which exists only
for the `static_stratified` baseline (the paper's method) and is labelled as such.
"""
from __future__ import annotations

from itertools import product

import numpy as np

ACTIONS = ("none", "low", "mid", "high")
TIERS = ("low", "mid", "high")
OUTCOMES = ("new", "dup", "invalid", "unparsed")
WINDOW = 5

HINTS = {
    "low": "The total number of blocks in your structure must be fewer than {t}.",
    "mid": "The total number of blocks in your structure must be exactly {t}.",
    "high": "The total number of blocks in your structure must be more than {t}.",
}


def k_of(observation: str) -> int:
    """Number of occupied top-view cells (k); read straight from the observation."""
    return str(observation).count("1")


# PILOT actions (not part of the frozen protocol): target truncation and repeats instead of complexity
PILOT_HINTS = {
    "brief": "Do not write any explanation, reasoning or verification. Reply with only the structure in the required format.",
    "differ": "Your structure must be different from every structure listed above (including the invalid ones). Do not repeat any of them.",
    "brief_differ": "Do not write any explanation, reasoning or verification; reply with only the structure in the required format. "
                    "Your structure must be different from every structure listed above (including the invalid ones).",
}


def hint_text(action: str, k: int) -> str:
    if action == "none":
        return ""
    if action.startswith("exact:"):  # paper-style: exactly c blocks
        return f"The total number of blocks in your structure must be exactly {int(action[6:])}."
    if action in PILOT_HINTS:
        return PILOT_HINTS[action]
    return HINTS[action].format(t=2 * k)


def add_hint(prompt: str, action: str, k: int) -> str:
    h = hint_text(action, k)
    return prompt if not h else prompt.rstrip() + "\n\n        Additional requirement: " + h + "\n"


def count_blocks(structure) -> int:
    return int(sum(sum(sum(row) for row in layer) for layer in structure.layers))


def tier_of(blocks: int, k: int) -> str:
    return "low" if blocks < 2 * k else ("mid" if blocks == 2 * k else "high")


def complies(action: str, k: int, blocks: int) -> bool:
    if action.startswith("exact:"):
        return blocks == int(action[6:])
    if action == "none" or action in PILOT_HINTS:
        return True  # no block-count constraint to obey
    return tier_of(blocks, k) == action


def gt_tier_counts(k: int, max_height: int = 3) -> dict[str, int]:
    """Size of each tier in the answer space (height vectors). Uses GT structure: baseline only."""
    c = {t: 0 for t in TIERS}
    for v in product(range(1, max_height + 1), repeat=k):
        c[tier_of(sum(v), k)] += 1
    return c


# ---------------------------------------------------------------- non-learned policies
def choose_tier_cycle(step: int) -> str:
    return ("high", "mid", "low")[step % 3]


def choose_random_level(rng) -> str:
    return TIERS[int(rng.integers(3))]


def choose_random_action(rng) -> str:
    return ACTIONS[int(rng.integers(len(ACTIONS)))]


def exact_counts(k: int, max_height: int = 3) -> dict[int, int]:
    """Answer-space size at each exact block count c (GT structure: paper baseline only)."""
    c: dict[int, int] = {}
    for v in product(range(1, max_height + 1), repeat=k):
        c[sum(v)] = c.get(sum(v), 0) + 1
    return dict(sorted(c.items()))


def choose_paper_stratified(step: int, budget: int, k: int) -> str:
    """Paper sec. 6.5 read literally: iterate over exact complexity levels c = c_min..c_max,
    prompting for exactly c blocks; the query budget of each level is proportional to the
    ground-truth count at that level (largest-remainder rounding, ascending order).
    USES GROUND-TRUTH COUNTS. The visiting order (ascending) is our reading of 'iterate over levels'."""
    counts = exact_counts(k); total = sum(counts.values())
    raw = {c: n / total * budget for c, n in counts.items()}
    quota = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - quota[c], reverse=True)[: budget - sum(quota.values())]:
        quota[c] += 1
    i = step
    for c in sorted(quota):
        if i < quota[c]:
            return f"exact:{c}"
        i -= quota[c]
    return f"exact:{max(quota)}"


def choose_tier_static(step: int, k: int) -> str:
    """OUR coarse (3-tier) variant of the above: budget proportional to GT tier sizes.
    Deterministic proportional schedule (largest deficit first; ties HIGH, MID, LOW). Uses GT counts."""
    counts = gt_tier_counts(k)
    total = sum(counts.values())
    assigned = {t: 0 for t in TIERS}
    pick = "high"
    for i in range(step + 1):
        pick = max(("high", "mid", "low"), key=lambda t: (counts[t] / total * (i + 1) - assigned[t]))
        assigned[pick] += 1
    return pick


# ---------------------------------------------------------------- history-only features
FEATURE_NAMES = (
    ["bias", "step_frac", "recent_new", "recent_invalid", "recent_dup"]
    + [f"new_{t}" for t in TIERS] + [f"invalid_{t}" for t in TIERS]
    + [f"last_{a}" for a in ACTIONS] + ["last_new", "k3"]
)


def features(history: list[dict], step: int, budget: int, k: int) -> np.ndarray:
    """history = past steps [{'action','outcome','tier'}...]. Uses only the model's own past
    outputs and the budget; never tier sizes and never the answer set."""
    recent = history[-WINDOW:]
    n = len(recent)
    rate = lambda o: (sum(h["outcome"] == o for h in recent) / n) if n else 0.0
    new_c = {t: 0 for t in TIERS}
    bad_c = {t: 0 for t in TIERS}
    for h in history:
        if h.get("tier") in TIERS:
            if h["outcome"] == "new":
                new_c[h["tier"]] += 1
            elif h["outcome"] == "invalid":
                bad_c[h["tier"]] += 1
    last = history[-1] if history else None
    x = [1.0, step / max(budget, 1), rate("new") if n else 0.5, rate("invalid"), rate("dup")]
    x += [new_c[t] / max(budget, 1) for t in TIERS] + [bad_c[t] / max(budget, 1) for t in TIERS]
    x += [1.0 if (last and last["action"] == a) else 0.0 for a in ACTIONS]
    x += [1.0 if (last and last["outcome"] == "new") else 0.0, 1.0 if k >= 3 else 0.0]
    assert len(x) == len(FEATURE_NAMES)
    return np.array(x, dtype=float)


def choose_learned(model: dict, history: list[dict], step: int, budget: int, k: int) -> str:
    """Greedy over per-action logistic predictions of P(new valid structure)."""
    x = features(history, step, budget, k)
    best, best_p = "none", -1.0
    for a in model["actions"]:
        w = np.array(model["weights"][a], dtype=float)
        p = 1.0 / (1.0 + np.exp(-float(x @ w)))
        if p > best_p + 1e-12:
            best, best_p = a, p
    return best
