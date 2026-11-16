"""VR / NR / RR exactly as the upstream runners compute them, plus extras.

Upstream semantics (per instance, N = query budget):
  VR = #valid proposals / N
  NR = #distinct canonical keys among parsed proposals / N
       (Boolean: only in-space proposals count)
  RR = |{canonical keys of valid proposals} ∩ H_O| / |H_O|
A query whose answer never parsed (after retries) still counts in N.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Hashable

import numpy as np


@dataclass
class Attempt:
    """One budgeted query (one slot of the N-query budget)."""

    text: str | None = None
    hyp: Any = None
    key: Hashable | None = None
    valid: bool = False
    novelty_ok: bool = False      # counts towards NR (Boolean: in-space)
    exhausted: bool = False       # model answered EXHAUSTED
    tag: str = ""                 # method-specific label (cell, level, ...)

    def to_json(self, task) -> dict:
        d = {"text": self.text, "key": str(self.key) if self.key is not None else None,
             "valid": self.valid, "exhausted": self.exhausted, "tag": self.tag}
        d["hyp"] = task.render(self.hyp) if self.hyp is not None else None
        return d


@dataclass
class InstanceScore:
    vr: float
    nr: float
    rr: float
    n_queries: int
    n_gt: int
    rr_curve: list[float] = field(default_factory=list)
    rr_simple: float | None = None
    rr_complex: float | None = None


def score(attempts: list[Attempt], gt_keys: set, n_queries: int,
          gt_complexity: dict | None = None, threshold: int | None = None) -> InstanceScore:
    n = max(n_queries, 1)
    n_valid = sum(a.valid for a in attempts)
    nov = {a.key for a in attempts if a.hyp is not None and a.novelty_ok and a.key is not None}
    found: set = set()
    curve = []
    for a in attempts:
        if a.valid and a.key in gt_keys:
            found.add(a.key)
        curve.append(len(found) / len(gt_keys) if gt_keys else 0.0)
    s = InstanceScore(vr=n_valid / n, nr=len(nov) / n,
                      rr=len(found) / len(gt_keys) if gt_keys else 0.0,
                      n_queries=n_queries, n_gt=len(gt_keys), rr_curve=curve)
    if gt_complexity is not None and threshold is not None:
        simple = {k for k, c in gt_complexity.items() if c <= threshold}
        comp = set(gt_complexity) - simple
        s.rr_simple = len(found & simple) / len(simple) if simple else None
        s.rr_complex = len(found & comp) / len(comp) if comp else None
    return s


def aggregate(scores: list[dict]) -> dict:
    out = {"n_instances": len(scores)}
    for m in ("vr", "nr", "rr", "rr_simple", "rr_complex"):
        vals = [s[m] for s in scores if s.get(m) is not None]
        if vals:
            out[m] = float(np.mean(vals))
            out[m + "_std"] = float(np.std(vals))
    return out


__all__ = ["Attempt", "InstanceScore", "score", "aggregate", "asdict"]
