"""Causal DAG inference from single-node perturbations (upstream: causal/)."""
from __future__ import annotations

import contextlib
import io
from itertools import combinations
from typing import Any, Sequence

from ._upstream import load
from .base import Instance, SplitFeature, Task


class CausalTask(Task):
    name = "causal"
    max_retries = 5

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        up = load("causal", "run_causal_benchmark")
        self._CausalGraph = up.CausalGraph
        with contextlib.redirect_stdout(io.StringIO()):
            self.bench = up.CausalBenchmarkEnhanced(dataset_path)
        self.nodes: list[str] = list(self.bench.nodes)
        self.max_edges = self.bench.max_edges
        self._instances: list[Instance] | None = None

    # ---- data -----------------------------------------------------------
    def all_instances(self) -> list[Instance]:
        if self._instances is None:
            out = []
            for i, s in enumerate(self.bench.all_observation_sets):
                gt = [self._CausalGraph.from_dict(g) for g in s["ground_truth_graphs"]]
                out.append(Instance(
                    id=str(s.get("observation_set_id", i)),
                    observations=s["observations"],
                    gt=gt,
                    gt_keys={g.get_hash() for g in gt},
                    raw=s,
                ))
            self._instances = out
        return self._instances

    # ---- generation -----------------------------------------------------
    def build_prompt(self, inst: Instance, history: Sequence[Any], extra: str = "") -> str:
        p = self.bench.create_prompt(inst.observations, list(history))
        return p + ("\n\n" + extra.strip() if extra else "")

    def parse(self, text: str):
        return self.bench.parse_llm_response(text)

    def render(self, h) -> str:
        if not h.edges:
            return "Graph: No edges"
        return "Graph: " + ", ".join(f"{u}->{v}" for u, v in h.edges)

    # ---- scoring --------------------------------------------------------
    def key(self, h):
        return h.get_hash()

    def is_valid(self, h, inst: Instance) -> bool:
        return self.bench.validate_hypothesis(h, inst.observations)

    # ---- structure ------------------------------------------------------
    def complexity(self, h) -> int:
        return len(h.edges)

    def describe_complexity(self, c: int) -> str:
        return f"The graph MUST have exactly {c} edge{'s' if c != 1 else ''}."

    def max_complexity(self) -> int:
        n = len(self.nodes)
        full = n * (n - 1) // 2
        return min(full, self.max_edges) if self.max_edges is not None else full

    def split_features(self, inst: Instance) -> list[SplitFeature]:
        feats = []
        for a, b in combinations(self.nodes, 2):
            def fn(h, a=a, b=b):
                e = set(map(tuple, h.edges))
                if (a, b) in e:
                    return f"{a}->{b}"
                if (b, a) in e:
                    return f"{b}->{a}"
                return "none"

            def describe(v, a=a, b=b):
                if v == "none":
                    return f"There MUST be no edge between {a} and {b} (in either direction)."
                return f"The graph MUST contain the edge {v}."

            feats.append(SplitFeature(f"pair:{a}{b}", [f"{a}->{b}", f"{b}->{a}", "none"], fn, describe))
        return feats
