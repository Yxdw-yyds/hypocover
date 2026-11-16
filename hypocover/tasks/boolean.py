"""Boolean genetic-interaction programs over parental phenotypes x, y (upstream: boolean/)."""
from __future__ import annotations

import ast
import contextlib
import io
import re
from typing import Any, Sequence

from ._upstream import load
from .base import Instance, SplitFeature, Task

_OP_RE = re.compile(r"\b(AND|OR|NOT|NOR|XOR)\b", re.IGNORECASE)


class BooleanTask(Task):
    name = "boolean"
    max_retries = 5

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        up = load("boolean", "boolean_benchmark")
        self._Obs = up.BooleanObservation
        with contextlib.redirect_stdout(io.StringIO()):
            self.bench = up.BooleanBenchmarkRefined(dataset_path)
        self.operators = sorted(self.bench.operators)
        self.max_depth = int(self.bench.max_depth)
        self._instances: list[Instance] | None = None

    def all_instances(self) -> list[Instance]:
        if self._instances is None:
            out = []
            for i, s in enumerate(self.bench.all_observation_sets):
                obs = [self._Obs(o["inputs"], o["output"]) for o in s["observations"]]
                gts = s["ground_truth_expressions"]
                keys = set()
                for g in gts:
                    k = g.get("mechanistic_key")
                    keys.add(ast.literal_eval(k) if isinstance(k, str) else k)
                out.append(Instance(
                    id=str(s.get("observation_set_id", i)),
                    observations=obs,
                    gt=[g["formula"] for g in gts],
                    gt_keys=keys,
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

    def render(self, h: str) -> str:
        return f"Expression: {h}"

    # ---- scoring --------------------------------------------------------
    def key(self, h: str):
        return self.bench.get_expression_mechanistic_key(h)

    def is_valid(self, h: str, inst: Instance) -> bool:
        return bool(self.bench.validate_expression(h, inst.observations)[0])

    def counts_for_novelty(self, h: str) -> bool:
        return self.bench._in_space(h)

    # ---- structure ------------------------------------------------------
    def complexity(self, h: str) -> int:
        return len(_OP_RE.findall(h))

    def describe_complexity(self, c: int) -> str:
        if c == 0:
            return "The expression MUST be a single variable with no operators."
        return (f"The expression MUST use exactly {c} operator occurrence{'s' if c != 1 else ''} "
                f"(count every AND / OR / NOT / NOR / XOR token).")

    def max_complexity(self) -> int:
        # every internal node of a depth-d tree is one operator token
        return 2 ** self.max_depth - 1

    def split_features(self, inst: Instance) -> list[SplitFeature]:
        def top(h: str):
            k = self.key(h)
            if not k:
                return None
            return "VAR" if k[0] == "VAR" else k[0]

        tops = ["VAR"] + self.operators

        def describe_top(v):
            if v == "VAR":
                return "The expression MUST be a single bare variable."
            if v == "NOR":
                return "The outermost (top-level) operation MUST be NOR(...)."
            return f"The outermost (top-level) operator MUST be {v}."

        n_ops = SplitFeature("n_ops", list(range(0, self.max_complexity() + 1)),
                             self.complexity, self.describe_complexity)
        has_not = SplitFeature(
            "has_not", [True, False],
            lambda h: bool(re.search(r"\bNOT\b", h, re.I)),
            lambda v: ("The expression MUST contain at least one NOT." if v
                       else "The expression MUST NOT contain any NOT."))
        feats = [SplitFeature("top", tops, top, describe_top), n_ops]
        if "NOT" in self.operators:
            feats.append(has_not)
        return feats
