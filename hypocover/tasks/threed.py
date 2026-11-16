"""Gravity-constrained 3D voxel reconstruction from a top view (upstream: 3d/)."""
from __future__ import annotations

import contextlib
import io
from typing import Any, Sequence

import numpy as np

from ._upstream import load
from .base import Instance, SplitFeature, Task


def _decode_layers(layers: list[str]) -> list[list[list[int]]]:
    g = int(round(len(layers[0]) ** 0.5))
    return [[[int(s[r * g + c]) for c in range(g)] for r in range(g)] for s in layers]


class ThreeDTask(Task):
    name = "3d"
    max_retries = 3
    history_unique = True

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        up = load("3d", "run_3d_benchmark")
        self._Structure3D = up.Structure3D
        with contextlib.redirect_stdout(io.StringIO()):
            self.bench = up.Benchmark3D(dataset_path)
        self.max_height = int(self.bench.metadata.get("max_height", 3))
        self._instances: list[Instance] | None = None

    def all_instances(self) -> list[Instance]:
        if self._instances is None:
            out = []
            for i, s in enumerate(self.bench.all_observation_sets):
                obs = s.get("observation", s.get("observations"))
                gt = []
                for g in s["ground_truth_structures"]:
                    layers = g["layers"]
                    if layers and isinstance(layers[0], str):
                        layers = _decode_layers(layers)
                    gt.append(self._Structure3D(layers).normalize())
                out.append(Instance(
                    id=str(s.get("observation_id", i)),
                    observations=obs,
                    gt=gt,
                    gt_keys={g.get_hash() for g in gt},
                    raw=s,
                ))
            self._instances = out
        return self._instances

    def top_view(self, inst: Instance) -> np.ndarray:
        return self.bench._parse_observation(inst.observations)

    # ---- generation -----------------------------------------------------
    def build_prompt(self, inst: Instance, history: Sequence[Any], extra: str = "") -> str:
        p = self.bench.create_prompt(inst.observations, prior_structures=list(history))
        return p + ("\n\n" + extra.strip() if extra else "")

    def parse(self, text: str):
        s = self.bench.parse_llm_response(text)
        return s.normalize() if s is not None else None

    def render(self, h) -> str:
        return "Structure:\n" + h.to_string()

    # ---- scoring --------------------------------------------------------
    def key(self, h):
        return h.get_hash()

    def is_valid(self, h, inst: Instance) -> bool:
        return self.bench.validate_structure_matches_observations(h, inst.observations)

    # ---- structure ------------------------------------------------------
    def complexity(self, h) -> int:
        return int(sum(int(np.sum(L)) for L in h.layers))

    def describe_complexity(self, c: int) -> str:
        return f"The structure MUST contain exactly {c} blocks in total (summed over all layers)."

    def complexity_range(self, inst: Instance) -> range:
        k = int(np.sum(self.top_view(inst)))
        return range(k, k * self.max_height + 1)

    def column_heights(self, h, inst: Instance) -> dict:
        tv = self.top_view(inst)
        out = {}
        for r, c in zip(*np.where(tv == 1)):
            out[(int(r), int(c))] = int(sum(int(L[r, c]) for L in h.layers
                                            if r < L.shape[0] and c < L.shape[1]))
        return out

    def split_features(self, inst: Instance) -> list[SplitFeature]:
        tv = self.top_view(inst)
        feats = []
        for r, c in zip(*np.where(tv == 1)):
            r, c = int(r), int(c)

            def fn(h, r=r, c=c):
                return int(sum(int(L[r, c]) for L in h.layers
                               if r < L.shape[0] and c < L.shape[1]))

            def describe(v, r=r, c=c):
                return (f"The column at row {r + 1}, column {c + 1} MUST have height exactly {v} "
                        f"(blocks in layers 1..{v} at that position, none above).")

            feats.append(SplitFeature(f"col:{r},{c}", list(range(1, self.max_height + 1)), fn, describe))
        return feats
