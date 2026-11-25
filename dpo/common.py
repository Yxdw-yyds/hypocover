"""Shared utilities, strict split loading, canonical formatting, and structural features."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "upstream" / "3d") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "3d"))

from run_3d_benchmark import Benchmark3D, Structure3D  # noqa: E402


@dataclass
class QuestionSpec:
    qid: str
    group: str
    k: int
    budget: int
    obs: str
    bench: Benchmark3D
    gt_structures: Optional[List[Structure3D]] = None


def parse_compact_gt(gt_entry: dict) -> Structure3D:
    """Convert a dataset ground_truth_structures entry into a normalized Structure3D."""
    raw_layers = gt_entry["layers"]
    if raw_layers and isinstance(raw_layers[0], str):
        grid_size = int(len(raw_layers[0]) ** 0.5)
        layers = []
        for s in raw_layers:
            layer = [
                [int(s[r * grid_size + c]) for c in range(grid_size)]
                for r in range(grid_size)
            ]
            layers.append(layer)
        return Structure3D(layers).normalize()
    return Structure3D(raw_layers).normalize()


def load_eval_questions(split: str, ids: Optional[List[str]] = None) -> List[QuestionSpec]:
    """Load questions for inference/evaluation WITHOUT ground-truth structures."""
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split!r}")
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    groups = manifest["groups"]
    out: List[QuestionSpec] = []
    for gid in sorted(groups.keys()):
        g = groups[gid]
        bench = Benchmark3D(str(ROOT / "data" / g["file"]))
        items = {it["observation_id"]: it for it in bench.all_observation_sets}
        for qid in g[split]:
            if ids and qid not in ids:
                continue
            obs = items[qid]["observation"]
            k = str(obs).count("1")
            out.append(
                QuestionSpec(
                    qid=qid,
                    group=gid,
                    k=k,
                    budget=int(g["budget"]),
                    obs=obs,
                    bench=bench,
                    gt_structures=None,
                )
            )
    return out


def load_train_questions_with_gt(split: str = "train") -> List[QuestionSpec]:
    """Load the 84 train questions WITH ground-truth structures.

    Hard anti-leakage guard: raises AssertionError if called on validation or test.
    """
    assert split == "train", (
        f"ANTI-LEAKAGE GUARD: ground-truth structures may ONLY be loaded for 'train', got {split!r}"
    )
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    groups = manifest["groups"]
    out: List[QuestionSpec] = []
    for gid in sorted(groups.keys()):
        g = groups[gid]
        bench = Benchmark3D(str(ROOT / "data" / g["file"]))
        items = {it["observation_id"]: it for it in bench.all_observation_sets}
        for qid in g["train"]:
            item = items[qid]
            obs = item["observation"]
            k = str(obs).count("1")
            gts = [parse_compact_gt(gt) for gt in item["ground_truth_structures"]]
            out.append(
                QuestionSpec(
                    qid=qid,
                    group=gid,
                    k=k,
                    budget=int(g["budget"]),
                    obs=obs,
                    bench=bench,
                    gt_structures=gts,
                )
            )
    return out


def canonical_response(struct: Structure3D) -> str:
    """Format a Structure3D in the exact official layered voxel representation."""
    norm = struct.normalize()
    return "Structure:\n" + norm.to_string()


def parse_and_normalize(bench: Benchmark3D, text: str) -> Optional[Structure3D]:
    """Parse an LLM response with the official upstream parser and normalize."""
    try:
        st = bench.parse_llm_response(text)
        if st is None:
            return None
        return st.normalize()
    except Exception:
        return None


def count_blocks(struct: Structure3D) -> int:
    """Total number of unit cubes across all layers."""
    return int(sum(int(np.sum(layer)) for layer in struct.layers))


def tier_of(blocks: int, k: int) -> str:
    """Complexity tier matching c4/layered_actions.py: low (<2k), mid (=2k), high (>2k)."""
    if blocks < 2 * k:
        return "low"
    if blocks == 2 * k:
        return "mid"
    return "high"


def voxel_tensor(struct: Structure3D, max_height: int = 3, grid_size: int = 3) -> np.ndarray:
    """Convert Structure3D to a fixed (max_height, grid_size, grid_size) binary array."""
    arr = np.zeros((max_height, grid_size, grid_size), dtype=np.int32)
    norm = struct.normalize()
    for z, layer in enumerate(norm.layers[:max_height]):
        h, w = min(layer.shape[0], grid_size), min(layer.shape[1], grid_size)
        arr[z, :h, :w] = layer[:h, :w]
    return arr


def voxel_hamming(s1: Structure3D, s2: Structure3D, max_height: int = 3) -> int:
    """3D voxel L1 / Hamming distance between two structures."""
    return int(np.sum(np.abs(voxel_tensor(s1, max_height) - voxel_tensor(s2, max_height))))


def format_chat_prompt(tokenizer, user_prompt: str) -> str:
    """Wrap the exact official benchmark prompt in the model's single-turn chat template."""
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
