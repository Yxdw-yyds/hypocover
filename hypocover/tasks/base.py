"""Common task interface over the three HypoSpace domains.

Every method in ``hypocover.methods`` talks to a task only through this
interface, so a method written once runs on causal / 3d / boolean.

Faithfulness rule: parsing, validity, canonical keys and the default
("memory") prompt are delegated to the official upstream code, so numbers we
report are comparable with the paper.
"""
from __future__ import annotations

import json
import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Sequence


@dataclass
class Instance:
    """One observation set plus its enumerated admissible set H_O."""

    id: str
    observations: Any                 # task-specific, passed to upstream code
    gt: list[Any]                     # ground-truth hypotheses (task objects)
    gt_keys: set[Hashable]            # canonical keys of H_O
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.gt_keys)


@dataclass
class SplitFeature:
    """A structural property used by partition search to cut the space.

    ``fn(h)`` maps a hypothesis to one of ``values``; ``describe(v)`` renders
    the constraint "the hypothesis must have value v" as prompt text.
    Features only look at the *form* of a hypothesis, never at the
    observations, so using them does not leak the validator.
    """

    name: str
    values: list[Any]
    fn: Callable[[Any], Any]
    describe: Callable[[Any], str]


class Task(ABC):
    name: str = "base"
    max_retries: int = 3          # upstream re-asks on parse failure
    # Upstream "memory" protocol: causal/boolean show every parsed prior
    # proposal (duplicates included); 3d shows only the distinct ones.
    history_unique: bool = False
    # What a model answers when it believes no further hypothesis exists
    # (under the current constraints). Not "NONE": upstream causal parsing
    # reads "none" as the empty graph.
    none_token: str = "EXHAUSTED"

    def __init__(self, dataset_path: str):
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        self.dataset_path = dataset_path

    # ---- data -----------------------------------------------------------
    @abstractmethod
    def all_instances(self) -> list[Instance]: ...

    def sample_instances(self, n: int, seed: int = 0,
                         min_size: int = 1, max_size: int | None = None) -> list[Instance]:
        pool = [i for i in self.all_instances()
                if i.size >= min_size and (max_size is None or i.size <= max_size)]
        rng = random.Random(seed)
        return rng.sample(pool, min(n, len(pool)))

    # ---- generation -----------------------------------------------------
    @abstractmethod
    def build_prompt(self, inst: Instance, history: Sequence[Any],
                     extra: str = "") -> str:
        """Official prompt with ``history`` as prior predictions, plus an
        optional ``extra`` block of instructions (constraints etc.)."""

    @abstractmethod
    def parse(self, text: str) -> Any | None: ...

    @abstractmethod
    def render(self, h: Any) -> str:
        """Hypothesis -> text in the exact answer format the prompt asks for."""

    # ---- scoring --------------------------------------------------------
    @abstractmethod
    def key(self, h: Any) -> Hashable | None:
        """Canonical key used for uniqueness and recovery."""

    @abstractmethod
    def is_valid(self, h: Any, inst: Instance) -> bool: ...

    def counts_for_novelty(self, h: Any) -> bool:
        """Upstream Boolean only counts in-space expressions towards NR."""
        return True

    # ---- structure (for stratified / partition methods) ----------------
    @abstractmethod
    def complexity(self, h: Any) -> int: ...

    @abstractmethod
    def describe_complexity(self, c: int) -> str: ...

    @abstractmethod
    def split_features(self, inst: Instance) -> list[SplitFeature]: ...

    # ---- helpers --------------------------------------------------------
    def says_none(self, text: str) -> bool:
        if not text:
            return False
        last = text.strip().splitlines()[-1] if text.strip() else ""
        return re.search(rf"\b{self.none_token}\b", last.upper()) is not None
