"""Offline simulator of a *peaked* hypothesis generator (paper Sec. 4).

Used for tests and for a cheap sanity demo of the theory: it samples ground
truth hypotheses from a Zipf distribution over a fixed random order, emits
some invalid answers, follows prompt constraints when the caller passes them
via ``meta`` (a real LLM reads them from the prompt), and only partially
respects the "avoid prior hypotheses" instruction.

It never sees anything a real model would not: when called by a method it
uses ``meta['cell']`` (structural constraints that are also in the prompt)
and ``meta['history_keys']`` (the hypotheses already listed in the prompt).
"""
from __future__ import annotations

import random
import zlib

from .client import LLM, Completion


class MockLLM(LLM):
    is_mock = True

    def __init__(self, alpha: float = 1.3, p_invalid: float = 0.1,
                 p_avoid_history: float = 0.5, seed: int = 0):
        self.alpha, self.p_invalid, self.p_avoid = alpha, p_invalid, p_avoid_history
        self.seed = seed
        self.name = f"mock:zipf{alpha}"
        self.task = self.inst = None

    def bind(self, task, inst) -> "MockLLM":
        m = MockLLM(self.alpha, self.p_invalid, self.p_avoid, self.seed)
        m.task, m.inst = task, inst
        m.rng = random.Random(self.seed * 1_000_003 + zlib.crc32(inst.id.encode()))
        order = list(inst.gt)
        m.rng.shuffle(order)
        m.order = order
        m.weights = [1.0 / (r + 1) ** self.alpha for r in range(len(order))]
        return m

    def generate(self, prompt: str, *, meta: dict | None = None) -> Completion:
        assert self.task is not None, "call bind(task, inst) first"
        meta = meta or {}
        if self.rng.random() < self.p_invalid:
            return Completion(text="I think the answer is unclear.", prompt_tokens=len(prompt) // 4,
                              completion_tokens=8)
        cell = meta.get("cell")  # callable h -> bool
        pool = [(h, w) for h, w in zip(self.order, self.weights) if cell is None or cell(h)]
        hist = meta.get("history_keys") or set()
        if hist and self.rng.random() < self.p_avoid:
            fresh = [(h, w) for h, w in pool if self.task.key(h) not in hist]
            if fresh or cell is not None:
                pool = fresh
        if not pool:
            return Completion(text=self.task.none_token, prompt_tokens=len(prompt) // 4,
                              completion_tokens=2)
        h = self.rng.choices([p[0] for p in pool], weights=[p[1] for p in pool])[0]
        return Completion(text=self.task.render(h), prompt_tokens=len(prompt) // 4,
                          completion_tokens=20)
