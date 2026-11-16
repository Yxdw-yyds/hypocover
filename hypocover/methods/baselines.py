"""Baselines from the paper.

direct      independent samples, no history (pure sampler; Sec. 4 setting)
memory      official HypoSpace protocol: prior proposals listed in the prompt
            ("do not repeat if avoidable")  <- main baseline, = paper Tables 1-3
stratified  complexity-stratified decoding (paper Sec. 6.5). The paper splits
            the budget across complexity levels in proportion to the
            ground-truth distribution (``alloc='oracle'``), which uses H_O.
            ``alloc='uniform'`` is our oracle-free variant.
"""
from __future__ import annotations

from collections import Counter

from ..llm import LLM, Usage
from ..tasks import Instance, Task
from .base import Method, RunResult, history_from, query_once


class Direct(Method):
    name = "direct"

    def run(self, task: Task, inst: Instance, llm: LLM, budget: int) -> RunResult:
        usage, atts = Usage(), []
        prompt = task.build_prompt(inst, [])
        for _ in range(budget):
            atts.append(query_once(task, inst, llm, prompt, usage))
        return RunResult(atts, usage)


class Memory(Method):
    name = "memory"

    def run(self, task: Task, inst: Instance, llm: LLM, budget: int) -> RunResult:
        usage, atts = Usage(), []
        for _ in range(budget):
            hist = history_from(task, atts)
            prompt = task.build_prompt(inst, hist)
            meta = {"history_keys": {task.key(h) for h in hist}}
            atts.append(query_once(task, inst, llm, prompt, usage, meta=meta))
        return RunResult(atts, usage)


def _levels(task: Task, inst: Instance) -> list[int]:
    if hasattr(task, "complexity_range"):
        return list(task.complexity_range(inst))
    return list(range(0, task.max_complexity() + 1))


def allocate(weights: dict[int, float], budget: int) -> dict[int, int]:
    """Largest-remainder split of ``budget`` proportional to ``weights``."""
    tot = sum(weights.values()) or 1.0
    raw = {c: budget * w / tot for c, w in weights.items()}
    alloc = {c: int(r) for c, r in raw.items()}
    rest = budget - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:rest]:
        alloc[c] += 1
    return alloc


class Stratified(Method):
    name = "stratified"

    def run(self, task: Task, inst: Instance, llm: LLM, budget: int) -> RunResult:
        mode = self.kwargs.get("alloc", "oracle")
        if mode == "oracle":
            w = Counter(task.complexity(h) for h in inst.gt)
        else:
            w = {c: 1.0 for c in _levels(task, inst)}
        alloc = allocate(dict(w), budget)
        usage, atts = Usage(), []
        for c in sorted(alloc):
            for _ in range(alloc[c]):
                hist = history_from(task, atts)
                prompt = task.build_prompt(inst, hist, extra="Additional constraint: "
                                           + task.describe_complexity(c))
                meta = {"history_keys": {task.key(h) for h in hist},
                        "cell": (lambda h, c=c: task.complexity(h) == c)}
                atts.append(query_once(task, inst, llm, prompt, usage, meta=meta, tag=f"c={c}"))
        return RunResult(atts, usage, info={"alloc": alloc, "mode": mode})
