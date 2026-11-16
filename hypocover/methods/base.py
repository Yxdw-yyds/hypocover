from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..llm import LLM, Usage
from ..metrics import Attempt
from ..tasks import Instance, Task


@dataclass
class RunResult:
    attempts: list[Attempt]
    usage: Usage
    aux_calls: int = 0               # LLM calls that do not produce a hypothesis
    info: dict = field(default_factory=dict)


class Method:
    name = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, task: Task, inst: Instance, llm: LLM, budget: int) -> RunResult:
        raise NotImplementedError


def query_once(task: Task, inst: Instance, llm: LLM, prompt: str, usage: Usage,
               meta: dict | None = None, allow_exhausted: bool = False,
               tag: str = "") -> Attempt:
    """One budgeted query with upstream retry semantics.

    Upstream re-asks the same prompt up to ``task.max_retries`` times when the
    answer does not parse; the retries cost tokens but not budget.
    """
    text = None
    for _ in range(task.max_retries):
        c = llm.generate(prompt, meta=meta)
        usage.add(c)
        text = c.text
        if text.startswith("Error querying"):
            continue
        if allow_exhausted and task.says_none(text):
            return Attempt(text=text, exhausted=True, tag=tag)
        h = task.parse(text)
        if h is not None:
            return make_attempt(task, inst, h, text, tag)
    return Attempt(text=text, tag=tag)


def make_attempt(task: Task, inst: Instance, h: Any, text: str, tag: str = "") -> Attempt:
    try:
        k = task.key(h)
    except Exception:
        k = None
    try:
        nov = task.counts_for_novelty(h)
    except Exception:
        nov = False
    try:
        valid = task.is_valid(h, inst)
    except Exception:
        valid = False
    return Attempt(text=text, hyp=h, key=k, valid=valid, novelty_ok=nov, tag=tag)


def history_from(task: Task, attempts: Sequence[Attempt],
                 filt: Callable[[Attempt], bool] | None = None) -> list:
    """Prior proposals in upstream order/format for the 'memory' prompt."""
    hist, seen = [], set()
    for a in attempts:
        if a.hyp is None or (filt and not filt(a)):
            continue
        if task.history_unique:
            if a.key in seen:
                continue
            seen.add(a.key)
        hist.append(a.hyp)
    return hist
