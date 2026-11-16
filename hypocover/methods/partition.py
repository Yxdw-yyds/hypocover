"""Method A: adaptive partition search (training-free, oracle-free).

Idea (paper Sec. 4): a peaked generator keeps re-sampling the same head of
H_O, so coverage grows like K + N(1-alpha). Instead of asking for "another
hypothesis" over the whole space, we split the space into disjoint cells by
structural constraints and ask *inside* a cell. Each cell's prompt only lists
the hypotheses already found in that cell, so the model's mass is pushed
into regions it would otherwise ignore, and context stays short.

Loop (budget = N hypothesis-producing queries, same as baselines):
  1. pick an open leaf cell by a UCB score on its "new hypothesis" rate;
  2. query the model with the cell's constraints + the cell's own history;
     the model may answer EXHAUSTED if it believes the cell is empty;
  3. route the answer to the leaf it actually belongs to (constraint
     violations are not wasted);
  4. a cell that answers EXHAUSTED is closed; a cell that keeps returning
     duplicates is split on the structural feature that best balances its
     known members.

What it uses: the observations, the model, and structural features of the
*form* of a hypothesis (edge between A and B? column height? top operator?).
It never touches H_O. With ``use_validator=True`` it may also call the task's
consistency check to score cells (report this as a separate setting).

Owner: member B. Knobs worth ablating are in ``DEFAULTS``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM, Usage
from ..metrics import Attempt
from ..tasks import Instance, SplitFeature, Task
from .base import Method, RunResult, history_from, query_once

DEFAULTS = dict(
    ucb_c=1.0,            # exploration weight in the cell score
    split_after_dups=2,   # consecutive non-new answers before a cell splits
    warmup=2,             # plain queries at the root before the first split
    max_cell_history=40,  # cap on hypotheses listed per prompt
    use_validator=False,  # score cells by new *valid* answers instead of new answers
    allow_exhausted=True, # let the model close a cell with EXHAUSTED
    seed=0,
)


@dataclass
class Cell:
    constraints: tuple[tuple[int, Any], ...]  # (feature index, required value)
    tries: int = 0
    new: int = 0
    dup_streak: int = 0
    closed: bool = False
    children: list["Cell"] = field(default_factory=list)
    members: list[Attempt] = field(default_factory=list)  # distinct hyps routed here

    @property
    def leaf(self) -> bool:
        return not self.children


class PartitionSearch(Method):
    name = "partition"

    def __init__(self, **kwargs):
        super().__init__(**{**DEFAULTS, **kwargs})

    # ---- helpers --------------------------------------------------------
    def _satisfies(self, feats: list[SplitFeature], cell: Cell, h) -> bool:
        try:
            return all(feats[i].fn(h) == v for i, v in cell.constraints)
        except Exception:
            return False

    def _leaf_of(self, feats, root: Cell, h) -> Cell | None:
        node = root
        if not self._satisfies(feats, node, h):
            return None
        while node.children:
            nxt = next((c for c in node.children if self._satisfies(feats, c, h)), None)
            if nxt is None:
                return node
            node = nxt
        return node

    def _leaves(self, node: Cell) -> list[Cell]:
        return [node] if node.leaf else [l for c in node.children for l in self._leaves(c)]

    def _split(self, feats, cell: Cell, rng: random.Random) -> bool:
        used = {i for i, _ in cell.constraints}
        cands = [i for i in range(len(feats)) if i not in used]
        if not cands:
            return False

        def balance(i):  # entropy of known members over this feature's values
            cnt: dict = {}
            for a in cell.members:
                try:
                    v = feats[i].fn(a.hyp)
                except Exception:
                    continue
                cnt[v] = cnt.get(v, 0) + 1
            n = sum(cnt.values())
            if n == 0:
                return 0.0
            return -sum(c / n * math.log(c / n) for c in cnt.values())

        rng.shuffle(cands)
        best = max(cands, key=balance)
        for v in feats[best].values:
            child = Cell(cell.constraints + ((best, v),))
            child.members = [a for a in cell.members if self._safe(feats[best], a.hyp) == v]
            # inherited members are evidence the child is non-empty
            child.new = len(child.members)
            child.tries = len(child.members)
            cell.children.append(child)
        return True

    @staticmethod
    def _safe(f: SplitFeature, h):
        try:
            return f.fn(h)
        except Exception:
            return None

    def _extra(self, task: Task, feats, cell: Cell, allow_exhausted: bool) -> str:
        if not cell.constraints:
            lines = []
        else:
            lines = ["Additional constraints for THIS answer (all must hold, in addition to the observations):"]
            lines += [f"- {feats[i].describe(v)}" for i, v in cell.constraints]
        if allow_exhausted:
            lines.append(
                f"If you are certain that no further hypothesis exists that satisfies all observations"
                f"{' and these constraints' if cell.constraints else ''} and differs from every prior one "
                f"listed above, reply with the single word {task.none_token} instead.")
        return "\n".join(lines)

    # ---- main loop ------------------------------------------------------
    def run(self, task: Task, inst: Instance, llm: LLM, budget: int) -> RunResult:
        kw = self.kwargs
        rng = random.Random(kw["seed"])
        feats = task.split_features(inst)
        root = Cell(())
        usage, atts = Usage(), []
        seen: set = set()
        n_splits = 0

        for t in range(budget):
            leaves = [l for l in self._leaves(root) if not l.closed]
            if not leaves:  # everything closed: fall back to global memory query
                cell = root
                hist = history_from(task, atts)
            else:
                total = sum(l.tries for l in leaves) + 1

                def ucb(l: Cell):
                    mean = (l.new + 1) / (l.tries + 2)
                    return mean + kw["ucb_c"] * math.sqrt(math.log(total + 1) / (l.tries + 1))

                best = max(ucb(l) for l in leaves)
                cell = rng.choice([l for l in leaves if ucb(l) >= best - 1e-12])
                hist = [a.hyp for a in cell.members][-kw["max_cell_history"]:]

            allow_ex = kw["allow_exhausted"] and (cell is not root or t >= kw["warmup"])
            prompt = task.build_prompt(inst, hist, extra=self._extra(task, feats, cell, allow_ex))
            meta = {"history_keys": {task.key(h) for h in hist},
                    "cell": (lambda h, c=cell: self._satisfies(feats, c, h))}
            tag = ";".join(f"{feats[i].name}={v}" for i, v in cell.constraints) or "root"
            a = query_once(task, inst, llm, prompt, usage, meta=meta,
                           allow_exhausted=allow_ex, tag=tag)
            atts.append(a)
            cell.tries += 1

            if a.exhausted:
                if cell is root and leaves:
                    n_splits += self._split(feats, root, rng)
                else:
                    cell.closed = True
                continue

            is_new = a.hyp is not None and a.key is not None and a.key not in seen
            if kw["use_validator"]:
                is_new = is_new and a.valid
            if a.hyp is not None and a.key is not None and a.key not in seen:
                seen.add(a.key)
                home = self._leaf_of(feats, root, a.hyp) or cell
                home.members.append(a)
            if is_new and self._satisfies(feats, cell, a.hyp):
                cell.new += 1
                cell.dup_streak = 0
            else:
                cell.dup_streak += 1

            ready = t + 1 >= kw["warmup"]
            if cell.leaf and ready and cell.dup_streak >= kw["split_after_dups"]:
                if self._split(feats, cell, rng):
                    n_splits += 1
                else:
                    cell.closed = True

        return RunResult(atts, usage, info={"n_splits": n_splits,
                                            "n_leaves": len(self._leaves(root))})
