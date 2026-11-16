"""Method B reward: multiplicity-normalised coverage reward for GRPO.

For one prompt (observations + a history of prior hypotheses) GRPO samples a
group of G completions. Each completion j gets

    r_j = -1                    unparseable answer
          -0.5                  parses but contradicts the observations
           0                    valid but already in the prompt's history
           1 / m_j              valid and new, where m_j = how many completions
                                in the same group gave the same canonical key

Why 1/m: the total reward a group can collect equals the number of *distinct*
new valid hypotheses it found, i.e. the marginal coverage of the group. A
popular answer that every sample repeats is shared out, so its per-sample
reward falls; a rare valid answer keeps the full point. The policy gradient
therefore moves mass from the head of p(h) towards the tail, which is the
quantity that limits Recovery in paper Sec. 4 (E[C_N] <= K + N(1 - alpha)).
Set ``share=False`` for the plain "new and valid = 1" ablation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from hypocover.tasks import make_task

R_PARSE, R_INVALID, R_DUP = -1.0, -0.5, 0.0


@lru_cache(maxsize=None)
def get_task(name: str, path: str):
    return make_task(name, path)


@lru_cache(maxsize=None)
def get_instance(name: str, path: str, inst_id: str):
    for i in get_task(name, path).all_instances():
        if i.id == inst_id:
            return i
    raise KeyError(inst_id)


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return str(c)


def score_group(task, inst, history_keys: set, texts: list[str], share: bool = True) -> list[float]:
    keys, base = [], []
    for t in texts:
        h = task.parse(t)
        if h is None:
            keys.append(None); base.append(R_PARSE); continue
        try:
            ok = task.is_valid(h, inst)
            k = task.key(h)
        except Exception:
            ok, k = False, None
        if not ok or k is None:
            keys.append(None); base.append(R_INVALID); continue
        if str(k) in history_keys:
            keys.append(None); base.append(R_DUP); continue
        keys.append(str(k)); base.append(1.0)
    mult = Counter(k for k in keys if k is not None)
    return [b / mult[k] if (share and k is not None) else b for b, k in zip(base, keys)]


def coverage_reward(completions, task_name, data_path, inst_id, history_keys, group_id,
                    share: bool = True, **_) -> list[float]:
    """TRL reward_funcs signature: extra dataset columns arrive as lists."""
    groups: dict[str, list[int]] = defaultdict(list)
    for j, g in enumerate(group_id):
        groups[g].append(j)
    out = [0.0] * len(completions)
    for g, idx in groups.items():
        j0 = idx[0]
        task = get_task(task_name[j0], data_path[j0])
        inst = get_instance(task_name[j0], data_path[j0], inst_id[j0])
        rs = score_group(task, inst, set(history_keys[j0]), [_text(completions[j]) for j in idx], share)
        for j, r in zip(idx, rs):
            out[j] = r
    return out


def coverage_reward_noshare(completions, **kw) -> list[float]:
    return coverage_reward(completions, share=False, **kw)
