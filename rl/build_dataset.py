"""Build the GRPO prompt set for method B.

Each row = one instance + a random "history" of k hypotheses already found,
rendered with the official memory prompt. The model must produce a new valid
hypothesis. Histories are drawn from H_O itself, which is fine for *training*
data because we generate it; evaluation uses different difficulty levels.

  python -m rl.build_dataset --out data/rl/train.jsonl --rows 6000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hypocover.tasks import make_task

# Train on easy/medium levels; keep hard + large for evaluation only.
TRAIN_SOURCES = [
    ("causal", "data/causal/n4.json", 1.0),
    ("causal", "data/causal/n5.json", 1.0),
    ("3d", "data/3d/tp2.json", 1.0),
    ("boolean", "data/boolean/basic_d2.json", 0.5),
    ("boolean", "data/boolean/extended_d2.json", 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rl/train.jsonl")
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--max-history", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma list of task names to keep")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    srcs = []
    for name, path, w in TRAIN_SOURCES:
        if a.only and name not in a.only.split(","):
            continue
        if not Path(path).exists():
            print(f"skip missing {path}")
            continue
        task = make_task(name, path)
        insts = [i for i in task.all_instances() if i.size >= 2]
        srcs.append((name, path, w, task, insts))
        print(f"{path}: {len(insts)} instances")
    weights = [s[2] for s in srcs]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in range(a.rows):
            name, path, _, task, insts = rng.choices(srcs, weights)[0]
            inst = rng.choice(insts)
            k = rng.randint(0, min(inst.size - 1, a.max_history))
            hist = rng.sample(inst.gt, k)
            prompt = task.build_prompt(inst, hist)
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": prompt}],
                "task_name": name, "data_path": path, "inst_id": inst.id,
                "history_keys": [str(task.key(h)) for h in hist],
                "group_id": f"{r}",
                "n_gt": inst.size, "n_hist": k,
            }) + "\n")
    print(f"wrote {a.rows} rows to {a.out}")


if __name__ == "__main__":
    main()
