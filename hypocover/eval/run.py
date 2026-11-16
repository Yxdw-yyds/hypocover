"""Run one (task, dataset, method, model) configuration.

Example:
  python -m hypocover.eval.run --task causal --data data/causal/n5.json \
      --method memory --model openrouter:openai/gpt-4o-mini --n-instances 30

Writes results/<run_name>/instances.jsonl (one line per instance, resumable)
and results/<run_name>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from ..llm import make_llm
from ..methods import make_method
from ..metrics import aggregate, score
from ..tasks import make_task

SIMPLE_THRESHOLD = {"causal": 3, "boolean": 3}  # paper App. D: <=3 edges / <=3 ops


def run_one(task, method, llm, inst, budget_mult: float) -> dict:
    budget = max(1, int(round(inst.size * budget_mult)))
    t0 = time.time()
    res = method.run(task, inst, llm.bind(task, inst), budget)
    thr = SIMPLE_THRESHOLD.get(task.name)
    gt_c = {task.key(h): task.complexity(h) for h in inst.gt} if thr is not None else None
    s = score(res.attempts, inst.gt_keys, budget, gt_c, thr)
    return {
        "id": inst.id, "n_gt": inst.size, "budget": budget,
        **{k: v for k, v in asdict(s).items() if k != "rr_curve"},
        "rr_curve": [round(x, 4) for x in s.rr_curve],
        "usage": res.usage.to_dict(), "aux_calls": res.aux_calls, "info": res.info,
        "seconds": round(time.time() - t0, 2),
        "attempts": [a.to_json(task) for a in res.attempts],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["causal", "3d", "boolean"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", default="memory")
    ap.add_argument("--method-args", default="{}", help='JSON, e.g. \'{"alloc": "uniform"}\'')
    ap.add_argument("--model", default="mock:zipf")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--n-instances", type=int, default=30)
    ap.add_argument("--min-gt", type=int, default=2, help="skip instances with |H_O| below this")
    ap.add_argument("--max-gt", type=int, default=None)
    ap.add_argument("--budget-mult", type=float, default=1.0, help="queries = mult * |H_O|")
    ap.add_argument("--seed", type=int, default=33550336)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="results dir (default: results/<auto name>)")
    args = ap.parse_args(argv)

    task = make_task(args.task, args.data)
    margs = json.loads(args.method_args)
    method = make_method(args.method, **margs)
    llm_kw = {"seed": args.seed} if args.model.startswith("mock") else \
        {"temperature": args.temperature, "max_tokens": args.max_tokens}
    llm = make_llm(args.model, **llm_kw)

    insts = task.sample_instances(args.n_instances, seed=args.seed,
                                  min_size=args.min_gt, max_size=args.max_gt)
    tag = "" if not margs else "_" + "_".join(f"{k}{v}" for k, v in sorted(margs.items()))
    name = f"{args.task}-{Path(args.data).stem}-{args.method}{tag}-{args.model.replace('/', '_').replace(':', '-')}"
    out = Path(args.out or f"results/{name}")
    out.mkdir(parents=True, exist_ok=True)
    inst_file = out / "instances.jsonl"
    done = {}
    if inst_file.exists():
        for line in inst_file.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [i for i in insts if i.id not in done]
    print(f"[{name}] {len(insts)} instances (|H_O| mean "
          f"{sum(i.size for i in insts) / max(len(insts), 1):.1f}), {len(todo)} to run")

    with open(inst_file, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_one, task, method, llm, i, args.budget_mult): i for i in todo}
        for fu in tqdm(as_completed(futs), total=len(futs), desc=name[:40]):
            r = fu.result()
            done[r["id"]] = r
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()

    rows = [done[i.id] for i in insts if i.id in done]
    summ = aggregate(rows)
    summ.update({
        "run": name, "task": args.task, "data": args.data, "method": args.method,
        "method_args": margs, "model": args.model, "temperature": args.temperature,
        "budget_mult": args.budget_mult, "seed": args.seed, "min_gt": args.min_gt,
        "mean_n_gt": sum(r["n_gt"] for r in rows) / max(len(rows), 1),
        "usage": {k: sum(r["usage"][k] for r in rows) for k in ("calls", "prompt_tokens", "completion_tokens", "cost")},
    })
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in summ.items() if k in ("vr", "nr", "rr", "rr_simple", "rr_complex", "usage")}))
    return summ


if __name__ == "__main__":
    main()
