"""Run the rest of the FROZEN protocol (c4/STRATEGY_LAYER_PLAN.md) unattended and resumable.

  python -B -m c4.run_all                 # real API (key from ~/.hypocover_openrouter_key or env)
  python -B -m c4.run_all --backend mock --out-root <dir> --results-dir <dir> --policy-file <dir>/policy.json

Steps (each is skipped if already finished; rerun the same command to resume after any stop):
  1  train trajectories, random_action, repeats 1 and 2
  2  fit the policy on TRAIN only and freeze it (an existing policy file is never refitted)
  3  validation: 6 arms x 2 repeats (the two repeats of an arm run in parallel)  -> analysis
  4  test, once, with the frozen policy                                           -> analysis
Spend is bounded by --max-cost per process and --total-cap over c4/spend_ledger.jsonl.
Progress is appended to <results-dir>/run_all.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("free_official", "paper_stratified", "tier_static", "random_level", "tier_cycle", "learned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "deepseek", "mock"])
    ap.add_argument("--model", default=None); ap.add_argument("--ledger", default=None)
    ap.add_argument("--price-in", type=float, default=0.0); ap.add_argument("--price-out", type=float, default=0.0); ap.add_argument("--price-hit", type=float, default=0.0)
    ap.add_argument("--out-root", default=str(ROOT / "c4" / "layered"))
    ap.add_argument("--results-dir", default=str(ROOT / "c4"))
    ap.add_argument("--policy-file", default=str(ROOT / "c4" / "policy.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-cost", type=float, default=1.0)
    ap.add_argument("--total-cap", type=float, default=3.5)
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    res = Path(a.results_dir); res.mkdir(parents=True, exist_ok=True)
    log = (res / "run_all.log").open("a", encoding="utf-8")

    def say(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    def runner(split, policy, rep, *extra):
        return [sys.executable, "-B", "-m", "c4.run_layered", "--split", split, "--policy", policy, "--repeat", str(rep),
                "--backend", a.backend, "--out-root", a.out_root, "--workers", str(a.workers),
                "--max-cost", str(a.max_cost), "--total-cap", str(a.total_cap), "--model-file", a.policy_file, *extra,
                *(["--model", a.model] if a.model else []), *(["--ledger", a.ledger] if a.ledger else []),
                "--price-in", str(a.price_in), "--price-out", str(a.price_out), "--price-hit", str(a.price_hit)]

    def run_parallel(cmds, label):
        procs = [subprocess.Popen(c, cwd=ROOT) for c in cmds]
        codes = [p.wait() for p in procs]
        if any(codes):
            say(f"STOPPED at {label}: exit codes {codes}. Rerun the same command to resume.")
            sys.exit(1)

    def analyze(split, name):
        out = subprocess.run([sys.executable, "-B", "-m", "c4.analyze_layered", "--split", split, "--root", a.out_root],
                             cwd=ROOT, capture_output=True, text=True)
        text = out.stdout + (("\nSTDERR:\n" + out.stderr[-1500:]) if out.returncode else "")
        (res / name).write_text(text, encoding="utf-8")
        say(f"analysis -> {res / name} (exit {out.returncode})")
        print(text, flush=True)
        if out.returncode:
            sys.exit(1)

    say(f"=== run_all start (backend={a.backend}) ===")
    # 1. training trajectories (two repeats in parallel; finished questions are skipped)
    run_parallel([runner("train", "random_action", r) for r in (1, 2)], "train collection")
    say("step 1 done: train trajectories")
    # 2. fit + freeze
    pol = Path(a.policy_file)
    if pol.exists():
        say(f"step 2 skipped: policy already frozen at {pol}")
    else:
        r = subprocess.run([sys.executable, "-B", "-m", "c4.fit_policy", "--traces",
                            str(Path(a.out_root) / "train" / "random_action"), "--out", str(pol)], cwd=ROOT)
        if r.returncode:
            say("STOPPED at fit_policy"); sys.exit(1)
        say(f"step 2 done: policy fitted on train only and frozen -> {pol}")
    # 3/4. validation, then test once
    for split, name in (("validation", "results_validation.txt"), ("test", "results_test.txt")):
        if split == "test" and a.skip_test:
            say("test skipped by flag"); break
        for arm in ARMS:
            run_parallel([runner(split, arm, r) for r in (1, 2)], f"{split}/{arm}")
            say(f"{split}/{arm}: both repeats done")
        analyze(split, name)
    say("=== ALL DONE ===")


if __name__ == "__main__":
    main()
