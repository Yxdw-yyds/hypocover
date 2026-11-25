"""Score every arm with the OFFICIAL scorer and apply the frozen success criteria
(c4/STRATEGY_LAYER_PLAN.md section 7).

  python -m c4.analyze_layered --split validation --repeats 1 2
Official RR/VR/NR come straight from score_predictions.py (no conversion).  Trace-based
extras (compliance, invalid/duplicate rate, cost) are descriptive and labelled as such.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("free_official", "paper_stratified", "tier_static", "random_level", "tier_cycle", "learned")
OTHERS = ("paper_stratified", "tier_static", "random_level", "tier_cycle")


def score(split, run_dir: Path):
    out = run_dir / "score.json"
    if not out.exists():
        pred = run_dir / "predictions.jsonl"
        r = subprocess.run([sys.executable, "-B", str(ROOT / "score_predictions.py"), split, str(pred), "--output", str(out)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"scoring {pred} failed: {r.stderr[-400:]}")
    return {c["observation_set_id"]: c for c in json.loads(out.read_text(encoding="utf-8"))["cases"]}


def trace_stats(run_dir: Path):
    n = bad = dup = unp = comp_n = comp_ok = terr = 0
    cost = 0.0; trunc = 0; prov = collections.Counter()
    for f in (run_dir / "traces").glob("*.jsonl"):
        for l in f.read_text(encoding="utf-8").splitlines():
            t = json.loads(l)
            if "transport_error" in t:
                terr += 1; continue
            n += 1; cost += t.get("cost") or 0.0
            trunc += t.get("finish_reason") == "length"; prov[t.get("provider")] += 1
            bad += t["outcome"] == "invalid"; dup += t["outcome"] == "dup"; unp += t["outcome"] == "unparsed"
            if t["action"] != "none" and t["complies"] is not None:
                comp_n += 1; comp_ok += bool(t["complies"])
    return dict(n=n, invalid=bad / max(n, 1), dup=dup / max(n, 1), unparsed=unp / max(n, 1),
                compliance=(comp_ok / comp_n) if comp_n else None, transport_err=terr, cost=cost,
                truncated=trunc / max(n, 1), providers=dict(prov))


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--root", default=str(ROOT / "c4" / "layered"))
    a = ap.parse_args()
    groups = json.loads((ROOT / "data" / "manifest.json").read_text())["groups"]
    kq = {q: int(g) for g, v in groups.items() for q in v[a.split]}
    qids = sorted(kq)

    per_arm, stats = {}, {}
    for arm in ARMS:
        runs = [Path(a.root) / a.split / arm / f"rep{r}" for r in a.repeats]
        runs = [r for r in runs if (r / "predictions.jsonl").exists()]
        if not runs:
            print(f"[skip] {arm}: no runs found"); continue
        sc = [score(a.split, r) for r in runs]
        per_arm[arm] = {m: np.array([np.mean([s[q][m] for s in sc]) for q in qids])
                        for m in ("recovery_rate", "valid_rate", "novelty_rate")}
        ts = [trace_stats(r) for r in runs]
        stats[arm] = {k: (None if ts[0][k] is None else float(np.mean([t[k] for t in ts])))
                      for k in ts[0] if k != "providers"}
        pc = collections.Counter()
        for t in ts:
            pc.update(t["providers"])
        stats[arm]["providers"] = dict(pc)
        stats[arm]["repeats"] = len(runs)

    k2 = np.array([kq[q] == 2 for q in qids]); k3 = ~k2
    print(f"\nsplit={a.split}  questions={len(qids)} (k=2: {k2.sum()}, k=3: {k3.sum()})  official scorer, mean over repeats\n")
    print(f"{'arm':18s} {'RR':>7} {'RR k2':>7} {'RR k3':>7} {'VR':>7} {'NR':>7} | {'invalid':>7} {'dup':>6} {'comply':>7} {'cost$':>7} reps")
    for arm, d in per_arm.items():
        rr = d["recovery_rate"]; s = stats[arm]
        comp = "-" if s["compliance"] is None else f"{s['compliance']:.0%}"
        print(f"{arm:18s} {rr.mean():7.2%} {rr[k2].mean():7.2%} {rr[k3].mean():7.2%} {d['valid_rate'].mean():7.2%} "
              f"{d['novelty_rate'].mean():7.2%} | {s['invalid']:7.1%} {s['dup']:6.1%} {comp:>7} {s['cost']:7.3f} {s['repeats']:.0f}")
    print("\n(paper_stratified and tier_static use ground-truth counts; invalid/dup/comply/cost are descriptive trace statistics)")
    print("\nprovider mix per arm (an arm served by a different provider is confounded) and length-truncation rate:")
    for arm in per_arm:
        tot = sum(stats[arm]["providers"].values()) or 1
        mix = ", ".join(f"{p}:{n / tot:.0%}" for p, n in sorted(stats[arm]["providers"].items(), key=lambda t: -t[1]))
        print(f"  {arm:18s} truncated={stats[arm]['truncated']:.1%}  providers: {mix}")

    if "learned" not in per_arm or "free_official" not in per_arm:
        print("\nVerdict needs both `learned` and `free_official` runs."); return
    L, F = per_arm["learned"]["recovery_rate"], per_arm["free_official"]["recovery_rate"]
    d1 = L - F; lo, hi = boot_ci(d1)
    s1 = d1.mean() >= 0.05 and lo > 0
    best_other = max(((o, per_arm[o]["recovery_rate"].mean()) for o in OTHERS if o in per_arm), key=lambda t: t[1], default=None)
    s2 = best_other is not None and L.mean() >= best_other[1]
    cmp_ = stats["learned"]["compliance"]
    s3 = cmp_ is not None and cmp_ >= 0.80 and stats["learned"]["invalid"] <= stats["free_official"]["invalid"]
    print(f"\nS1 learned - free_official = {d1.mean():+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]  (need >= +5.00% and CI low > 0)  -> {'PASS' if s1 else 'FAIL'}")
    if best_other:
        print(f"S2 learned {L.mean():.2%} vs best non-learned `{best_other[0]}` {best_other[1]:.2%}  -> {'PASS' if s2 else 'FAIL'}")
        for o in OTHERS:
            if o in per_arm:
                dd = L - per_arm[o]["recovery_rate"]; l2, h2 = boot_ci(dd)
                print(f"     learned - {o}: {dd.mean():+.2%}  95% CI [{l2:+.2%}, {h2:+.2%}]")
    print(f"S3 compliance={'n/a' if cmp_ is None else f'{cmp_:.0%}'} (>=80%), invalid learned {stats['learned']['invalid']:.1%} vs free {stats['free_official']['invalid']:.1%}  -> {'PASS' if s3 else 'FAIL'}")
    if s1 and s2 and s3:
        print("\nVERDICT: learning adds value over free and over the best non-learned arm.")
    elif s1 and s3:
        print("\nVERDICT: tiered hints help, but learning adds nothing over the best fixed schedule (report honestly).")
    else:
        print("\nVERDICT: negative / inconclusive under the frozen criteria (report honestly; CIs are wide with few questions).")


if __name__ == "__main__":
    main()
