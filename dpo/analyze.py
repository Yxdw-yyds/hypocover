"""Post-hoc statistical analysis, complexity-tier recovery, RR@k curves, and paired bootstrap CIs.

Reads score.json and trace files produced by dpo/evaluate.py alongside the frozen DeepSeek baseline
(summary.json / results/*.json) and outputs publication-ready Markdown tables + analysis JSON.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dpo.common import ROOT, count_blocks, parse_and_normalize, parse_compact_gt, tier_of
from run_3d_benchmark import Benchmark3D  # noqa: E402


def load_gt_lookup_for_posthoc_metrics() -> Dict[str, dict]:
    """Load ground-truth sets strictly for post-hoc evaluation breakdown (tier RR & RR curves)."""
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for gid, g in manifest["groups"].items():
        bench = Benchmark3D(str(ROOT / "data" / g["file"]))
        for item in bench.all_observation_sets:
            qid = item["observation_id"]
            obs = item["observation"]
            k = str(obs).count("1")
            gt_tiers = {}
            for gt in item["ground_truth_structures"]:
                st = parse_compact_gt(gt)
                gt_tiers[st.get_hash()] = tier_of(count_blocks(st), k)
            out[qid] = {"k": k, "budget": int(g["budget"]), "bench": bench, "gt_tiers": gt_tiers}
    return out


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def analyze_arm_split(runs_root: Path, split: str, arm: str, gt_lookup: Dict[str, dict]) -> Optional[dict]:
    arm_dir = runs_root / split / arm
    if not arm_dir.exists():
        return None
    rep_dirs = sorted(p for p in arm_dir.glob("rep*") if (p / "score.json").exists())
    if not rep_dirs:
        return None

    per_q_metrics = defaultdict(lambda: defaultdict(list))
    rep_rrs = []

    for rd in rep_dirs:
        sc = json.loads((rd / "score.json").read_text(encoding="utf-8"))
        rep_rrs.append(sc["summary"]["recovery_rate"] * 100.0)
        for c in sc["cases"]:
            qid = c["observation_set_id"]
            for m in ("parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate"):
                per_q_metrics[qid][m].append(c[m] * 100.0)

            # Detailed trace diagnostics (truncation, tokens, tier RR, late-slot RR)
            tf = rd / "traces" / f"{qid}.jsonl"
            if tf.exists():
                rows = [json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
                rows.sort(key=lambda r: r["slot"])
                info = gt_lookup[qid]
                gt_tiers = info["gt_tiers"]
                tier_totals = defaultdict(int)
                for t in gt_tiers.values():
                    tier_totals[t] += 1

                found_gt_by_tier = defaultdict(set)
                seen_all = set()
                late_new_valid = 0
                late_total = 0
                trunc_cnt = 0
                comp_toks = []

                for r in rows:
                    if r.get("finish_reason") == "length":
                        trunc_cnt += 1
                    comp_toks.append(r.get("completion_tokens", 0))
                    st = parse_and_normalize(info["bench"], r.get("scorer_text", ""))
                    h = st.get_hash() if st is not None else None
                    is_new_valid = (h is not None) and (h in gt_tiers) and (h not in seen_all)
                    if r["slot"] > len(rows) // 3:
                        late_total += 1
                        if is_new_valid:
                            late_new_valid += 1
                    if h is not None:
                        seen_all.add(h)
                        if h in gt_tiers:
                            found_gt_by_tier[gt_tiers[h]].add(h)

                per_q_metrics[qid]["trunc_rate"].append(100.0 * trunc_cnt / len(rows))
                per_q_metrics[qid]["avg_comp_tokens"].append(float(np.mean(comp_toks)))
                per_q_metrics[qid]["late_new_valid_rate"].append(
                    100.0 * late_new_valid / max(late_total, 1)
                )
                for t_name in ("low", "mid", "high"):
                    if tier_totals[t_name] > 0:
                        per_q_metrics[qid][f"rr_{t_name}"].append(
                            100.0 * len(found_gt_by_tier[t_name]) / tier_totals[t_name]
                        )

    qids = sorted(per_q_metrics.keys())
    q_avg = {qid: {m: float(np.mean(vals)) for m, vals in per_q_metrics[qid].items()} for qid in qids}
    k2_qids = [q for q in qids if gt_lookup[q]["k"] == 2]
    k3_qids = [q for q in qids if gt_lookup[q]["k"] == 3]

    def avg_over(qs, key):
        vals = [q_avg[q][key] for q in qs if key in q_avg[q]]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "arm": arm,
        "n_repeats": len(rep_dirs),
        "rep_rr_mean": float(np.mean(rep_rrs)),
        "rep_rr_std": float(np.std(rep_rrs, ddof=1)) if len(rep_rrs) > 1 else 0.0,
        "overall": {
            m: avg_over(qids, m)
            for m in (
                "parse_success_rate",
                "valid_rate",
                "novelty_rate",
                "recovery_rate",
                "trunc_rate",
                "avg_comp_tokens",
                "late_new_valid_rate",
                "rr_low",
                "rr_mid",
                "rr_high",
            )
        },
        "rr_k2": avg_over(k2_qids, "recovery_rate"),
        "rr_k3": avg_over(k3_qids, "recovery_rate"),
        "per_question_rr": {q: q_avg[q]["recovery_rate"] for q in qids},
    }


def load_deepseek_baseline_per_q(split: str) -> Dict[str, float]:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    qids = [q for g in manifest["groups"].values() for q in g[split]]
    out = {}
    for q in qids:
        p = ROOT / "results" / f"{q}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            out[q] = float(d["recovery_rate"]) * 100.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=ROOT / "dpo" / "runs")
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--arms", nargs="+", default=["base", "sft_validity", "sft_history", "coverage_dpo"])
    ap.add_argument("--out", type=Path, default=ROOT / "dpo" / "results_summary.json")
    args = ap.parse_args()

    gt_lookup = load_gt_lookup_for_posthoc_metrics()
    full_report = {}

    for split in args.splits:
        print(f"\n==================== {split.upper()} SPLIT ====================")
        ds_base = load_deepseek_baseline_per_q(split)
        arm_res = {}
        for arm in args.arms:
            r = analyze_arm_split(args.runs_root, split, arm, gt_lookup)
            if r is not None:
                arm_res[arm] = r

        header = (
            f"{'Arm':<16} | {'Reps':>4} | {'k=2 RR':>7} | {'k=3 RR':>7} | {'Overall RR':>13} | "
            f"{'VR':>6} | {'NR':>6} | {'Trunc':>6} | {'Toks':>5} | {'High-RR':>7}"
        )
        print(header)
        print("-" * len(header))
        if ds_base:
            k2 = np.mean([v for q, v in ds_base.items() if "_b2_" in q])
            k3 = np.mean([v for q, v in ds_base.items() if "_b3_" in q])
            ov = np.mean(list(ds_base.values()))
            print(f"{'deepseek_512_ref':<16} | {'1':>4} | {k2:6.2f}% | {k3:6.2f}% | {ov:12.2f}% | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>5} | {'-':>7}")

        for arm, r in arm_res.items():
            ov = r["overall"]
            rr_str = f"{ov['recovery_rate']:5.2f}±{r['rep_rr_std']:4.2f}%"
            print(
                f"{arm:<16} | {r['n_repeats']:>4} | {r['rr_k2']:6.2f}% | {r['rr_k3']:6.2f}% | {rr_str:>13} | "
                f"{ov['valid_rate']:5.1f}% | {ov['novelty_rate']:5.1f}% | {ov['trunc_rate']:5.1f}% | "
                f"{ov['avg_comp_tokens']:5.1f} | {ov['rr_high']:6.2f}%"
            )

        comparisons = {}
        if "coverage_dpo" in arm_res:
            qids = sorted(arm_res["coverage_dpo"]["per_question_rr"].keys())
            dpo_vec = np.array([arm_res["coverage_dpo"]["per_question_rr"][q] for q in qids])
            for other in ("base", "sft_validity", "sft_history"):
                if other in arm_res:
                    oth_vec = np.array([arm_res[other]["per_question_rr"][q] for q in qids])
                    mean_d, lo, hi = paired_bootstrap_ci(dpo_vec - oth_vec)
                    comparisons[f"coverage_dpo_vs_{other}"] = {"diff": mean_d, "ci95": [lo, hi]}
                    print(f"  Paired diff (coverage_dpo - {other:<12}): {mean_d:+6.2f} pp  95% CI [{lo:+6.2f}, {hi:+6.2f}]")
            if ds_base and all(q in ds_base for q in qids):
                ds_vec = np.array([ds_base[q] for q in qids])
                mean_d, lo, hi = paired_bootstrap_ci(dpo_vec - ds_vec)
                comparisons["coverage_dpo_vs_deepseek_512"] = {"diff": mean_d, "ci95": [lo, hi]}
                print(f"  Paired diff (coverage_dpo - deepseek_512): {mean_d:+6.2f} pp  95% CI [{lo:+6.2f}, {hi:+6.2f}]")

        full_report[split] = {"arms": arm_res, "paired_comparisons": comparisons}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved full analysis to {args.out}")


if __name__ == "__main__":
    main()
