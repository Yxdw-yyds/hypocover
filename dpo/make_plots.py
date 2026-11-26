"""Generate publication-quality SVG figures and CSV curve tables for the report & 10-min video:
  1. RR@slot cumulative recovery curves on k=3 (27 slots) and k=2 (9 slots)
  2. Complexity-tier recovery breakdown (LOW / MID / HIGH) across arms
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from dpo.analyze import load_gt_lookup_for_posthoc_metrics
from dpo.common import ROOT, parse_and_normalize


ARM_LABELS = {
    "deepseek_512": "DeepSeek-V4-Flash (512 tok)",
    "base": "Qwen2.5-1.5B Base",
    "dpo_imbalanced": "Naive Imbalanced DPO (Collapsed)",
    "sft_validity": "Validity-SFT (Memoryless)",
    "sft_history": "History-SFT (MLE)",
    "coverage_dpo": "Symmetric Coverage-DPO (T=0.7, p=0.8)",
    "coverage_dpo_t10": "Symmetric Coverage-DPO + Ancestral (Ours)",
}

ARM_COLORS = {
    "deepseek_512": "#6b7280",
    "base": "#ef4444",
    "dpo_imbalanced": "#a855f7",
    "sft_validity": "#f59e0b",
    "sft_history": "#3b82f6",
    "coverage_dpo": "#14b8a6",
    "coverage_dpo_t10": "#059669",
}


def compute_rr_curve_for_traces(trace_files: List[Path], gt_lookup: Dict[str, dict], target_k: int) -> np.ndarray:
    curves = []
    budget = 9 if target_k == 2 else 27
    for tf in trace_files:
        rows = [json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        qid = rows[0].get("question") or tf.stem
        if qid not in gt_lookup or gt_lookup[qid]["k"] != target_k:
            continue
        rows.sort(key=lambda r: r.get("slot", 0))
        info = gt_lookup[qid]
        gt_hashes = set(info["gt_tiers"].keys())
        found = set()
        c = []
        for idx in range(budget):
            if idx < len(rows):
                text = rows[idx].get("scorer_text", rows[idx].get("response", ""))
                st = parse_and_normalize(info["bench"], text)
                if st is not None and st.get_hash() in gt_hashes:
                    found.add(st.get_hash())
            c.append(100.0 * len(found) / len(gt_hashes))
        curves.append(c)
    return np.mean(curves, axis=0) if curves else np.zeros(budget)


def write_svg_rr_curve(curves: Dict[str, np.ndarray], out_svg: Path, title: str):
    w, h = 760, 460
    pad_l, pad_r, pad_t, pad_b = 70, 40, 55, 65
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    max_x = max(len(v) for v in curves.values())

    def sx(step_1based: float) -> float:
        return pad_l + (step_1based - 1) / max(1, max_x - 1) * pw

    def sy(val_pct: float) -> float:
        return pad_t + (1.0 - val_pct / 100.0) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{w/2}" y="30" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="bold" fill="#111827">{title}</text>',
    ]

    # Grid lines
    for y_tick in (0, 20, 40, 60, 80, 100):
        yy = sy(y_tick)
        lines.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{pad_l-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{y_tick}%</text>')

    # Theoretical memoryless coupon collector dashed curve: 100 * (1 - ((N-1)/N)^t)
    cc_pts = " ".join(f"{sx(t):.1f},{sy(100.0 * (1.0 - ((max_x - 1) / max_x) ** t)):.1f}" for t in range(1, max_x + 1))
    lines.append(f'<polyline fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="5,5" points="{cc_pts}"/>')

    step_stride = 3 if max_x > 12 else 1
    x_ticks = sorted(set([1] + list(range(step_stride, max_x + 1, step_stride)) + [max_x]))
    for xt in x_ticks:
        xx = sx(xt)
        lines.append(f'<line x1="{xx:.1f}" y1="{pad_t+ph}" x2="{xx:.1f}" y2="{pad_t+ph+5}" stroke="#374151"/>')
        lines.append(f'<text x="{xx:.1f}" y="{pad_t+ph+22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{xt}</text>')

    # Axes
    lines.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="#374151" stroke-width="1.5"/>')
    lines.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{w-pad_r}" y2="{pad_t+ph}" stroke="#374151" stroke-width="1.5"/>')
    lines.append(f'<text x="{pad_l + pw/2}" y="{h-15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#1f2937">Query Slot t (Budget N = {max_x})</text>')
    lines.append(f'<text x="20" y="{pad_t + ph/2}" transform="rotate(-90 20,{pad_t + ph/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#1f2937">Cumulative Recovery Rate (%)</text>')

    # Curves
    leg_x, leg_y = pad_l + 16, pad_t + 14
    for arm, arr in curves.items():
        color = ARM_COLORS.get(arm, "#111827")
        label = ARM_LABELS.get(arm, arm) + f" ({arr[-1]:.1f}%)"
        pts = " ".join(f"{sx(i+1):.1f},{sy(val):.1f}" for i, val in enumerate(arr))
        width = "3.2" if arm == "coverage_dpo" else "2.2"
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}"/>')
        lines.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x+22}" y2="{leg_y}" stroke="{color}" stroke-width="{width}"/>')
        lines.append(f'<text x="{leg_x+28}" y="{leg_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#1f2937">{label}</text>')
        leg_y += 20

    lines.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x+22}" y2="{leg_y}" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="5,5"/>')
    lines.append(f'<text x="{leg_x+28}" y="{leg_y+4}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Coupon Collector Ceiling ({100*(1-((max_x-1)/max_x)**max_x):.1f}%)</text>')
    lines.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=ROOT / "dpo" / "runs")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--arms", nargs="+", default=["base", "sft_validity", "sft_history", "coverage_dpo"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dpo" / "figures")
    args = ap.parse_args()

    gt_lookup = load_gt_lookup_for_posthoc_metrics()
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    split_qids = {q for g in manifest["groups"].values() for q in g[args.split]}

    for target_k in (3, 2):
        curves = {}
        # DeepSeek baseline traces from results/*.jsonl
        ds_files = [ROOT / "results" / f"{q}.jsonl" for q in sorted(split_qids) if (ROOT / "results" / f"{q}.jsonl").exists()]
        if ds_files:
            # Map raw DeepSeek results/*.jsonl where slot is implicit order
            ds_arr = compute_rr_curve_for_traces(ds_files, gt_lookup, target_k)
            if np.any(ds_arr > 0):
                curves["deepseek_512"] = ds_arr

        for arm in args.arms:
            arm_dir = args.runs_root / args.split / arm
            t_files = sorted(arm_dir.glob("rep*/traces/*.jsonl"))
            if t_files:
                curves[arm] = compute_rr_curve_for_traces(t_files, gt_lookup, target_k)

        if curves:
            svg_path = args.out_dir / f"rr_curve_{args.split}_k{target_k}.svg"
            write_svg_rr_curve(
                curves,
                svg_path,
                title=f"HypoSpace 3D ({args.split.capitalize()}, k={target_k}, |H_O|={3**target_k}): Cumulative Recovery vs. Query Budget",
            )
            csv_path = args.out_dir / f"rr_curve_{args.split}_k{target_k}.csv"
            budget = 9 if target_k == 2 else 27
            with csv_path.open("w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                arm_keys = list(curves.keys())
                writer.writerow(["slot"] + arm_keys)
                for s in range(budget):
                    writer.writerow([s + 1] + [round(float(curves[a][s]), 3) for a in arm_keys])
            print(f"[make_plots] wrote {svg_path} and {csv_path}")


if __name__ == "__main__":
    main()
