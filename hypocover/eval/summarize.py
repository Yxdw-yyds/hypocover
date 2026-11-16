"""Collect every results/*/summary.json into one table.

  python -m hypocover.eval.summarize            # markdown to stdout
  python -m hypocover.eval.summarize --csv results/table.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(root: str = "results") -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob("*/summary.json")):
        s = json.loads(p.read_text())
        rows.append({
            "task": s["task"], "data": Path(s["data"]).stem, "method": s["method"],
            "args": json.dumps(s.get("method_args", {})) if s.get("method_args") else "",
            "model": s["model"], "n": s["n_instances"], "|H_O|": round(s.get("mean_n_gt", 0), 1),
            "VR": s.get("vr"), "NR": s.get("nr"), "RR": s.get("rr"),
            "RR_simple": s.get("rr_simple"), "RR_complex": s.get("rr_complex"),
            "calls": s["usage"]["calls"], "tokens": s["usage"]["prompt_tokens"] + s["usage"]["completion_tokens"],
            "cost$": round(s["usage"]["cost"], 3),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    df = load(a.root)
    if df.empty:
        print("no results yet")
        return
    for c in ("VR", "NR", "RR", "RR_simple", "RR_complex"):
        df[c] = (df[c] * 100).round(1)
    df = df.sort_values(["task", "data", "model", "method", "args"])
    if a.csv:
        df.to_csv(a.csv, index=False)
    print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))


if __name__ == "__main__":
    main()
