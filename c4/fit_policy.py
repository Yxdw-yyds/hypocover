"""Fit the strategy-layer policy from collected (random-behaviour) trajectories.

  python -m c4.fit_policy --traces c4/layered/train/random_action --out c4/policy.json

Model: one L2-regularised logistic regression per action predicting P(this step yields a new
valid structure | history-only features, action).  Policy = greedy over those predictions.
The L2 strength is chosen by question-grouped cross-validation (log loss).  The CV also
reports a one-step inverse-propensity estimate of the greedy policy's per-step reward against
the behaviour policy and against "always none" -- myopic (ignores how choices change later
history), so it is a screening number, not an evaluation.  Only train data is read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c4 import layered_actions as LA  # noqa: E402

LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def load_rows(trace_dirs):
    X, A, R, G = [], [], [], []
    for d in trace_dirs:
        for f in sorted(Path(d).glob("traces/*.jsonl")) if (Path(d) / "traces").exists() else sorted(Path(d).glob("*.jsonl")):
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
            recs.sort(key=lambda r: r["slot"])
            hist = []
            for r in recs:
                if "outcome" not in r:  # transport error: no model output, skipped everywhere
                    continue
                x = LA.features(hist, r["slot"] - 1, r["budget"], r["k"])
                X.append(x); A.append(r["action"]); R.append(1.0 if r["outcome"] == "new" else 0.0)
                G.append(f"{Path(d).parent.parent.name}/{r['question']}")
                hist.append({"action": r["action"], "outcome": r["outcome"], "tier": r["tier"]})
    return np.array(X), np.array(A), np.array(R), np.array(G)


def fit_logistic(X, y, lam):
    d = X.shape[1]
    if len(y) < 20 or y.min() == y.max():
        w = np.zeros(d); p = (y.sum() + 1) / (len(y) + 2)
        w[0] = np.log(p / (1 - p)); return w

    def f(w):
        z = X @ w; loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[1:] ** 2)
        g = X.T @ (1 / (1 + np.exp(-z)) - y); g[1:] += lam * w[1:]
        return loss, g
    return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B").x


def fit_all(X, A, R, lam):
    return {a: fit_logistic(X[A == a], R[A == a], lam) for a in LA.ACTIONS}


def predict(W, X):
    return np.stack([1 / (1 + np.exp(-(X @ W[a]))) for a in LA.ACTIONS], axis=1)


def cross_validate(X, A, R, G, k_folds=5, seed=0):
    qs = np.unique(G); rng = np.random.default_rng(seed); rng.shuffle(qs)
    fold_of = {q: i % k_folds for i, q in enumerate(qs)}
    folds = np.array([fold_of[g] for g in G])
    out = {}
    base = R.mean(); n_act = len(LA.ACTIONS)
    none_idx = LA.ACTIONS.index("none")
    for lam in LAMBDAS:
        ll, ips, ips_none = [], [], []
        for f in range(k_folds):
            tr, te = folds != f, folds == f
            W = fit_all(X[tr], A[tr], R[tr], lam)
            P = predict(W, X[te]); a_te = np.array([LA.ACTIONS.index(a) for a in A[te]])
            p_taken = np.clip(P[np.arange(len(a_te)), a_te], 1e-6, 1 - 1e-6)
            ll.append(-np.mean(R[te] * np.log(p_taken) + (1 - R[te]) * np.log(1 - p_taken)))
            greedy = P.argmax(axis=1)
            ips.append(np.mean((a_te == greedy) * R[te] * n_act))
            ips_none.append(np.mean((a_te == none_idx) * R[te] * n_act))
        out[lam] = {"logloss": float(np.mean(ll)), "ips_greedy": float(np.mean(ips)), "ips_always_none": float(np.mean(ips_none))}
    return out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="dirs like c4/layered/train/random_action (all rep*/ inside are read)")
    ap.add_argument("--out", default=str(ROOT / "c4" / "policy.json"))
    a = ap.parse_args()
    dirs = []
    for t in a.traces:
        p = Path(t); dirs += sorted(p.glob("rep*")) or [p]
    X, A, R, G = load_rows(dirs)
    if len(R) < 100:
        sys.exit(f"only {len(R)} usable steps; collect more training trajectories first")
    print(f"steps={len(R)} questions={len(set(G))} behaviour reward={R.mean():.3f}")
    for act in LA.ACTIONS:
        m = A == act; print(f"  {act:5s} n={m.sum():5d}  mean reward={R[m].mean():.3f}")
    cv, base = cross_validate(X, A, R, G)
    print("\nCV by lambda (question-grouped):")
    for lam, v in cv.items():
        print(f"  lam={lam:<6} logloss={v['logloss']:.4f}  IPS(greedy)={v['ips_greedy']:.3f}  IPS(always none)={v['ips_always_none']:.3f}  (behaviour {base:.3f})")
    best = min(cv, key=lambda l: cv[l]["logloss"])
    W = fit_all(X, A, R, best)
    model = {"actions": list(LA.ACTIONS), "features": LA.FEATURE_NAMES, "lambda": best,
             "weights": {k: v.tolist() for k, v in W.items()}, "n_steps": int(len(R)),
             "cv": {str(k): v for k, v in cv.items()}, "behaviour_reward": float(base)}
    Path(a.out).write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"\nchosen lambda={best}; policy -> {a.out}")


if __name__ == "__main__":
    main()
