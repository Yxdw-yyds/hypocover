"""Construct strictly train-only datasets for:
  1. sft_validity.jsonl  : matched prompt prefixes (O, H_<t), target sampled uniformly from ALL H_O
                           (learns 100% physical validity & format, bounded by Coupon Collector ~64.6%)
  2. sft_history.jsonl   : exact same prompt prefixes (O, H_<t), target sampled from H_O \\ H_<t
                           (standard MLE behavioral cloning over uncovered hypotheses)
  3. dpo_coverage.jsonl  : Coverage-Weighted DPO preference pairs combining on-policy sft_history
                           rollout mistakes (duplicates + invalids) with hard 1-voxel Hamming
                           collision negatives and inverse-tier-frequency complexity weighting.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dpo.common import (
    ROOT,
    QuestionSpec,
    Structure3D,
    canonical_response,
    count_blocks,
    load_train_questions_with_gt,
    parse_and_normalize,
    tier_of,
    voxel_hamming,
    voxel_tensor,
)


def make_synthetic_invalid(q: QuestionSpec, rng: random.Random) -> Structure3D:
    """Create a realistic invalid 3D structure (floating block, missing column, or extra column)
    using only the observation top view, to make history conditioning robust to occasional errors
    and provide physical-validity negatives."""
    obs_grid = q.bench._parse_observation(q.obs)
    ones = [(r, c) for r in range(3) for c in range(3) if obs_grid[r, c] == 1]
    zeros = [(r, c) for r in range(3) for c in range(3) if obs_grid[r, c] == 0]
    mode = rng.choice(["floating", "missing_col", "extra_col", "empty_bottom"])
    if mode == "floating" and ones:
        # Layer 1 misses a column or is valid, Layer 2/3 puts a block above 0
        l1 = obs_grid.copy().tolist()
        l2 = [[0] * 3 for _ in range(3)]
        l3 = [[0] * 3 for _ in range(3)]
        r, c = rng.choice(ones)
        l3[r][c] = 1  # l2[r][c] == 0 -> floating violation!
        return Structure3D([l1, l2, l3]).normalize()
    if mode == "missing_col" and len(ones) >= 2:
        l1 = obs_grid.copy()
        r, c = rng.choice(ones)
        l1[r, c] = 0  # fails top-view consistency
        return Structure3D([l1.tolist()]).normalize()
    if mode == "extra_col" and zeros:
        l1 = obs_grid.copy()
        r, c = rng.choice(zeros)
        l1[r, c] = 1  # extra block outside top-view
        return Structure3D([l1.tolist()]).normalize()
    # empty bottom layer + block in layer 2
    l1 = [[0] * 3 for _ in range(3)]
    l2 = obs_grid.copy().tolist()
    return Structure3D([l1, l2]).normalize()


def order_simplicity_biased(gts: List[Structure3D], rng: random.Random, temp: float = 0.8) -> List[Structure3D]:
    """Sample a without-replacement trajectory biased toward fewer blocks first,
    mimicking natural LLM simplicity bias so late prefixes resemble real test-time states."""
    pool = list(gts)
    ordered = []
    while pool:
        weights = [np.exp(-temp * count_blocks(s)) for s in pool]
        tot = sum(weights)
        probs = [w / tot for w in weights]
        idx = rng.choices(range(len(pool)), weights=probs, k=1)[0]
        ordered.append(pool.pop(idx))
    return ordered


def build_sft_datasets(out_dir: Path, n_trajectories_k2: int = 4, n_trajectories_k3: int = 3, seed: int = 42) -> Tuple[Path, Path]:
    """Build matched sft_validity.jsonl and sft_history.jsonl strictly from the 84 train questions."""
    rng = random.Random(seed)
    train_qs = load_train_questions_with_gt("train")
    out_dir.mkdir(parents=True, exist_ok=True)
    val_only_path = out_dir / "sft_validity.jsonl"
    hist_path = out_dir / "sft_history.jsonl"

    validity_rows = []
    history_rows = []

    for q in train_qs:
        gts = list(q.gt_structures or [])
        gt_by_hash = {s.get_hash(): s for s in gts}
        n_traj = n_trajectories_k2 if q.k == 2 else n_trajectories_k3

        for traj_idx in range(n_traj):
            if traj_idx % 3 == 0:
                seq = list(gts)
                rng.shuffle(seq)
            elif traj_idx % 3 == 1:
                seq = order_simplicity_biased(gts, rng, temp=0.75)
            else:
                # Complexity-diverse / reverse-biased permutation so all history compositions are covered
                seq = order_simplicity_biased(gts, rng, temp=-0.5)

            for step in range(len(seq)):
                prior_valid = list(seq[:step])
                prior_for_prompt = list(prior_valid)
                # In 15% of non-empty prefixes, inject 1 synthetic invalid attempt to match no-validator rollouts
                if step >= 2 and rng.random() < 0.15:
                    inv = make_synthetic_invalid(q, rng)
                    if inv.get_hash() not in gt_by_hash:
                        insert_pos = rng.randint(0, len(prior_for_prompt))
                        prior_for_prompt.insert(insert_pos, inv)

                prompt = q.bench.create_prompt(q.obs, prior_structures=prior_for_prompt)
                seen_hashes = {s.get_hash() for s in prior_valid}
                uncovered = [s for s in gts if s.get_hash() not in seen_hashes]

                # Complexity-balanced sampling of uncovered positive target for sft_history:
                # inverse tier count among uncovered structures prevents simplicity collapse at step 0..k
                tier_counts = Counter(tier_of(count_blocks(s), q.k) for s in uncovered)
                u_weights = [1.0 / tier_counts[tier_of(count_blocks(s), q.k)] for s in uncovered]
                chosen_novel = rng.choices(uncovered, weights=u_weights, k=1)[0]

                # Uniform random valid structure (ignoring history!) for sft_validity control
                chosen_any_valid = rng.choice(gts)

                meta = {"qid": q.qid, "k": q.k, "step": step, "n_prior": len(prior_for_prompt)}
                validity_rows.append({
                    "prompt": prompt,
                    "completion": canonical_response(chosen_any_valid),
                    **meta,
                })
                history_rows.append({
                    "prompt": prompt,
                    "completion": canonical_response(chosen_novel),
                    **meta,
                })

    rng.shuffle(validity_rows)
    rng.shuffle(history_rows)

    with val_only_path.open("w", encoding="utf-8") as f:
        for r in validity_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with hist_path.open("w", encoding="utf-8") as f:
        for r in history_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build_sft_datasets] train_questions={len(train_qs)} "
          f"sft_validity={len(validity_rows)} sft_history={len(history_rows)}")
    return val_only_path, hist_path


def build_dpo_dataset_from_rollouts(
    rollout_traces_dir: Optional[Path],
    out_path: Path,
    seed: int = 123,
) -> Path:
    """Build Coverage-Weighted DPO pairs strictly from the 84 train questions.

    Combines:
      (A) On-policy mistakes mined from sft_history rollouts on train questions
          (exact duplicate collisions y^- in H_<t and on-policy invalid outputs y^- not in H_O).
      (B) Hard 1-voxel Hamming-neighbor collision pairs on mid/late prefixes (t >= 1):
          chosen y^+ in H_O \\ H_<t (weighted toward under-covered / high-complexity tiers)
          vs rejected y^- = argmin_{h in H_<t} Hamming(y^+, h) (the closest boundary duplicate!),
          plus physical-validity negatives at early/mid steps.
    """
    rng = random.Random(seed)
    train_qs = load_train_questions_with_gt("train")
    q_map: Dict[str, QuestionSpec] = {q.qid: q for q in train_qs}
    pairs: List[dict] = []

    # ---- Part A: Mine actual on-policy errors from sft_history train rollouts ----
    on_policy_dups = 0
    on_policy_invs = 0
    if rollout_traces_dir and rollout_traces_dir.exists():
        for tf in sorted(rollout_traces_dir.glob("rep*/traces/*.jsonl")):
            recs = [json.loads(line) for line in tf.read_text(encoding="utf-8").splitlines() if line.strip()]
            recs.sort(key=lambda r: r["slot"])
            qid = recs[0]["question"] if recs else None
            if qid not in q_map:
                # Strict guard: ignore anything not in the 84 train questions
                continue
            q = q_map[qid]
            gts = list(q.gt_structures or [])
            gt_by_hash = {s.get_hash(): s for s in gts}

            unique_prior: List[Structure3D] = []
            seen_hashes = set()
            found_valid_tiers = Counter()

            for r in recs:
                prompt = q.bench.create_prompt(q.obs, prior_structures=unique_prior)
                resp_text = r.get("scorer_text", "")
                st = parse_and_normalize(q.bench, resp_text)
                uncovered = [s for s in gts if s.get_hash() not in seen_hashes]

                if uncovered:
                    # Symmetric uniform sampling over uncovered H_O \ H_<t:
                    # preserves exact marginal parity E[p(y+)] == E[p(y-)] = Unif(H_O)
                    # so DPO CANNOT exploit any history-blind prior shortcut!
                    if st is None:
                        chosen = rng.choice(uncovered)
                        rejected_text = resp_text if resp_text.strip() else canonical_response(make_synthetic_invalid(q, rng))
                        pairs.append({
                            "qid": qid,
                            "prompt": prompt,
                            "chosen": canonical_response(chosen),
                            "rejected": rejected_text,
                            "weight": 1.0,
                            "pair_type": "onpolicy_unparsed",
                        })
                        on_policy_invs += 1
                    else:
                        h = st.get_hash()
                        is_valid = h in gt_by_hash
                        if h in seen_hashes:
                            # On-policy duplicate collision: pair against uniform uncovered structure
                            chosen = rng.choice(uncovered)
                            pairs.append({
                                "qid": qid,
                                "prompt": prompt,
                                "chosen": canonical_response(chosen),
                                "rejected": canonical_response(st),
                                "weight": 1.25,
                                "pair_type": "onpolicy_dup",
                            })
                            on_policy_dups += 1
                        elif not is_valid:
                            chosen = rng.choice(uncovered)
                            pairs.append({
                                "qid": qid,
                                "prompt": prompt,
                                "chosen": canonical_response(chosen),
                                "rejected": canonical_response(st),
                                "weight": 1.0,
                                "pair_type": "onpolicy_invalid",
                            })
                            on_policy_invs += 1

                # Advance state exactly like no-validator test rollouts
                if st is not None:
                    h = st.get_hash()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        unique_prior.append(st)
                        if h in gt_by_hash:
                            found_valid_tiers[tier_of(count_blocks(st), q.k)] += 1

    # ---- Part B: Symmetric Uniform-Permutation Hard-Collision Pairs ----
    # Because seq is a uniform random permutation of H_O, every h in H_O has EXACTLY identical
    # marginal probability of being chosen (in H_O \ H_<t) and being rejected (in H_<t).
    # Hence no unconditional prior shift can gain reward; only true attention to H_<t succeeds!
    boundary_dups = 0
    validity_negs = 0
    for q in train_qs:
        gts = list(q.gt_structures or [])
        n_traj = 3 if q.k == 2 else 2
        for traj_idx in range(n_traj):
            seq = list(gts)
            rng.shuffle(seq)
            for step in range(1, len(seq)):
                if q.k == 3 and step < 10 and rng.random() > 0.40:
                    continue
                prior_valid = list(seq[:step])
                seen_hashes = {s.get_hash() for s in prior_valid}
                uncovered = [s for s in gts if s.get_hash() not in seen_hashes]
                if not uncovered:
                    continue

                prompt = q.bench.create_prompt(q.obs, prior_structures=prior_valid)
                chosen = rng.choice(uncovered)
                # Closest 3D Hamming neighbor in H_<t (symmetric because Hamming is symmetric!)
                hard_dup = min(prior_valid, key=lambda p: (voxel_hamming(chosen, p), rng.random()))
                step_frac = step / len(seq)
                pairs.append({
                    "qid": q.qid,
                    "prompt": prompt,
                    "chosen": canonical_response(chosen),
                    "rejected": canonical_response(hard_dup),
                    "weight": round(0.85 + 0.5 * step_frac, 4),
                    "pair_type": "symmetric_hamming_dup",
                })
                boundary_dups += 1

                if rng.random() < 0.10:
                    inv = make_synthetic_invalid(q, rng)
                    pairs.append({
                        "qid": q.qid,
                        "prompt": prompt,
                        "chosen": canonical_response(chosen),
                        "rejected": canonical_response(inv),
                        "weight": 0.8,
                        "pair_type": "synthetic_invalid",
                    })
                    validity_negs += 1

    rng.shuffle(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(
        f"[build_dpo_dataset] total_pairs={len(pairs)} "
        f"(onpolicy_dup={on_policy_dups}, onpolicy_inv={on_policy_invs}, "
        f"hard_hamming_dup={boundary_dups}, synthetic_invalid={validity_negs}) -> {out_path}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo"], required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "dpo" / "data_gen")
    ap.add_argument("--rollout-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.mode == "sft":
        build_sft_datasets(args.out_dir, seed=args.seed)
    else:
        build_dpo_dataset_from_rollouts(
            args.rollout_dir, args.out_dir / "dpo_coverage.jsonl", seed=args.seed
        )


if __name__ == "__main__":
    main()
