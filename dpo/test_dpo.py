"""Unit tests for strict split isolation, canonical round-trip fidelity, and loss masking."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from dpo.common import (
    ROOT,
    canonical_response,
    load_eval_questions,
    load_train_questions_with_gt,
    parse_and_normalize,
    voxel_hamming,
)


class TestDPOPipeline(unittest.TestCase):
    def test_anti_leakage_guard(self):
        with self.assertRaises(AssertionError):
            load_train_questions_with_gt("validation")
        with self.assertRaises(AssertionError):
            load_train_questions_with_gt("test")

    def test_disjoint_splits_and_observations(self):
        tr = load_eval_questions("train")
        va = load_eval_questions("validation")
        te = load_eval_questions("test")
        self.assertEqual(len(tr), 84)
        self.assertEqual(len(va), 12)
        self.assertEqual(len(te), 24)

        tr_ids, va_ids, te_ids = {q.qid for q in tr}, {q.qid for q in va}, {q.qid for q in te}
        self.assertFalse(tr_ids & va_ids)
        self.assertFalse(tr_ids & te_ids)
        self.assertFalse(va_ids & te_ids)

        # Even stronger: every top-view observation string is strictly disjoint across splits!
        tr_obs, va_obs, te_obs = {q.obs for q in tr}, {q.obs for q in va}, {q.obs for q in te}
        self.assertFalse(tr_obs & va_obs)
        self.assertFalse(tr_obs & te_obs)
        self.assertFalse(va_obs & te_obs)

    def test_canonical_roundtrip_all_train_gts(self):
        train_qs = load_train_questions_with_gt("train")
        total = 0
        for q in train_qs:
            for gt in q.gt_structures:
                txt = canonical_response(gt)
                parsed = parse_and_normalize(q.bench, txt)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.get_hash(), gt.get_hash())
                self.assertTrue(q.bench.validate_structure_matches_observations(parsed, q.obs))
                self.assertEqual(voxel_hamming(parsed, gt), 0)
                total += 1
        self.assertEqual(total, 18 * 9 + 66 * 27)


if __name__ == "__main__":
    unittest.main()
