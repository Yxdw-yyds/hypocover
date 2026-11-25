import unittest
import numpy as np
from c4 import layered_actions as LA


class Layered(unittest.TestCase):
    def test_k_and_hints(self):
        self.assertEqual(LA.k_of("100000101"), 3)
        self.assertIn("more than 6", LA.hint_text("high", 3))
        self.assertIn("exactly 4", LA.hint_text("mid", 2))
        self.assertIn("fewer than 6", LA.hint_text("low", 3))
        self.assertEqual(LA.hint_text("none", 3), "")
        p = "PROMPT\n"
        self.assertEqual(LA.add_hint(p, "none", 3), p)
        self.assertTrue(LA.add_hint(p, "high", 3).startswith("PROMPT"))

    def test_tiers_and_compliance(self):
        self.assertEqual([LA.tier_of(b, 3) for b in (3, 5, 6, 7, 9)], ["low", "low", "mid", "high", "high"])
        self.assertTrue(LA.complies("high", 3, 7)); self.assertFalse(LA.complies("high", 3, 6))
        self.assertTrue(LA.complies("none", 3, 3))

    def test_gt_tier_counts_match_earlier_analysis(self):
        self.assertEqual(LA.gt_tier_counts(3), {"low": 10, "mid": 7, "high": 10})
        self.assertEqual(LA.gt_tier_counts(2), {"low": 3, "mid": 3, "high": 3})

    def test_tier_static_schedule_is_proportional(self):
        picks = [LA.choose_tier_static(i, 3) for i in range(27)]
        self.assertEqual({t: picks.count(t) for t in LA.TIERS}, {"low": 10, "mid": 7, "high": 10})
        picks9 = [LA.choose_tier_static(i, 2) for i in range(9)]
        self.assertEqual({t: picks9.count(t) for t in LA.TIERS}, {"low": 3, "mid": 3, "high": 3})

    def test_exact_counts_and_paper_schedule(self):
        self.assertEqual(LA.exact_counts(3), {3: 1, 4: 3, 5: 6, 6: 7, 7: 6, 8: 3, 9: 1})
        self.assertEqual(LA.exact_counts(2), {2: 1, 3: 2, 4: 3, 5: 2, 6: 1})
        seq = [LA.choose_paper_stratified(i, 27, 3) for i in range(27)]
        self.assertEqual(seq[0], "exact:3")
        self.assertEqual(seq[-1], "exact:9")
        self.assertEqual({c: seq.count(f"exact:{c}") for c in range(3, 10)}, LA.exact_counts(3))
        self.assertEqual([int(x[6:]) for x in seq], sorted(int(x[6:]) for x in seq))  # ascending levels
        seq2 = [LA.choose_paper_stratified(i, 9, 2) for i in range(9)]
        self.assertEqual({c: seq2.count(f"exact:{c}") for c in range(2, 7)}, LA.exact_counts(2))

    def test_exact_action_text_and_compliance(self):
        self.assertIn("exactly 7", LA.hint_text("exact:7", 3))
        self.assertTrue(LA.complies("exact:7", 3, 7)); self.assertFalse(LA.complies("exact:7", 3, 6))

    def test_cycle_and_random(self):
        self.assertEqual([LA.choose_tier_cycle(i) for i in range(4)], ["high", "mid", "low", "high"])
        rng = np.random.default_rng(0)
        self.assertTrue(all(LA.choose_random_level(rng) in LA.TIERS for _ in range(50)))
        self.assertIn("none", {LA.choose_random_action(rng) for _ in range(200)})

    def test_features_history_only(self):
        x0 = LA.features([], 0, 27, 3)
        self.assertEqual(len(x0), len(LA.FEATURE_NAMES))
        h = [{"action": "high", "outcome": "invalid", "tier": "high"}, {"action": "none", "outcome": "new", "tier": "low"}]
        x = LA.features(h, 2, 27, 3)
        self.assertAlmostEqual(x[LA.FEATURE_NAMES.index("new_low")], 1 / 27)
        self.assertAlmostEqual(x[LA.FEATURE_NAMES.index("invalid_high")], 1 / 27)
        self.assertEqual(x[LA.FEATURE_NAMES.index("last_none")], 1.0)
        # tier SIZES / answer-set quantities must not be features
        self.assertFalse(any("size" in n or "gt" in n for n in LA.FEATURE_NAMES))

    def test_learned_choice(self):
        d = len(LA.FEATURE_NAMES)
        w = {a: [0.0] * d for a in LA.ACTIONS}; w["mid"][0] = 5.0
        self.assertEqual(LA.choose_learned({"actions": list(LA.ACTIONS), "weights": w}, [], 0, 27, 3), "mid")


if __name__ == "__main__":
    unittest.main()
