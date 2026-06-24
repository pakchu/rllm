import unittest

import numpy as np

from training.event_candidate_pairwise_ranker import _pair_time_weights, build_pairs
from training.event_candidate_ridge_ranker import _feature_names, _write_policy


class TestEventCandidatePairwiseRanker(unittest.TestCase):
    def test_build_pairs_uses_best_against_gap_filtered_losers(self):
        rows = [
            {"date": "d", "signal_pos": 1, "reward": {"rank_utility": 0.02}},
            {"date": "d", "signal_pos": 1, "reward": {"rank_utility": 0.0195}},
            {"date": "d", "signal_pos": 1, "reward": {"rank_utility": -0.01}},
            {"date": "d2", "signal_pos": 2, "reward": {"rank_utility": 0.0}},
        ]
        self.assertEqual(build_pairs(rows, max_pairs_per_signal=4, min_utility_gap=0.001), [(0, 2)])

    def test_build_pairs_caps_per_signal(self):
        rows = [{"date": "d", "signal_pos": 1, "reward": {"rank_utility": 1.0 - i}} for i in range(5)]
        self.assertEqual(len(build_pairs(rows, max_pairs_per_signal=2, min_utility_gap=0.0)), 2)

    def test_pair_time_weights_favor_recent_pairs(self):
        rows = [
            {"date": "2024-01-01 00:00:00"},
            {"date": "2024-01-11 00:00:00"},
        ]
        weights = _pair_time_weights(rows, [(0, 1), (1, 0)], half_life_days=10)
        self.assertIsNotNone(weights)
        self.assertTrue(np.allclose(weights, [0.5, 1.0]))

    def test_write_policy_can_filter_allowed_sides(self):
        import tempfile
        from pathlib import Path
        best = [
            {"row": {"date": "d", "signal_pos": 1, "side": "LONG", "candidate": {"hold_bars": 1}}, "score": 2.0},
            {"row": {"date": "d2", "signal_pos": 2, "side": "SHORT", "candidate": {"hold_bars": 1}}, "score": 2.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = _write_policy(best, str(Path(tmp) / "p.jsonl"), threshold=1.0, full_margin=0.0, allowed_sides={"LONG"})
        self.assertEqual(summary["counts"]["TRADE"], 1)
        self.assertEqual(summary["counts"]["NO_TRADE"], 1)

    def test_write_policy_can_scale_sides(self):
        import json, tempfile
        from pathlib import Path
        best = [{"row": {"date": "d", "signal_pos": 1, "side": "LONG", "candidate": {"hold_bars": 1}}, "score": 2.0}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p.jsonl"
            _write_policy(best, str(out), threshold=1.0, full_margin=0.0, side_scale_by_side={"LONG": 0.25})
            row = json.loads(out.read_text().strip())
        self.assertAlmostEqual(row["position_scale"], 0.25)

    def test_feature_names_can_drop_prefixes(self):
        rows = [{"feature_snapshot": {"a": 1.0, "rex_x": 2.0}, "state_tokens": {}}]
        nums, _ = _feature_names(rows, drop_prefixes=("rex_",))
        self.assertEqual(nums, ["a"])

    def test_write_policy_can_filter_max_feature(self):
        import tempfile
        from pathlib import Path
        best = [
            {"row": {"date": "d", "signal_pos": 1, "side": "LONG", "candidate": {"hold_bars": 1}, "feature_snapshot": {"rex": 2.0}}, "score": 2.0},
            {"row": {"date": "d2", "signal_pos": 2, "side": "LONG", "candidate": {"hold_bars": 1}, "feature_snapshot": {"rex": 5.0}}, "score": 2.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = _write_policy(best, str(Path(tmp) / "p.jsonl"), threshold=1.0, full_margin=0.0, max_feature_name="rex", max_feature_value=3.0)
        self.assertEqual(summary["counts"]["TRADE"], 1)
        self.assertEqual(summary["counts"]["NO_TRADE"], 1)


if __name__ == "__main__":
    unittest.main()
