import unittest

import numpy as np

from training.event_candidate_pairwise_ranker import _pair_time_weights, build_pairs


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


if __name__ == "__main__":
    unittest.main()
