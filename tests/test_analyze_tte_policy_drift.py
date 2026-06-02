import unittest

from training.analyze_tte_policy_drift import _aggregate_flat, _by_month


class TestAnalyzeTtePolicyDrift(unittest.TestCase):
    def test_aggregate_flat_tracks_win_rate_and_sum(self):
        stats = _aggregate_flat([
            {"net_return": 0.01, "mae": 0.001, "utility": 0.009},
            {"net_return": -0.02, "mae": 0.002, "utility": -0.022},
        ])
        self.assertEqual(stats["samples"], 2)
        self.assertAlmostEqual(stats["win_rate"], 0.5)
        self.assertAlmostEqual(stats["sum_net_return"], -0.01)

    def test_by_month_groups_dates(self):
        rows = [{"date": "2025-01-01", "net_return": 0.01, "mae": 0.0, "utility": 0.01}]
        self.assertIn("2025-01", _by_month(rows))
