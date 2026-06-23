import unittest

from training.rolling_event_context_utility_ranker import FeatureSpace, _candidate_value_pct, _predict_rows


class TestRollingEventContextUtilityRanker(unittest.TestCase):
    def test_candidate_value_reads_reward_audit_percent(self):
        row = {"reward_audit": {"LONG": {"net_return_pct": 1.25}, "SHORT": {"net_return_pct": -0.4}}}
        self.assertEqual(_candidate_value_pct(row, "LONG"), 1.25)
        self.assertEqual(_candidate_value_pct(row, "SHORT"), -0.4)

    def test_feature_space_predicts_both_sides(self):
        rows = [
            {"date": "2024-01-01", "signal_pos": 1, "state_tokens": {"pa_event_pressure": "downside", "trend_alignment": "mixed"}, "reward_audit": {"LONG": {"net_return_pct": 1.0}, "SHORT": {"net_return_pct": -1.0}}},
            {"date": "2024-01-02", "signal_pos": 2, "state_tokens": {"pa_event_pressure": "upside", "trend_alignment": "mixed"}, "reward_audit": {"LONG": {"net_return_pct": -1.0}, "SHORT": {"net_return_pct": 1.0}}},
        ]
        fs = FeatureSpace.fit(rows, min_count=1)
        x = fs.matrix_candidates(rows)
        self.assertEqual(x.shape[0], 4)
        preds = _predict_rows(rows, fs, w=[0.0] * x.shape[1], threshold=0.0, min_gap=0.0)
        self.assertEqual(len(preds), 2)
        self.assertIn(preds[0]["prediction"]["gate"], {"TRADE", "NO_TRADE"})


if __name__ == "__main__":
    unittest.main()
