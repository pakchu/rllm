import unittest

from training.rolling_event_context_preference_ranker import FeatureSpace, _predict_rows, _utility_pct


class TestRollingEventContextPreferenceRanker(unittest.TestCase):
    def test_wait_utility_is_zero(self):
        row = {"reward_audit": {"LONG": {"net_return_pct": 1.0}, "SHORT": {"net_return_pct": -1.0}}}
        self.assertEqual(_utility_pct(row, "WAIT"), 0.0)
        self.assertEqual(_utility_pct(row, "LONG"), 1.0)

    def test_predict_rows_can_abstain(self):
        rows = [{"date": "2024-01-01", "signal_pos": 1, "state_tokens": {"pa_event_pressure": "none"}, "reward_audit": {"LONG": {"net_return_pct": -1.0}, "SHORT": {"net_return_pct": -1.0}}}]
        fs = FeatureSpace.fit(rows, min_count=1)
        preds = _predict_rows(rows, fs, w=[0.0] * len(fs.vocab), edge_threshold=0.1, min_gap=0.0)
        self.assertEqual(preds[0]["prediction"]["gate"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
