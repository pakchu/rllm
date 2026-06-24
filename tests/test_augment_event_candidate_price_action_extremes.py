import unittest

import pandas as pd

from training.augment_event_candidate_price_action_extremes import AugmentPriceActionExtremesCfg, augment_rows


class TestAugmentEventCandidatePriceActionExtremes(unittest.TestCase):
    def test_augment_rows_uses_backward_asof_and_preserves_reward(self):
        rows = [
            {"date": "2024-01-01 00:05:00", "feature_snapshot": {"x": 1.0}, "state_tokens": {}, "reward": {"rank_utility": 0.1}},
        ]
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:10:00"]),
                "pa_ext_36_max_high_bar_spread_pct": [0.01, 0.99],
            }
        )
        cfg = AugmentPriceActionExtremesCfg(input_jsonl="in", market_csv="m", output_jsonl="out", tolerance="5min")
        out, summary = augment_rows(rows, features, cfg)
        self.assertEqual(summary["matched_rows"], 1)
        self.assertAlmostEqual(out[0]["feature_snapshot"]["pa_ext_36_max_high_bar_spread_pct"], 0.01)
        self.assertEqual(out[0]["state_tokens"]["tok:pa_ext_36_max_high_bar_spread_pct"], "high")
        self.assertEqual(out[0]["reward"], {"rank_utility": 0.1})
        self.assertTrue(out[0]["leakage_guard"]["price_action_extreme_features_backward_asof"])


if __name__ == "__main__":
    unittest.main()
