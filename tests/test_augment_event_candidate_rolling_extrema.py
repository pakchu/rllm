import unittest

import pandas as pd

from training.augment_event_candidate_rolling_extrema import AugmentRollingExtremaCfg, augment_rows, build_rolling_extrema_features


class TestAugmentEventCandidateRollingExtrema(unittest.TestCase):
    def test_build_rolling_extrema_features(self):
        market = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4, freq="5min"), "open": [10, 11, 12, 13], "high": [11, 12, 14, 13], "low": [9, 10, 11, 12], "close": [10, 11, 13, 12]})
        feats = build_rolling_extrema_features(market, (3,))
        self.assertTrue(pd.isna(feats.loc[1, "rex_3_range_pos"]))
        self.assertAlmostEqual(feats.loc[2, "rex_3_range_pos"], (13 - 9) / (14 - 9))
        self.assertLessEqual(feats.loc[2, "rex_3_cur_to_max_pct"], 0.0)

    def test_augment_adds_side_interactions(self):
        rows = [{"date": "2024-01-01 00:05:00", "side": "SHORT", "feature_snapshot": {}, "state_tokens": {}, "reward": {}}]
        feats = pd.DataFrame({"date": pd.to_datetime(["2024-01-01 00:00:00"]), "rex_3_range_pos": [0.9], "rex_3_max_to_cur_pct": [1.0], "rex_3_cur_to_min_pct": [5.0]})
        cfg = AugmentRollingExtremaCfg(input_jsonl="i", market_csv="m", output_jsonl="o", windows=(3,), token_windows=(3,), tolerance="5min")
        out, summary = augment_rows(rows, feats, cfg)
        self.assertEqual(summary["matched_rows"], 1)
        self.assertEqual(out[0]["feature_snapshot"]["rex_3_range_pos_x_side"], -0.9)
        self.assertEqual(out[0]["state_tokens"]["tok:rex_3_loc"], "near_max")


if __name__ == "__main__":
    unittest.main()
