import unittest

import pandas as pd

from training.augment_event_candidate_market_regime import AugmentMarketRegimeCfg, augment_rows, build_market_regime_features


class TestAugmentEventCandidateMarketRegime(unittest.TestCase):
    def test_build_market_regime_features_uses_past_window(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="5min"),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
            }
        )
        feats = build_market_regime_features(market, (3,))
        self.assertTrue(pd.isna(feats.loc[1, "mreg_3_range_pos"]))
        self.assertAlmostEqual(feats.loc[3, "mreg_3_ret_pct"], 3.0)
        self.assertGreaterEqual(feats.loc[3, "mreg_3_range_pos"], 0.0)

    def test_augment_rows_backward_asof(self):
        rows = [{"date": "2024-01-01 00:05:00", "feature_snapshot": {}, "state_tokens": {}, "reward": {"rank_utility": 0.1}}]
        feats = pd.DataFrame({"date": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:10:00"]), "mreg_3_ret_pct": [1.0, 99.0], "mreg_3_range_pos": [0.9, 0.1]})
        cfg = AugmentMarketRegimeCfg(input_jsonl="in", market_csv="m", output_jsonl="out", windows=(3,), token_windows=(3,), tolerance="5min")
        out, summary = augment_rows(rows, feats, cfg)
        self.assertEqual(summary["matched_rows"], 1)
        self.assertEqual(out[0]["feature_snapshot"]["mreg_3_ret_pct"], 1.0)
        self.assertEqual(out[0]["state_tokens"]["tok:mreg_3_range"], "upper")


if __name__ == "__main__":
    unittest.main()
