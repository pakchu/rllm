import unittest

import pandas as pd

from training.price_action_extreme_feature_audit import build_extreme_bar_features


class TestPriceActionExtremeFeatureAudit(unittest.TestCase):
    def test_extreme_bar_features_use_high_bar_low_and_low_bar_high(self):
        market = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [11.0, 15.0, 13.0],
                "low": [9.0, 10.0, 8.0],
                "close": [10.5, 14.0, 12.0],
            }
        )
        feats = build_extreme_bar_features(market, (3,))
        row = feats.iloc[2]
        # Highest high is bar 1, so the requested paired min price is low[1] = 10.
        self.assertAlmostEqual(row["pa_ext_3_to_low_of_max_high_pct"], (12.0 - 10.0) / 12.0)
        # Lowest low is bar 2, so the requested paired max price is high[2] = 13.
        self.assertAlmostEqual(row["pa_ext_3_to_high_of_min_low_pct"], (12.0 - 13.0) / 12.0)
        self.assertAlmostEqual(row["pa_ext_3_max_high_age_frac"], 0.5)
        self.assertAlmostEqual(row["pa_ext_3_min_low_age_frac"], 0.0)

    def test_extreme_bar_features_are_nan_until_full_window(self):
        market = pd.DataFrame({"open": [1.0, 1.0], "high": [2.0, 3.0], "low": [0.5, 0.7], "close": [1.5, 2.5]})
        feats = build_extreme_bar_features(market, (3,))
        self.assertTrue(feats["pa_ext_3_range_pos"].isna().all())


if __name__ == "__main__":
    unittest.main()
