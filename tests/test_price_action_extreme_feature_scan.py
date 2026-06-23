import unittest

import numpy as np
import pandas as pd

from training.price_action_extreme_feature_scan import build_price_action_extreme_features, extreme_bar_levels


class TestPriceActionExtremeFeatureScan(unittest.TestCase):
    def test_extreme_bar_levels_use_extreme_candle_opposite_price(self):
        high = np.array([10, 12, 11, 9, 13], dtype=float)
        low = np.array([8, 9, 7, 6, 10], dtype=float)
        levels = extreme_bar_levels(high, low, 3)
        # i=2 window rows 0..2: max high row1 => low 9, min low row2 => high 11
        self.assertAlmostEqual(levels["low_at_window_high"][2], 9.0)
        self.assertAlmostEqual(levels["high_at_window_low"][2], 11.0)
        # i=4 window rows 2..4: max high row4 => low 10, min low row3 => high 9
        self.assertAlmostEqual(levels["low_at_window_high"][4], 10.0)
        self.assertAlmostEqual(levels["high_at_window_low"][4], 9.0)

    def test_build_features_are_relative_and_no_future_needed(self):
        market = pd.DataFrame({
            "close": [9, 11, 10, 8, 12],
            "high": [10, 12, 11, 9, 13],
            "low": [8, 9, 7, 6, 10],
        })
        feat = build_price_action_extreme_features(market, [3])
        self.assertIn("pa_w3_high_candle_low_dist", feat.columns)
        self.assertAlmostEqual(float(feat.loc[2, "pa_w3_high_candle_low_dist"]), (10 - 9) / 10)
        self.assertAlmostEqual(float(feat.loc[2, "pa_w3_low_candle_high_dist"]), (10 - 11) / 10)
        self.assertEqual(float(feat.loc[0, "pa_w3_high_candle_low_dist"]), 0.0)


if __name__ == "__main__":
    unittest.main()
