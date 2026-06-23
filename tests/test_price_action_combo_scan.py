import unittest

from training.price_action_combo_scan import _feature_groups


class TestPriceActionComboScan(unittest.TestCase):
    def test_feature_groups_bundle_price_action_with_other_families(self):
        cols = [
            "pa_w36_high_candle_low_dist",
            "pa_w36_low_candle_high_dist",
            "trend_96",
            "body_ratio",
            "dxy_zscore",
            "funding_zscore",
        ]
        groups = _feature_groups(cols)
        self.assertEqual(len(groups["pa_only"]), 2)
        self.assertIn("trend_96", groups["pa_market"])
        self.assertIn("dxy_zscore", groups["pa_external"])
        self.assertIn("funding_zscore", groups["pa_derivatives"])
        self.assertIn("funding_zscore", groups["pa_market_external_derivatives"])


if __name__ == "__main__":
    unittest.main()
