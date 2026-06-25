import unittest

from training.alpha_linear_combo_scan import _feature_groups, _invert_rule


class TestAlphaLinearComboScan(unittest.TestCase):
    def test_feature_groups_include_optional_fx_components(self):
        cols = [
            "dxy_zscore",
            "usdkrw_momentum",
            "btckrw_zscore",
            "fx_eurusd_zscore",
            "fx_usdjpy_momentum",
            "trend_12",
            "range_pos",
            "taker_imbalance",
        ]

        groups = _feature_groups(cols)

        self.assertIn("fx_components", groups)
        self.assertEqual(groups["fx_components"], ["fx_eurusd_zscore", "fx_usdjpy_momentum"])
        self.assertIn("btckrw_zscore", groups["external"])
        self.assertIn("fx_eurusd_zscore", groups["fx_plus_external"])
        self.assertIn("trend_12", groups["fx_external_plus_market"])

    def test_invert_rule_swaps_linear_combo_sides(self):
        rule = {"high_side": "LONG", "low_side": "SHORT", "low_threshold": -1, "high_threshold": 1}
        inv = _invert_rule(rule)
        self.assertEqual(inv["high_side"], "SHORT")
        self.assertEqual(inv["low_side"], "LONG")
        self.assertEqual(inv["low_threshold"], -1)


if __name__ == "__main__":
    unittest.main()
