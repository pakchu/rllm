import unittest

from training.alpha_linear_combo_scan import _feature_groups as linear_feature_groups
from training.wave_feature_ridge_policy import _groups as wave_feature_groups
from training.alpha_regime_rule_scan import _default_regime_columns


class TestAlphaFeatureGroups(unittest.TestCase):
    def test_linear_derivatives_group_uses_real_aux_and_excludes_missing_oi(self):
        groups = linear_feature_groups([
            "funding_rate",
            "funding_zscore",
            "funding_available",
            "premium_index_zscore",
            "premium_available",
            "binance_aux_any_available",
            "oi_change",
            "oi_zscore",
        ])
        self.assertIn("derivatives_aux", groups)
        self.assertIn("funding_rate", groups["derivatives_aux"])
        self.assertIn("premium_index_zscore", groups["derivatives_aux"])
        self.assertNotIn("oi_change", groups["derivatives_aux"])
        self.assertNotIn("oi_zscore", groups["derivatives_aux"])

    def test_regime_columns_exclude_missing_oi_and_include_repaired_aux(self):
        cols = ["funding_zscore", "funding_rate", "premium_index_zscore", "premium_index_change", "oi_zscore"]
        selected = _default_regime_columns(cols)
        self.assertIn("funding_zscore", selected)
        self.assertIn("premium_index_zscore", selected)
        self.assertNotIn("oi_zscore", selected)

    def test_wave_derivative_groups_include_funding_and_premium(self):
        groups = wave_feature_groups([
            "mom_12",
            "flow_mom",
            "funding_zscore",
            "premium_index_change",
            "dxy_zscore",
        ])
        self.assertIn("funding_zscore", groups["wave_derivatives"])
        self.assertIn("premium_index_change", groups["wave_external_derivatives"])
        self.assertIn("dxy_zscore", groups["wave_external_derivatives"])


if __name__ == "__main__":
    unittest.main()
