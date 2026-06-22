import unittest

import pandas as pd

from training.feature_coverage_audit import _family, _feature_stats


class TestFeatureCoverageAudit(unittest.TestCase):
    def test_family_classification(self):
        self.assertEqual(_family("mkt__dxy_zscore"), "external_macro_kimchi")
        self.assertEqual(_family("mkt__htf_1w_return_4"), "higher_timeframe")
        self.assertEqual(_family("wave__cvd_mom_55"), "flow_volume")
        self.assertEqual(_family("mkt__funding_rate"), "derivatives_aux")

    def test_feature_stats_marks_constant_unusable(self):
        years = pd.Series(["2024", "2024", "2025"])
        stats = _feature_stats(pd.Series([0.0, 0.0, 0.0]), years, min_nonzero_fraction=0.1, min_std=1e-12)
        self.assertFalse(stats["usable"])
        self.assertEqual(stats["nonzero_fraction"], 0.0)

    def test_feature_stats_marks_varying_usable(self):
        years = pd.Series(["2024", "2024", "2025"])
        stats = _feature_stats(pd.Series([0.0, 1.0, 2.0]), years, min_nonzero_fraction=0.1, min_std=1e-12)
        self.assertTrue(stats["usable"])
        self.assertIn("2025", stats["per_year"])


if __name__ == "__main__":
    unittest.main()
