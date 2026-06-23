import unittest

from training.rolling_price_action_combo_scan import _month_starts, _score_validation


class TestRollingPriceActionComboScan(unittest.TestCase):
    def test_month_starts_excludes_end_boundary(self):
        months = _month_starts("2026-01-15", "2026-04-01")
        self.assertEqual([str(m.date()) for m in months], ["2026-01-01", "2026-02-01", "2026-03-01"])

    def test_validation_score_rejects_thin_or_high_mdd_candidates(self):
        report = {
            "sim": {"trade_entries": 19, "strict_mdd_pct": 10.0, "cagr_pct": 50.0, "cagr_to_strict_mdd": 5.0},
            "trade_stats": {"p_value_mean_ret_approx": 0.01, "mean_trade_ret_pct": 0.2},
        }
        self.assertLess(_score_validation(report, min_trades=20, max_mdd=25.0), -1e8)
        report["sim"]["trade_entries"] = 20
        report["sim"]["strict_mdd_pct"] = 30.0
        self.assertLess(_score_validation(report, min_trades=20, max_mdd=25.0), -1e8)
        report["sim"]["strict_mdd_pct"] = 10.0
        self.assertGreater(_score_validation(report, min_trades=20, max_mdd=25.0), 0.0)


if __name__ == "__main__":
    unittest.main()
