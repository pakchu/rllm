import unittest

from training.calibrated_regime_policy import CalibratedPolicyConfig
from training.yearly_stable_regime_policy import YearlyStableConfig, fit_yearly_stable_rules


class TestYearlyStableRegimePolicy(unittest.TestCase):
    def test_fit_requires_each_train_year_to_qualify(self):
        cfg = CalibratedPolicyConfig(min_train_samples=2, min_train_mean_net=0.0, min_train_win_rate=0.5, max_train_mean_mae=0.1)
        stable = YearlyStableConfig(min_year_samples=1, min_year_mean_net=0.0, min_year_win_rate=0.5, max_year_mean_mae=0.1)
        rows = [
            {"date": "2023-01-01", "key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.02, "mae": 0.01, "utility": 0.01}}},
            {"date": "2024-01-01", "key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": -0.01, "mae": 0.01, "utility": -0.02}}},
        ]
        self.assertNotIn("k", fit_yearly_stable_rules(rows, cfg, stable))
        rows[1]["actions"]["LONG_48"]["net_return"] = 0.01
        rows[1]["actions"]["LONG_48"]["utility"] = 0.0
        self.assertIn("k", fit_yearly_stable_rules(rows, cfg, stable))


if __name__ == "__main__":
    unittest.main()
