import unittest

from training.calibrated_regime_policy import CalibratedPolicyConfig
from training.sweep_yearly_stable_policy import _fit_from_stats
from training.yearly_stable_regime_policy import YearlyStableConfig


class TestSweepYearlyStablePolicy(unittest.TestCase):
    def test_fit_from_stats_requires_yearly_thresholds(self):
        stats = {
            "k": {
                "group_samples": 2,
                "actions": [
                    {
                        "overall": {
                            "samples": 2,
                            "side": "LONG",
                            "hold_bars": 48,
                            "mean_net_return": 0.01,
                            "mean_utility": 0.0,
                            "win_rate": 1.0,
                            "mean_mae": 0.01,
                        },
                        "yearly": {
                            "2023": {
                                "samples": 1,
                                "side": "LONG",
                                "hold_bars": 48,
                                "mean_net_return": 0.01,
                                "mean_utility": 0.0,
                                "win_rate": 1.0,
                                "mean_mae": 0.01,
                            },
                            "2024": {
                                "samples": 1,
                                "side": "LONG",
                                "hold_bars": 48,
                                "mean_net_return": -0.01,
                                "mean_utility": -0.02,
                                "win_rate": 0.0,
                                "mean_mae": 0.01,
                            },
                        },
                    }
                ],
            }
        }
        cfg = CalibratedPolicyConfig(
            min_train_samples=2,
            min_train_mean_net=0,
            min_train_win_rate=0.5,
            max_train_mean_mae=0.1,
        )
        stable = YearlyStableConfig(
            min_year_samples=1,
            min_year_mean_net=0,
            min_year_win_rate=0.5,
            max_year_mean_mae=0.1,
        )
        self.assertFalse(_fit_from_stats(stats, cfg, stable))
