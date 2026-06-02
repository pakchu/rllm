import unittest

from training.calibrated_regime_policy import CalibratedPolicyConfig, evaluate_rules, fit_rules


class TestCalibratedRegimePolicy(unittest.TestCase):
    def test_fit_rules_uses_train_group_expectancy_and_eval_fixed_action(self):
        cfg = CalibratedPolicyConfig(min_train_samples=2, min_train_mean_net=0.001, min_train_win_rate=0.5, max_train_mean_mae=0.02)
        train = [
            {"date": "d0", "key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.002, "utility": 0.008}}},
            {"date": "d1", "key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.02, "mae": 0.003, "utility": 0.017}}},
            {"date": "d2", "key": "bad", "actions": {"SHORT_48": {"side": "SHORT", "hold_bars": 48, "net_return": -0.01, "mae": 0.004, "utility": -0.014}}},
        ]
        rules = fit_rules(train, cfg)
        self.assertIn("k", rules)
        self.assertNotIn("bad", rules)
        ev = evaluate_rules(train, rules)
        self.assertEqual(ev["trades"], 2)
        self.assertGreater(ev["mean_net_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
