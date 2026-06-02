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

    def test_evaluate_rules_can_skip_overlapping_positions_and_count_mae_drawdown(self):
        rules = {
            "k": {
                "action": {"side": "LONG", "hold_bars": 3},
            }
        }
        rows = [
            {"date": "d0", "signal_pos": 0, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.10, "mae": 0.05, "utility": 0.05}}},
            {"date": "d1", "signal_pos": 1, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": 0.10, "mae": 0.02, "utility": 0.08}}},
            {"date": "d4", "signal_pos": 4, "key": "k", "actions": {"LONG_3": {"side": "LONG", "hold_bars": 3, "net_return": -0.01, "mae": 0.20, "utility": -0.21}}},
        ]
        ev = evaluate_rules(rows, rules, non_overlapping=True, include_intratrade_mdd=True)
        self.assertEqual(ev["trades"], 2)
        self.assertTrue(ev["non_overlapping"])
        self.assertGreater(ev["strict_mdd_proxy"], 0.19)


if __name__ == "__main__":
    unittest.main()
