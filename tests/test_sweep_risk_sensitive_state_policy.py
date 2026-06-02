import unittest

from training.sweep_risk_sensitive_state_policy import RiskSensitiveConfig, _risk_objective, fit_risk_sensitive_rules


class TestSweepRiskSensitiveStatePolicy(unittest.TestCase):
    def test_risk_objective_penalizes_mae_and_cvar(self):
        cfg = RiskSensitiveConfig(mae_weight=1.0, cvar_weight=2.0)
        self.assertAlmostEqual(_risk_objective({"mean_net_return": 0.1, "mean_mae": 0.01, "cvar_loss": 0.02}, cfg), 0.05)

    def test_fit_selects_lower_risk_action(self):
        rows = []
        for _ in range(3):
            rows.append({"key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.02, "mae": 0.02, "utility": 0.0}, "SHORT_48": {"side": "SHORT", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}}})
        rules = fit_risk_sensitive_rules(rows, RiskSensitiveConfig(min_samples=3, min_mean_net=0, min_win_rate=0, max_mean_mae=1, mae_weight=10.0, cvar_weight=0))
        self.assertEqual(rules["k"]["action"]["side"], "SHORT")
