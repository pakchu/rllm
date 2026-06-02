import unittest

from training.sweep_skip_gate_policy import evaluate_policy_with_skip, fit_skip_allowlist


class TestSweepSkipGatePolicy(unittest.TestCase):
    def test_fit_skip_allowlist_requires_router_edge(self):
        rows = [
            {"date": "2023-01-01", "summary": {"regime": "UP"}, "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}}},
            {"date": "2024-01-01", "summary": {"regime": "UP"}, "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}}},
            {"date": "2023-01-01", "summary": {"regime": "DOWN"}, "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": -0.01, "mae": 0.001, "utility": -0.011}}},
        ]
        allow = fit_skip_allowlist(rows, router_fields=("regime",), action_side="LONG", action_hold_bars=48, min_samples=2, min_mean_net=0, min_win_rate=0.5, max_mean_mae=0.01, min_good_years=2, min_year_samples=1, min_year_mean_net=0, min_year_win_rate=0.5, max_bad_year_mean_net=-0.005)
        self.assertIn("regime=UP", allow)
        self.assertNotIn("regime=DOWN", allow)

    def test_evaluate_policy_with_skip_blocks_unallowed_router(self):
        records = [{"date": "2025-01-01", "signal_pos": 1, "summary": {"regime": "DOWN", "key": "K"}, "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}}}]
        metrics = evaluate_policy_with_skip(records, base_rules={"key=K": {"action": {"side": "LONG", "hold_bars": 48}}}, base_key_fields=("key",), skip_allowlist={}, skip_router_fields=("regime",))
        self.assertEqual(metrics["trades"], 0)
