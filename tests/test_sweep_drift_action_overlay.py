import unittest

from training.sweep_drift_action_overlay import _resolve_action_key, _scale_outcome, simulate_action_overlay


class TestSweepDriftActionOverlay(unittest.TestCase):
    def test_resolve_flip_action(self):
        self.assertEqual(_resolve_action_key({"action": {"side": "LONG", "hold_bars": 144}}, "FLIP_48"), ("SHORT_48", 48))

    def test_scale_outcome(self):
        out = _scale_outcome({"net_return": 0.02, "mae": 0.01, "utility": 0.01}, 0.5)
        self.assertAlmostEqual(out["net_return"], 0.01)
        self.assertAlmostEqual(out["mae"], 0.005)

    def test_high_risk_can_flip(self):
        records = [{"date": "2025-01-01", "signal_pos": 10, "key": "k", "actions": {"LONG_144": {"side": "LONG", "hold_bars": 144, "net_return": -0.01, "mae": 0.02, "utility": -0.03}, "SHORT_48": {"side": "SHORT", "hold_bars": 48, "net_return": 0.02, "mae": 0.001, "utility": 0.019}}}]
        rep = simulate_action_overlay(records, rules={"k": {"action": {"side": "LONG", "hold_bars": 144}}}, baseline={"k": {"mean_net_return": 0.01}}, seed_history={"k": [{"net_return": -0.01, "mae": 0.001, "utility": -0.011}] * 3}, recent_window=3, min_recent_samples=3, high_min_mean_net=0.0, high_min_win_rate=0.45, high_max_mean_mae=0.02, medium_mean_drop=0.004, medium_action="KEEP", high_action="FLIP_48", medium_size=1.0, high_size=1.0, years=1.0)
        self.assertEqual(rep["metrics"]["trades"], 1)
        self.assertGreater(rep["metrics"]["compounded_return"], 0)
