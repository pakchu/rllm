import unittest

from training.sweep_drift_aware_overlay import _parse_ints, _period_years, simulate_drift_overlay


class TestSweepDriftAwareOverlay(unittest.TestCase):
    def test_parse_ints(self):
        self.assertEqual(_parse_ints("1,2"), [1, 2])

    def test_period_years_positive(self):
        self.assertGreater(_period_years("2025-01-01", "2025-02-01"), 0)

    def test_simulate_drift_overlay_skips_high_risk(self):
        records = [{"date": "2025-01-01", "signal_pos": 10, "key": "k", "actions": {"LONG_48": {"side": "LONG", "hold_bars": 48, "net_return": 0.01, "mae": 0.001, "utility": 0.009}}}]
        rules = {"k": {"action": {"side": "LONG", "hold_bars": 48}}}
        rep = simulate_drift_overlay(
            records,
            rules=rules,
            baseline={"k": {"mean_net_return": 0.01}},
            seed_history={"k": [{"net_return": -0.01, "mae": 0.001, "utility": -0.011}] * 3},
            recent_window=3,
            min_recent_samples=3,
            high_min_mean_net=0.0,
            high_min_win_rate=0.45,
            high_max_mean_mae=0.02,
            medium_mean_drop=0.004,
            skip_risks=("HIGH",),
            years=1.0,
        )
        self.assertEqual(rep["metrics"]["trades"], 0)
