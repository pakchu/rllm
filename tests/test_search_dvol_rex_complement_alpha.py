import unittest

import numpy as np
import pandas as pd

from training.search_dvol_rex_complement_alpha import build_schedule, simulate_schedule


class TestDvolRexComplementAlpha(unittest.TestCase):
    def test_rex_short_preempts_open_dvol_long(self):
        events = [
            {"signal_pos": 1, "side": 1, "hold_bars": 10, "source": "dvol"},
            {"signal_pos": 5, "side": -1, "hold_bars": 2, "source": "rex"},
        ]
        schedule = build_schedule(events, "rex_short_preempt")
        self.assertEqual(schedule[0]["exit_pos"], 6)
        self.assertTrue(schedule[0]["preempted"])
        self.assertEqual(schedule[1]["source"], "rex")

    def test_union_does_not_preempt(self):
        events = [
            {"signal_pos": 1, "side": 1, "hold_bars": 10, "source": "dvol"},
            {"signal_pos": 5, "side": -1, "hold_bars": 2, "source": "rex"},
        ]
        schedule = build_schedule(events, "union")
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["exit_pos"], 12)

    def test_simulator_tracks_favorable_high_water_before_adverse(self):
        dates = pd.date_range("2023-01-01", periods=8, freq="5min")
        market = pd.DataFrame(
            {
                "date": dates,
                "open": np.full(8, 100.0),
                "high": [100, 100, 200, 100, 100, 100, 100, 100],
                "low": [100, 100, 90, 100, 100, 100, 100, 100],
                "close": np.full(8, 100.0),
            }
        )
        result = simulate_schedule(
            market,
            [{"signal_pos": 1, "side": 1, "hold_bars": 1, "source": "dvol"}],
            mode="union",
            window=(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")),
            leverage=1.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(result["return_pct"], 0.0, places=9)
        self.assertAlmostEqual(result["strict_mdd_pct"], 55.0, places=9)


if __name__ == "__main__":
    unittest.main()
