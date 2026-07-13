import unittest

import numpy as np
import pandas as pd

from training.search_rex_pre2025_exit_overlay import parse_fractions, simulate_exit_overlay


class TestRexPre2025ExitOverlay(unittest.TestCase):
    def test_fraction_parser_is_unique_and_nonnegative(self):
        self.assertEqual(parse_fractions("0.02,0,0.01,0.02"), (0.0, 0.01, 0.02))

    def test_same_bar_stop_wins_over_take(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="5min"),
                "open": np.full(6, 100.0),
                "high": [100, 100, 102, 100, 100, 100],
                "low": [100, 100, 98, 100, 100, 100],
                "close": np.full(6, 100.0),
            }
        )
        predictions = [
            {
                "date": "2024-01-01 00:05:00",
                "signal_pos": 1,
                "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 2},
            }
        ]
        result = simulate_exit_overlay(
            predictions,
            market,
            window=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
            hold_bars=2,
            stop_loss=0.01,
            take_profit=0.01,
            leverage=1.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(result["return_pct"], -1.0, places=9)
        self.assertEqual(result["exit_counts"], {"stop": 1})

    def test_no_stop_tracks_intratrade_high_water(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=6, freq="5min"),
                "open": np.full(6, 100.0),
                "high": [100, 100, 200, 100, 100, 100],
                "low": [100, 100, 90, 100, 100, 100],
                "close": np.full(6, 100.0),
            }
        )
        predictions = [
            {
                "date": "2024-01-01 00:05:00",
                "signal_pos": 1,
                "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 1},
            }
        ]
        result = simulate_exit_overlay(
            predictions,
            market,
            window=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
            hold_bars=1,
            stop_loss=0.0,
            take_profit=0.0,
            leverage=1.0,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertAlmostEqual(result["strict_mdd_pct"], 55.0, places=9)


if __name__ == "__main__":
    unittest.main()
