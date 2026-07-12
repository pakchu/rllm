import unittest
from datetime import datetime, timedelta

import pandas as pd

from training.hierarchical_direct_split_search import HierSimConfig
from training.hierarchical_regime_filter_search import RegimeFilter
from training.strict_bar_backtest import BarExecutionConfig, simulate_bar_by_bar


def _rows(n=20):
    base = datetime(2025, 1, 1)
    return [
        {
            "date": (base + timedelta(minutes=5 * i)).isoformat(sep=" "),
            "next_return": -0.99,  # should be ignored by strict bar simulator
            "_gate_margin": 5.0,
            "_side_margin": 2.0,
            "_side_dir": 1.0,
            "_features": {"trend_48": 0.02},
        }
        for i in range(n)
    ]


def _market(opens, lows=None, highs=None):
    base = datetime(2025, 1, 1)
    lows = lows or opens
    highs = highs or opens
    return pd.DataFrame(
        {
            "date": [base + timedelta(minutes=5 * i) for i in range(len(opens))],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": opens,
        }
    )


class TestStrictBarBacktest(unittest.TestCase):
    def test_uses_actual_open_to_open_bars_not_forward_return_column(self):
        # Signal at t=0, entry_delay=1 -> enter at open[1]=100, exit after
        # 2 bars at open[3]=104. next_return is intentionally very negative.
        market = _market([100, 100, 102, 104, 104], lows=[100, 100, 102, 104, 104], highs=[100, 100, 102, 104, 104])
        out = simulate_bar_by_bar(
            _rows(1),
            market,
            HierSimConfig(False, 3.0, 1.0, hold_bars=2, cooldown_bars=0),
            RegimeFilter("tf", abs_trend_min=0.01, align_mode="trend_follow", trend_col="trend_48"),
            BarExecutionConfig(1.0, 0.0, 0.0, 1.0, 1, 1.0, entry_delay_bars=1),
        )
        self.assertEqual(out["sim"]["trade_entries"], 1)
        self.assertAlmostEqual(out["sim"]["ret_pct"], 4.0, places=6)
        self.assertEqual(out["sim"]["return_application"], "actual_ohlc_bar_by_bar_strict_mdd")

    def test_strict_mdd_counts_intrabar_adverse_excursion(self):
        market = _market(
            [100, 100, 101, 101],
            lows=[100, 90, 101, 101],
            highs=[100, 101, 101, 101],
        )
        out = simulate_bar_by_bar(
            _rows(1),
            market,
            HierSimConfig(False, 3.0, 1.0, hold_bars=1, cooldown_bars=0),
            RegimeFilter("none"),
            BarExecutionConfig(1.0, 0.0, 0.0, 1.0, 1, 1.0, entry_delay_bars=1),
        )
        self.assertEqual(out["sim"]["trade_entries"], 1)
        self.assertAlmostEqual(out["sim"]["ret_pct"], 1.0, places=6)
        self.assertAlmostEqual(out["sim"]["strict_mdd_pct"], (1.0 - 0.9 / 1.01) * 100.0, places=6)

    def test_strict_mdd_uses_intratrade_favorable_high_water(self):
        market = _market(
            [100, 100, 100, 100],
            lows=[100, 90, 100, 100],
            highs=[100, 200, 100, 100],
        )
        out = simulate_bar_by_bar(
            _rows(1),
            market,
            HierSimConfig(False, 3.0, 1.0, hold_bars=1, cooldown_bars=0),
            RegimeFilter("none"),
            BarExecutionConfig(1.0, 0.0, 0.0, 1.0, 1, 1.0, entry_delay_bars=1),
        )
        self.assertAlmostEqual(out["sim"]["ret_pct"], 0.0, places=6)
        self.assertAlmostEqual(out["sim"]["strict_mdd_pct"], 55.0, places=6)


if __name__ == "__main__":
    unittest.main()
