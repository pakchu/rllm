import unittest

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _attach_delayed_metrics, _simulate_no_stop


class TestPositioningDisagreementAlpha(unittest.TestCase):
    def test_metrics_are_delayed_one_complete_bar(self):
        market = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="5min")})
        metrics = pd.DataFrame(
            {
                "create_time": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "symbol": "BTCUSDT",
                "ratio": [1.0, 2.0, 3.0],
            }
        )
        joined = _attach_delayed_metrics(market, metrics, tolerance="10min", delay_bars=1)
        self.assertTrue(np.isnan(joined.loc[0, "ratio"]))
        self.assertEqual(joined.loc[1:, "ratio"].tolist(), [1.0, 2.0])
        self.assertEqual(joined["positioning_available"].tolist(), [0.0, 1.0, 1.0])

    def test_no_stop_sim_annualises_full_window_and_tracks_adverse_path(self):
        dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31 23:55", freq="5min"))
        n = len(dates)
        market = pd.DataFrame(
            {
                "open": np.full(n, 100.0),
                "high": np.full(n, 100.0),
                "low": np.full(n, 100.0),
            }
        )
        market.loc[1, "low"] = 80.0
        market.loc[2, "open"] = 110.0
        long_active = np.zeros(n, dtype=bool)
        short_active = np.zeros(n, dtype=bool)
        long_active[0] = True
        result = _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window="select2023",
            hold_bars=1,
            stride_bars=1,
            leverage=0.5,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        self.assertEqual(result["trades"], 1)
        self.assertAlmostEqual(result["return_pct"], 5.0, places=6)
        self.assertAlmostEqual(result["strict_mdd_pct"], 10.0, places=6)
        self.assertAlmostEqual(result["cagr_pct"], 5.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
