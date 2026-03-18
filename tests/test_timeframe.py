import unittest

import pandas as pd

from preprocessing.timeframe import aggregate_ohlcv, make_window


class TestTimeframeAggregation(unittest.TestCase):
    def test_aggregate_5m_drops_incomplete_last_candle(self):
        # 00:00 ~ 00:10 (11 bars): for 5m bins -> [00:00, 00:05, 00:10(incomplete)]
        # policy: drop last aggregated candle
        dates = pd.date_range("2025-01-01 00:00:00", periods=11, freq="1min")
        base = pd.Series(range(100, 111), dtype=float)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 1.0,
                "tic": "BTCUSDT",
            }
        )

        out = aggregate_ohlcv(df, timeframe="5m", drop_incomplete_last_candle=True)

        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["date"], pd.Timestamp("2025-01-01 00:00:00"))
        self.assertEqual(out.iloc[1]["date"], pd.Timestamp("2025-01-01 00:05:00"))

        self.assertEqual(out.iloc[0]["open"], 100.0)
        self.assertEqual(out.iloc[0]["high"], 105.0)
        self.assertEqual(out.iloc[0]["low"], 99.0)
        self.assertEqual(out.iloc[0]["close"], 104.5)
        self.assertEqual(out.iloc[0]["volume"], 5.0)

    def test_make_window_is_leak_safe(self):
        df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=200, freq="1min")})
        window = make_window(df, t=100, w=96)

        self.assertEqual(len(window), 96)
        self.assertEqual(window.index.min(), 5)
        self.assertEqual(window.index.max(), 100)
        self.assertNotIn(df.iloc[101]["date"], set(window["date"]))

    def test_make_window_requires_sufficient_history(self):
        df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=20, freq="1min")})
        with self.assertRaises(ValueError):
            make_window(df, t=10, w=96)


if __name__ == "__main__":
    unittest.main()

