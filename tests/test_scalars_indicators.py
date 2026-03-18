import unittest

import pandas as pd

from preprocessing.indicators import bollinger_bands, envelopes, mfi, rsi, sma
from preprocessing.scalars import build_scalar_frame, extract_scalars
from preprocessing.timeframe import make_window


class TestScalarsIndicators(unittest.TestCase):
    def test_extract_scalars(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=5, freq="1min"),
                "high": [10, 12, 11, 13, 14],
                "low": [8, 9, 9, 10, 11],
            }
        )
        window = make_window(df, t=4, w=3)
        scalars = extract_scalars(window, position_size_pct=50.0, last_entry_price=100.0)
        self.assertEqual(scalars["position_size_pct"], 50.0)
        self.assertEqual(scalars["last_entry_price"], 100.0)
        self.assertGreater(scalars["range_volatility_pct"], 0.0)

    def test_build_scalar_frame_len(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=10, freq="1min"),
                "high": [11 + i for i in range(10)],
                "low": [9 + i for i in range(10)],
            }
        )
        out = build_scalar_frame(df, window_size=4)
        self.assertEqual(len(out), 7)
        self.assertIn("range_volatility_pct", out.columns)

    def test_indicator_shapes(self):
        close = pd.Series([100 + i for i in range(50)], dtype=float)
        high = close + 1.0
        low = close - 1.0
        volume = pd.Series([1000 + i for i in range(50)], dtype=float)

        self.assertEqual(len(sma(close, 5)), len(close))
        self.assertEqual(len(bollinger_bands(close, 20)), len(close))
        self.assertEqual(len(envelopes(close, 10, 1.0)), len(close))
        self.assertEqual(len(rsi(close, 14)), len(close))
        self.assertEqual(len(mfi(high, low, close, volume, 14)), len(close))


if __name__ == "__main__":
    unittest.main()
