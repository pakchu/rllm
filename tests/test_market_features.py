import unittest

import numpy as np
import pandas as pd

from preprocessing.market_features import (
    CORE_MARKET_FEATURE_COLUMNS,
    EXTENDED_MARKET_FEATURE_COLUMNS,
    build_market_feature_frame,
)


def _market_df(n: int = 120) -> pd.DataFrame:
    base = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * (1.0 + np.sin(np.linspace(0, 8, n)) * 0.002),
            "volume": np.linspace(10.0, 40.0, n),
            "number_of_trades": np.linspace(100, 300, n),
            "taker_buy_base": np.linspace(3.0, 25.0, n),
            "funding_rate": np.sin(np.linspace(0, 6, n)) * 0.01,
            "open_interest": np.linspace(1000, 1300, n),
        }
    )


class TestMarketFeatures(unittest.TestCase):
    def test_build_market_feature_frame_contains_expected_columns(self):
        frame = build_market_feature_frame(_market_df(), window_size=32)
        self.assertEqual(len(frame), 120)
        for col in CORE_MARKET_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)
        for col in EXTENDED_MARKET_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=np.float64)).all())


if __name__ == "__main__":
    unittest.main()
