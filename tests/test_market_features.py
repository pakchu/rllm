import unittest
import warnings

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
    def test_zero_open_interest_does_not_emit_log_warning(self):
        market = _market_df()
        market.loc[10:20, "open_interest"] = 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*divide by zero encountered in log.*")
            frame = build_market_feature_frame(market, window_size=32)
        self.assertTrue(np.isfinite(frame["oi_change"].to_numpy(dtype=np.float64)).all())

    def test_build_market_feature_frame_contains_expected_columns(self):
        frame = build_market_feature_frame(_market_df(), window_size=32)
        self.assertEqual(len(frame), 120)
        for col in CORE_MARKET_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)
        for col in EXTENDED_MARKET_FEATURE_COLUMNS:
            self.assertIn(col, frame.columns)
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=np.float64)).all())

    def test_completed_timeframe_features_do_not_depend_on_frame_start_phase(self):
        rows = 80 * 24 * 12
        phase = np.linspace(0.0, 40.0, rows)
        close = 100.0 + np.linspace(0.0, 25.0, rows) + np.sin(phase) * 3.0
        market = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=rows, freq="5min"),
                "open": close * (1.0 - 0.0005),
                "high": close * (1.0 + 0.002),
                "low": close * (1.0 - 0.002),
                "close": close,
                "volume": 100.0 + np.cos(phase) * 10.0,
            }
        )

        full = build_market_feature_frame(market, window_size=144)
        # Start on a different three-day phase and a non-midnight timestamp.
        # Once both frames contain enough completed HTF history, identical
        # source rows must produce identical calendar features.
        tail_start = 10 * 24 * 12 + 27
        tail_market = market.iloc[tail_start:].reset_index(drop=True)
        tail = build_market_feature_frame(tail_market, window_size=144)
        compare_rows = 10 * 24 * 12
        columns = [
            column
            for column in full.columns
            if column.startswith(("htf_4h_", "htf_1d_", "htf_3d_", "htf_1w_"))
        ]

        np.testing.assert_allclose(
            full.loc[len(full) - compare_rows :, columns].to_numpy(),
            tail.loc[len(tail) - compare_rows :, columns].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
