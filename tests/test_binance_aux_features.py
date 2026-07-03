import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_features, attach_binance_um_aux_frames, normalise_premium_index_frame
from preprocessing.market_features import build_market_feature_frame


class TestBinanceAuxFeatures(unittest.TestCase):
    def test_backward_asof_funding_and_premium_are_causal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market = pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01 00:00:00", periods=4, freq="30min"),
                    "open": [100.0, 101.0, 102.0, 103.0],
                    "high": [101.0, 102.0, 103.0, 104.0],
                    "low": [99.0, 100.0, 101.0, 102.0],
                    "close": [100.5, 101.5, 102.5, 103.5],
                    "volume": [10.0, 11.0, 12.0, 13.0],
                }
            )
            funding = pd.DataFrame(
                {
                    "date": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
                    "funding_rate": [0.001, -0.002],
                }
            )
            funding_path = root / "funding.csv"
            funding.to_csv(funding_path, index=False)
            premium = pd.DataFrame(
                {
                    "date": ["2025-01-01 00:00:00"],
                    "close": [100.25],
                    "close_time": [int(pd.Timestamp("2025-01-01 00:59:59", tz="UTC").timestamp() * 1000)],
                }
            )
            premium_path = root / "premium.csv"
            premium.to_csv(premium_path, index=False)

            out = attach_binance_um_aux_features(
                market,
                funding_csv=funding_path,
                premium_csv=premium_path,
                funding_tolerance="2h",
                premium_tolerance="2h",
            )

        self.assertEqual(out["funding_rate"].tolist(), [0.001, 0.001, -0.002, -0.002])
        self.assertEqual(out["funding_available"].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(out.loc[0, "premium_index"]))
        self.assertTrue(np.isnan(out.loc[1, "premium_index"]))
        self.assertEqual(float(out.loc[2, "premium_index"]), 100.25)
        self.assertEqual(out["premium_available"].tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_preserves_input_order_when_market_is_unsorted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            market = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 00:00:00"]),
                    "open": [102.0, 100.0],
                    "high": [103.0, 101.0],
                    "low": [101.0, 99.0],
                    "close": [102.5, 100.5],
                    "volume": [12.0, 10.0],
                }
            )
            funding_path = root / "funding.csv"
            pd.DataFrame(
                {
                    "date": ["2025-01-01 00:00:00", "2025-01-01 01:00:00"],
                    "funding_rate": [0.001, -0.002],
                }
            ).to_csv(funding_path, index=False)
            out = attach_binance_um_aux_features(market, funding_csv=funding_path, funding_tolerance="2h")
        self.assertEqual(pd.to_datetime(out["date"]).tolist(), pd.to_datetime(market["date"]).tolist())
        self.assertEqual(out["funding_rate"].tolist(), [-0.002, 0.001])

    def test_attach_aux_from_db_like_frames(self):
        market = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01 00:00:00", periods=4, freq="1min"),
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [10.0, 11.0, 12.0, 13.0],
            }
        )
        funding = pd.DataFrame(
            {
                "funding_time": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
                "funding_rate": [0.0001],
            }
        )
        premium = pd.DataFrame(
            {
                "ts": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
                "close_time": [pd.Timestamp("2026-01-01 00:00:59.999", tz="UTC")],
                "close": [-0.0002],
            }
        )

        out = attach_binance_um_aux_frames(
            market,
            funding_frame=funding,
            premium_frame=premium,
            funding_tolerance="12h",
            premium_tolerance="5min",
            zscore_window=2,
        )

        self.assertEqual(out["funding_available"].tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(float(out.loc[0, "funding_rate"]), 0.0001)
        self.assertTrue(np.isnan(out.loc[0, "premium_index"]))
        self.assertEqual(float(out.loc[1, "premium_index"]), -0.0002)
        self.assertEqual(out["premium_available"].tolist(), [0.0, 1.0, 1.0, 1.0])

    def test_premium_normaliser_accepts_timestamp_close_time(self):
        premium = pd.DataFrame({"close_time": [pd.Timestamp("2026-01-01 00:00:59.999", tz="UTC")], "close": ["-0.001"]})
        out = normalise_premium_index_frame(premium)
        self.assertEqual(str(out.loc[0, "date"]), "2026-01-01 00:00:59.999000")
        self.assertEqual(float(out.loc[0, "premium_index"]), -0.001)

    def test_market_feature_frame_exposes_binance_aux_columns(self):
        n = 120
        base = np.linspace(100.0, 105.0, n)
        market = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=n, freq="5min"),
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base,
                "volume": np.linspace(10.0, 20.0, n),
                "funding_rate": np.sin(np.linspace(0, 6, n)) * 0.001,
                "funding_available": 1.0,
                "premium_index": np.linspace(99.0, 101.0, n),
                "premium_index_zscore": np.sin(np.linspace(0, 4, n)),
                "premium_index_change": np.linspace(-0.01, 0.01, n),
                "premium_available": 1.0,
                "binance_aux_any_available": 1.0,
            }
        )
        frame = build_market_feature_frame(market, window_size=32)
        for col in [
            "funding_available",
            "premium_index",
            "premium_index_zscore",
            "premium_index_change",
            "premium_available",
            "binance_aux_any_available",
        ]:
            self.assertIn(col, frame.columns)
            self.assertTrue(np.isfinite(frame[col].to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
