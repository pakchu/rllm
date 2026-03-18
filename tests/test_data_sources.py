import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from training.data_sources import load_market_data

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\..*")


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="1min"),
            "open": [100 + i for i in range(12)],
            "high": [101 + i for i in range(12)],
            "low": [99 + i for i in range(12)],
            "close": [100.5 + i for i in range(12)],
            "volume": [1.0 for _ in range(12)],
            "tic": ["BTCUSDT" for _ in range(12)],
        }
    )


class TestDataSources(unittest.TestCase):
    def test_load_synthetic(self):
        df = load_market_data(source="synthetic", num_rows=200, seed=1)
        self.assertEqual(len(df), 200)
        self.assertTrue({"date", "open", "high", "low", "close", "volume", "tic"}.issubset(df.columns))

    def test_load_synthetic_aggregate(self):
        df = load_market_data(source="synthetic", num_rows=120, seed=1, timeframe="5m")
        self.assertGreater(len(df), 0)
        self.assertLess(len(df), 120)

    def test_load_synthetic_regime_parameter_affects_series(self):
        base = load_market_data(
            source="synthetic",
            num_rows=200,
            seed=7,
            synthetic_regime_amplitude=0.0,
        )
        regime = load_market_data(
            source="synthetic",
            num_rows=200,
            seed=7,
            synthetic_regime_amplitude=0.001,
            synthetic_regime_period=50,
        )
        self.assertFalse(base["open"].equals(regime["open"]))

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mkt.csv"
            _base_df().to_csv(path, index=False)
            df = load_market_data(source="csv", input_csv=str(path), symbol="BTCUSDT")
            self.assertEqual(len(df), 12)
            self.assertEqual(df["tic"].iloc[0], "BTCUSDT")

    def test_load_csv_timeframe_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mkt.csv"
            _base_df().to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_market_data(
                    source="csv",
                    input_csv=str(path),
                    symbol="BTCUSDT",
                    timeframe="5m",
                )

    def test_load_csv_timeframe_match_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mkt_5m.csv"
            df_5m = _base_df().iloc[::5].copy()
            df_5m["date"] = pd.date_range("2025-01-01", periods=len(df_5m), freq="5min")
            df_5m.to_csv(path, index=False)
            loaded = load_market_data(
                source="csv",
                input_csv=str(path),
                symbol="BTCUSDT",
                timeframe="5m",
            )
            self.assertEqual(len(loaded), len(df_5m))

    def test_load_csv_applies_date_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mkt.csv"
            _base_df().to_csv(path, index=False)
            df = load_market_data(
                source="csv",
                input_csv=str(path),
                symbol="BTCUSDT",
                start_date="2025-01-01 00:03:00",
                end_date="2025-01-01 00:05:00",
            )
            self.assertEqual(len(df), 3)
            self.assertEqual(str(df["date"].iloc[0]), "2025-01-01 00:03:00")

    def test_load_binance_mocked(self):
        mocked_df = _base_df()
        with patch("downloader.download", return_value=mocked_df) as mocked:
            df = load_market_data(
                source="binance",
                symbol="BTCUSDT",
                start_date="2025-01-01",
                end_date="2025-01-02",
                timeframe="1m",
                market_type="futures",
            )
            mocked.assert_called_once()
            self.assertEqual(len(df), len(mocked_df))

    def test_load_binance_requires_dates(self):
        with self.assertRaises(ValueError):
            load_market_data(source="binance", symbol="BTCUSDT")


if __name__ == "__main__":
    unittest.main()
