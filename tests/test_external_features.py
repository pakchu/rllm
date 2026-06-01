from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from preprocessing.external_features import (
    attach_external_features,
    attach_wave_trading_external_features,
    calculate_kimchi_premium,
)
from preprocessing.market_features import build_market_feature_frame
from training.text_analyzer_trader_data import build_analyzer_summary


def _market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01 00:00:00", periods=8, freq="1min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100, 101, 102, 103, 104, 105, 106, 107],
            "volume": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )


class TestExternalFeatures(unittest.TestCase):
    def test_backward_asof_external_join_never_uses_future_rows(self):
        market = _market()
        external = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01 00:00:30", "2024-01-01 00:02:30", "2024-01-01 00:04:30"]),
                "dxy": [100.0, 101.0, 102.0],
            }
        )

        out = attach_external_features(market, external)

        self.assertEqual(out.loc[0, "dxy"], 0.0)  # first external row is in the future
        self.assertEqual(out.loc[1, "dxy"], 100.0)
        self.assertEqual(out.loc[2, "dxy"], 100.0)
        self.assertEqual(out.loc[3, "dxy"], 101.0)
        self.assertIn("dxy_zscore", out.columns)
        self.assertIn("dxy_momentum", out.columns)

    def test_kimchi_premium_uses_btcusdt_btckrw_usdkrw_formula(self):
        market = _market().assign(close=[100.0] * 8)
        btckrw = pd.DataFrame(
            {"date": market["date"], "tic": "KRW-BTC", "close": [132_000.0] * 8}
        )
        usdkrw = pd.DataFrame(
            {"date": market["date"], "tic": "USDKRW", "close": [1_200.0] * 8}
        )

        out = calculate_kimchi_premium(market, btckrw, usdkrw, interval="1min")

        self.assertEqual(round(float(out.loc[0, "kimchi_premium"]), 6), 0.1)
        self.assertEqual(float(out.loc[0, "usdkrw"]), 1200.0)

    def test_wave_trading_cache_loader_attaches_dxy_and_kimchi_features(self):
        market = _market()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            forex_dates = market["date"]
            forex_rows = []
            for tic in ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]:
                for i, ts in enumerate(forex_dates):
                    forex_rows.append({"date": ts, "tic": tic, "close": 1.0 + i * 0.001})
            pd.DataFrame(forex_rows).to_csv(data / "forex.csv.gz", compression="gzip")
            pd.DataFrame({"date": forex_dates, "tic": "USDKRW", "close": [1200.0] * len(market)}).to_csv(
                data / "usdkrw.csv.gz", compression="gzip"
            )
            pd.DataFrame({"date": forex_dates, "tic": "KRW-BTC", "close": market["close"] * 1200.0 * 1.02}).to_csv(
                data / "btckrw.csv.gz", compression="gzip"
            )

            out = attach_wave_trading_external_features(market, wave_trading_root=root)

        for col in ["dxy", "dxy_zscore", "kimchi_premium", "kimchi_premium_zscore", "usdkrw"]:
            self.assertIn(col, out.columns)
        self.assertEqual(round(float(out.loc[len(out) - 1, "kimchi_premium"]), 6), 0.02)

    def test_external_features_reach_analyzer_summary_and_feature_frame(self):
        market = _market()
        external = pd.DataFrame(
            {
                "date": market["date"],
                "dxy": [100, 100, 101, 102, 103, 104, 105, 106],
                "kimchi_premium": [0.0, 0.01, 0.02, 0.04, 0.04, 0.05, 0.05, 0.05],
                "usdkrw": [1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207],
            }
        )
        enriched = attach_external_features(market, external, zscore_window=3, momentum_period=2)
        features = build_market_feature_frame(enriched, window_size=6, zscore_window=6, volume_window=6)
        summary = build_analyzer_summary(enriched, 6, window_size=6, feature_frame=features)

        self.assertIn("Dollar Index", summary["numeric_feature_names"])
        self.assertIn("Kimchi Premium", summary["numeric_feature_names"])
        self.assertIn("Macro Dollar State", summary["symbolic_features"])
        self.assertIn("Korea Premium State", summary["symbolic_features"])


if __name__ == "__main__":
    unittest.main()
