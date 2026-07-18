import unittest

import numpy as np
import pandas as pd

from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    build_live_feature_frame_from_frames,
    latest_live_feature_snapshot,
    live_source_sql,
    resample_market_bars,
)


def _ohlcv(dates, tic, close_base=100.0):
    close = close_base + np.linspace(0.0, 5.0, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "tic": tic,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(10.0, 20.0, len(dates)),
        }
    )


class TestLiveDbFeatures(unittest.TestCase):
    def test_resamples_1m_to_5m_open_time_bars(self):
        dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1min")
        market = _ohlcv(dates, "BTCUSDT", 100.0)
        out = resample_market_bars(market, "5min")
        self.assertEqual(len(out), 2)
        self.assertEqual(str(out.loc[0, "date"]), "2026-01-01 00:00:00")
        self.assertEqual(float(out.loc[0, "open"]), float(market.loc[0, "open"]))
        self.assertEqual(float(out.loc[0, "close"]), float(market.loc[4, "close"]))

    def test_resample_drops_incomplete_latest_decision_bar(self):
        dates = pd.date_range("2026-01-01 00:00:00", periods=7, freq="1min")
        market = _ohlcv(dates, "BTCUSDT", 100.0)
        out = resample_market_bars(market, "5min")
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.loc[0, "date"]), "2026-01-01 00:00:00")

    def test_builds_latest_snapshot_from_db_like_frames(self):
        dates = pd.date_range("2026-01-01 00:00:00", periods=1800, freq="1min")
        btc = _ohlcv(dates, "BTCUSDT", 100.0)
        spot = _ohlcv(dates, "BTCUSDT", 99.5)
        btckrw = _ohlcv(dates, "KRW-BTC", 100.0 * 1300.0 * 1.02)
        usdkrw = _ohlcv(dates, "USDKRW", 1300.0)
        forex_rows = []
        for tic in ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]:
            frame = _ohlcv(dates, tic, 1.0 + 0.01 * len(tic))
            forex_rows.append(frame)
        forex = pd.concat(forex_rows, ignore_index=True)
        premium = pd.DataFrame(
            {
                "date": dates,
                "close_time": dates + pd.Timedelta(seconds=59, milliseconds=999),
                "close": np.linspace(-0.001, 0.001, len(dates)),
            }
        )
        funding = pd.DataFrame(
            {
                "funding_time": [dates[0], dates[480], dates[960], dates[1440]],
                "funding_rate": [0.0001, 0.0002, -0.0001, 0.00005],
                "mark_price": [100.0, 101.0, 102.0, 103.0],
            }
        )
        cfg = LiveDbFeatureConfig(feature_window_size=144, zscore_window=24, volume_window=24, lookback_minutes=1800)

        enriched, features = build_live_feature_frame_from_frames(
            btcusdt_1m=btc,
            btckrw_1m=btckrw,
            usdkrw_1m=usdkrw,
            forex_1m=forex,
            premium_1m=premium,
            funding=funding,
            spot_1m=spot,
            cfg=cfg,
        )
        snapshot = latest_live_feature_snapshot(enriched, features)

        for col in [
            "kimchi_premium",
            "dxy_momentum",
            "premium_index",
            "premium_index_change",
            "funding_rate",
            "rex_144_range_width_pct",
        ]:
            self.assertIn(col, snapshot["feature_snapshot"])
        self.assertEqual(snapshot["data_quality"]["kimchi_available"], 1.0)
        self.assertEqual(snapshot["data_quality"]["premium_available"], 1.0)
        self.assertEqual(snapshot["data_quality"]["funding_available"], 1.0)
        self.assertTrue(np.isfinite(snapshot["feature_snapshot"]["premium_index"]))
        self.assertIn("premium_index_1m_close", enriched.columns)
        self.assertIn("premium_rows", enriched.columns)
        self.assertIn("spot_close", enriched.columns)
        self.assertIn("spot_rows", enriched.columns)
        self.assertEqual(int(enriched.iloc[-1]["premium_rows"]), 5)
        self.assertEqual(int(enriched.iloc[-1]["spot_rows"]), 5)
        self.assertAlmostEqual(float(enriched.iloc[0]["premium_index_1m_close"]), float(premium.loc[4, "close"]))
        self.assertAlmostEqual(float(enriched.iloc[0]["spot_close"]), float(spot.loc[4, "close"]))

    def test_intrabar_rank7_sources_fail_closed_on_incomplete_rows(self):
        dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1min")
        btc = _ohlcv(dates, "BTCUSDT", 100.0)
        spot = _ohlcv(dates.delete([8, 9]), "BTCUSDT", 99.5)
        btckrw = _ohlcv(dates, "KRW-BTC", 100.0 * 1300.0 * 1.02)
        usdkrw = _ohlcv(dates, "USDKRW", 1300.0)
        forex = pd.concat([_ohlcv(dates, tic, 1.0) for tic in ["EURUSD", "USDJPY", "GBPUSD", "USDCAD", "USDSEK", "USDCHF"]])
        premium = pd.DataFrame({"date": dates.delete([7, 8, 9]), "close": np.arange(7, dtype=float) / 1000.0})
        funding = pd.DataFrame({"funding_time": [dates[0]], "funding_rate": [0.0001], "mark_price": [100.0]})
        cfg = LiveDbFeatureConfig(feature_window_size=2, zscore_window=6, volume_window=6, lookback_minutes=10)

        enriched, _ = build_live_feature_frame_from_frames(
            btcusdt_1m=btc,
            btckrw_1m=btckrw,
            usdkrw_1m=usdkrw,
            forex_1m=forex,
            premium_1m=premium,
            funding=funding,
            spot_1m=spot,
            cfg=cfg,
        )

        self.assertEqual(list(enriched["date"]), list(pd.date_range("2026-01-01", periods=2, freq="5min")))
        self.assertEqual(list(enriched["spot_rows"].astype(int)), [5, 3])
        self.assertEqual(list(enriched["premium_rows"].astype(int)), [5, 2])
        self.assertAlmostEqual(float(enriched.loc[1, "spot_close"]), float(spot.iloc[-1]["close"]))
        self.assertAlmostEqual(float(enriched.loc[1, "premium_index_1m_close"]), float(premium.iloc[-1]["close"]))

    def test_live_source_sql_mentions_required_tables(self):
        sql = live_source_sql(LiveDbFeatureConfig())
        self.assertIn("bars_binance", sql["btcusdt_1m"])
        self.assertIn("bars_upbit", sql["btckrw_1m"])
        self.assertIn("bars_polygon", sql["forex_1m"])
        self.assertIn("bars_binance_premium", sql["premium_1m"])
        self.assertIn("bars_binance_spot", sql["spot_1m"])
        self.assertIn("funding_rates_binance", sql["funding"])
        self.assertIn("ts <= :asof", sql["btcusdt_1m"])
        self.assertIn("funding_time <= :asof", sql["funding"])


if __name__ == "__main__":
    unittest.main()
