import pandas as pd
import unittest
import numpy as np

from execution.portfolio_live import (
    LiveFeatureFrameCache,
    LiveExternalFrameCache,
    LiveOiFrameCache,
    LiveSourceFrameCache,
    _apply_portfolio_selector_overlay,
    _build_external_from_frames,
    _build_portfolio_feature_frame,
)
from preprocessing.external_features import attach_external_features, DXY_WEIGHTS
from preprocessing.live_db_features import LiveDbFeatureConfig


def _frames():
    enriched = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3, freq="5min")})
    features = pd.DataFrame(
        {
            "htf_1d_return_1": [0.0, 0.0, 0.02],
            "htf_1d_range_pos": [0.5, 0.5, 0.9],
            "dxy_momentum": [0.0, 0.0, 0.0],
            "kimchi_premium_zscore": [0.0, 0.0, 1.0],
        }
    )
    return enriched, features


class TestPortfolioLiveSelector(unittest.TestCase):
    def test_portfolio_selector_blocks_bad_context_new_entries_only(self):
        enriched, features = _frames()
        sleeves = [
            {"name": "bear_rex_short", "active": True, "reasons": []},
            {"name": "already_inactive", "active": False, "reasons": []},
        ]
        overlay = {
            "name": "portfolio_bull_bear_oi_rex_llm_selector_overlay",
            "output_space": ["ALLOW", "BLOCK_RISK"],
            "symbolic_proxy": {
                "context_keys": ["trend_1d", "range_pos_1d", "dxy", "kimchi"],
                "blocked_contexts": [
                    {"context_id": "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot"}
                ],
            },
        }

        record = _apply_portfolio_selector_overlay(sleeves, overlay=overlay, enriched=enriched, features=features)

        self.assertEqual(record["action"], "BLOCK_RISK")
        self.assertFalse(record["allowed"])
        self.assertEqual(record["context_id"], "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot")
        self.assertFalse(sleeves[0]["active"])
        self.assertIn("portfolio_selector_context=", sleeves[0]["reasons"][-1])
        self.assertFalse(sleeves[1]["active"])

    def test_portfolio_selector_allows_unblocked_context(self):
        enriched, features = _frames()
        features.loc[len(features) - 1, "kimchi_premium_zscore"] = 0.0
        sleeves = [{"name": "bear_rex_short", "active": True, "reasons": []}]
        overlay = {
            "name": "portfolio_bull_bear_oi_rex_llm_selector_overlay",
            "symbolic_proxy": {
                "context_keys": ["trend_1d", "range_pos_1d", "dxy", "kimchi"],
                "blocked_contexts": [
                    {"context_id": "trend_1d=up|range_pos_1d=high|dxy=flat|kimchi=hot"}
                ],
            },
        }

        record = _apply_portfolio_selector_overlay(sleeves, overlay=overlay, enriched=enriched, features=features)

        self.assertEqual(record["action"], "ALLOW")
        self.assertTrue(record["allowed"])
        self.assertTrue(sleeves[0]["active"])


class TestLiveSourceFrameCache(unittest.TestCase):
    def test_merge_trims_and_replaces_overlap_rows(self):
        cache = LiveSourceFrameCache(
            frames={
                "btcusdt_1m": pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:02"]),
                        "close": [1.0, 2.0, 3.0],
                        "tic": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
                    }
                )
            }
        )
        merged = cache._merge_and_trim(
            {
                "btcusdt_1m": pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2026-01-01 00:02", "2026-01-01 00:03"], utc=True),
                        "close": [30.0, 4.0],
                        "tic": ["BTCUSDT", "BTCUSDT"],
                    }
                )
            },
            lookback_start=pd.Timestamp("2026-01-01 00:01", tz="UTC"),
            asof=pd.Timestamp("2026-01-01 00:03", tz="UTC"),
        )

        out = merged["btcusdt_1m"]
        self.assertEqual(out["date"].astype(str).tolist(), ["2026-01-01 00:01:00", "2026-01-01 00:02:00", "2026-01-01 00:03:00"])
        self.assertEqual(out["close"].tolist(), [2.0, 30.0, 4.0])


class TestLiveOiFrameCache(unittest.TestCase):
    def test_merge_replaces_overlap_and_trims(self):
        cache = LiveOiFrameCache(
            frame=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]),
                    "open_interest": [1.0, 2.0, 3.0],
                }
            )
        )
        out = cache._merge(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-01-01 00:10", "2026-01-01 00:15"], utc=True),
                    "open_interest": [30.0, 4.0],
                }
            ),
            start=pd.Timestamp("2026-01-01 00:05", tz="UTC"),
            asof=pd.Timestamp("2026-01-01 00:15", tz="UTC"),
        )

        self.assertEqual(out["date"].astype(str).tolist(), ["2026-01-01 00:05:00", "2026-01-01 00:10:00", "2026-01-01 00:15:00"])
        self.assertEqual(out["open_interest"].tolist(), [2.0, 30.0, 4.0])


class TestLiveFeatureFrameCache(unittest.TestCase):
    def test_tail_refresh_matches_full_compute_for_latest_rows(self):
        n = 8_900
        rng = np.random.default_rng(7)
        close = 60_000 + np.cumsum(rng.normal(0, 8, n))
        open_ = close + rng.normal(0, 2, n)
        high = np.maximum(open_, close) + rng.uniform(1, 10, n)
        low = np.minimum(open_, close) - rng.uniform(1, 10, n)
        volume = rng.uniform(10, 100, n)
        enriched = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=n, freq="5min"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_asset_volume": volume * close,
                "number_of_trades": rng.integers(100, 500, n),
                "taker_buy_base": volume * rng.uniform(0.35, 0.65, n),
                "open_interest": 1_000_000 + np.cumsum(rng.normal(0, 100, n)),
                "premium_index": rng.normal(0, 0.0001, n),
                "funding_rate": rng.normal(0, 0.00005, n),
                "premium_index_zscore": rng.normal(0, 1, n),
                "premium_index_change": rng.normal(0, 0.0001, n),
                "funding_available": 1.0,
                "premium_available": 1.0,
                "binance_aux_any_available": 1.0,
                "dxy": 100 + np.cumsum(rng.normal(0, 0.01, n)),
                "dxy_zscore": rng.normal(0, 1, n),
                "dxy_momentum": rng.normal(0, 0.01, n),
                "kimchi_premium": rng.normal(0, 0.001, n),
                "kimchi_premium_zscore": rng.normal(0, 1, n),
                "kimchi_premium_change": rng.normal(0, 0.001, n),
                "usdkrw_zscore": rng.normal(0, 1, n),
                "usdkrw_momentum": rng.normal(0, 0.01, n),
                "dxy_available": 1.0,
                "kimchi_available": 1.0,
                "usdkrw_available": 1.0,
                "external_any_available": 1.0,
            }
        )
        cfg = LiveDbFeatureConfig()
        cache = LiveFeatureFrameCache(output_bars=64)
        _ = cache.refresh(enriched.iloc[:-2].copy(), cfg)
        cached = cache.refresh(enriched.copy(), cfg)
        full = _build_portfolio_feature_frame(enriched, cfg)

        cols = sorted(set(full.columns).intersection(cached.columns))
        pd.testing.assert_frame_equal(
            cached.loc[n - 64 :, cols].reset_index(drop=True),
            full.loc[n - 64 :, cols].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-8,
        )

        no_activity_cache = LiveFeatureFrameCache(output_bars=64)
        _ = no_activity_cache.refresh(enriched.iloc[:-2].copy(), cfg, include_activity_flow=False)
        no_activity = no_activity_cache.refresh(enriched.copy(), cfg, include_activity_flow=False)
        no_activity_full = _build_portfolio_feature_frame(enriched, cfg, include_activity_flow=False)
        self.assertNotIn("activity_flow_htf", no_activity.columns)
        cols = sorted(set(no_activity_full.columns).intersection(no_activity.columns))
        pd.testing.assert_frame_equal(
            no_activity.loc[n - 64 :, cols].reset_index(drop=True),
            no_activity_full.loc[n - 64 :, cols].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-8,
        )


class TestLiveExternalFrameCache(unittest.TestCase):
    def test_tail_external_matches_full_for_decision_features(self):
        n = 720
        rng = np.random.default_rng(11)
        dates_5m = pd.date_range("2026-01-01", periods=n, freq="5min")
        btc_close = 60_000 + np.cumsum(rng.normal(0, 15, n))
        market = pd.DataFrame(
            {
                "date": dates_5m,
                "open": btc_close,
                "high": btc_close + 20,
                "low": btc_close - 20,
                "close": btc_close,
                "volume": rng.uniform(10, 100, n),
                "open_interest": 1_000_000 + np.cumsum(rng.normal(0, 100, n)),
            }
        )
        dates_1m = pd.date_range("2026-01-01", periods=n * 5, freq="1min")
        btckrw = pd.DataFrame(
            {
                "date": dates_1m,
                "open": 90_000_000.0,
                "high": 90_000_100.0,
                "low": 89_999_900.0,
                "close": 90_000_000 + np.cumsum(rng.normal(0, 5000, len(dates_1m))),
                "volume": 1.0,
                "tic": "KRW-BTC",
            }
        )
        usdkrw = pd.DataFrame(
            {
                "date": dates_1m,
                "open": 1400.0,
                "high": 1401.0,
                "low": 1399.0,
                "close": 1400 + np.cumsum(rng.normal(0, 0.02, len(dates_1m))),
                "volume": 1.0,
                "tic": "USDKRW",
            }
        )
        forex_parts = []
        for i, tic in enumerate(DXY_WEIGHTS):
            base = 1.0 + 0.1 * i
            forex_parts.append(
                pd.DataFrame(
                    {
                        "date": dates_1m,
                        "open": base,
                        "high": base + 0.001,
                        "low": base - 0.001,
                        "close": base + np.cumsum(rng.normal(0, 0.00001, len(dates_1m))),
                        "volume": 1.0,
                        "tic": tic,
                    }
                )
            )
        frames = {
            "btckrw_1m": btckrw,
            "usdkrw_1m": usdkrw,
            "forex_1m": pd.concat(forex_parts, ignore_index=True),
        }
        cfg = LiveDbFeatureConfig()
        cache = LiveExternalFrameCache(output_bars=64)
        _ = cache.refresh(market=market.iloc[:-2].copy(), frames=frames, cfg=cfg)
        cached = cache.refresh(market=market.copy(), frames=frames, cfg=cfg)
        full_external = _build_external_from_frames(market=market, frames=frames, cfg=cfg)
        full = attach_external_features(
            market,
            full_external,
            tolerance=cfg.external_tolerance,
            zscore_window=cfg.zscore_window,
            momentum_period=cfg.zscore_window,
        )
        cols = [
            "dxy_zscore",
            "dxy_momentum",
            "kimchi_premium_zscore",
            "kimchi_premium_change",
            "usdkrw_zscore",
            "usdkrw_momentum",
            "btckrw_zscore",
            "btckrw_momentum",
        ]
        pd.testing.assert_frame_equal(
            cached.loc[n - 64 :, cols].reset_index(drop=True),
            full.loc[n - 64 :, cols].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
