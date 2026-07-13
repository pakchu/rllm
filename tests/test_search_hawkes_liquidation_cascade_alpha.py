from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_hawkes_liquidation_cascade_alpha import (
    HawkesCascadeConfig,
    build_hawkes_features,
    build_signals,
    fit_abs_threshold,
    load_market,
)


def _market(n: int, start: str = "2023-12-31 22:00") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="5min")
    returns = 0.0001 * np.sin(np.arange(n, dtype=float) / 11.0)
    if n > 600:
        returns[600] = 0.08
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )


def _metrics(n: int, start: str = "2023-12-31 22:00") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame(
        {
            "create_time": dates,
            "symbol": "BTCUSDT",
            "sum_open_interest": 1_000.0 + np.arange(n),
            "sum_open_interest_value": 100_000.0 + np.arange(n),
            "count_toptrader_long_short_ratio": 1.0,
            "sum_toptrader_long_short_ratio": 1.0,
            "count_long_short_ratio": 1.0,
            "sum_taker_long_short_vol_ratio": 1.0,
        }
    )


def test_hawkes_feature_prefix_does_not_depend_on_future_suffix() -> None:
    n = 800
    market = _market(n, start="2021-01-01")
    market["sum_open_interest"] = 1_000.0 + np.arange(n)
    prefix = 700
    full = build_hawkes_features(market, jump_z_threshold=2.0, intensity_half_life_bars=12)
    short = build_hawkes_features(market.iloc[:prefix], jump_z_threshold=2.0, intensity_half_life_bars=12)
    pd.testing.assert_frame_equal(full.iloc[:prefix].reset_index(drop=True), short.reset_index(drop=True))
    assert full.loc[600, "hawkes_imbalance"] > 0.0


def test_signal_requires_oi_build_and_is_direction_deterministic() -> None:
    features = pd.DataFrame(
        {
            "hawkes_imbalance": [0.8, -0.9, 0.7, -0.6],
            "oi_change_288": [1.0, 1.0, -1.0, -1.0],
            "decision_event": [True, True, True, True],
        }
    )
    long_signal, short_signal = build_signals(features, threshold=0.5, mode="build_follow")
    np.testing.assert_array_equal(long_signal, [True, False, False, False])
    np.testing.assert_array_equal(short_signal, [False, True, False, False])
    flip_long, flip_short = build_signals(
        features,
        threshold=0.5,
        mode="build_follow",
        direction_flip=True,
    )
    np.testing.assert_array_equal(flip_long, short_signal)
    np.testing.assert_array_equal(flip_short, long_signal)


def test_threshold_uses_fit_rows_only() -> None:
    values = np.r_[np.arange(10_000, dtype=float), 1_000_000.0]
    mask = np.r_[np.ones(10_000, dtype=bool), False]
    assert fit_abs_threshold(values, mask, 0.5) == 4_999.5


def test_physical_loader_excludes_2024_and_delays_metrics(tmp_path: Path) -> None:
    market = _market(48)
    metrics = _metrics(48)
    market_path = tmp_path / "market.csv"
    metrics_path = tmp_path / "metrics.csv"
    market.to_csv(market_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    cfg = replace(
        HawkesCascadeConfig(),
        input_csv=str(market_path),
        metrics_csv=str(metrics_path),
        selection_end="2024-01-01",
    )
    loaded = load_market(cfg, cutoff=cfg.selection_end)
    assert pd.to_datetime(loaded["date"]).max() < pd.Timestamp("2024-01-01")
    source = pd.to_datetime(loaded["positioning_source_time"], errors="coerce")
    assert source.max() < pd.Timestamp("2024-01-01")
    valid = source.notna()
    assert (source[valid] <= pd.to_datetime(loaded.loc[valid, "date"]) - pd.Timedelta("5min")).all()
