from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_inventory_conservation_residual_alpha import (
    Config,
    _fit_feature_threshold,
    _fit_linear_residual,
    _load_pre2024,
    _rolling_z,
    _spec_masks,
)


def test_rolling_z_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = pd.Series(np.sin(np.arange(200, dtype=float) / 17.0))
    full = pd.concat([prefix, pd.Series([1_000_000.0] * 20)], ignore_index=True)
    pd.testing.assert_series_equal(
        _rolling_z(full, 48).iloc[: len(prefix)].reset_index(drop=True),
        _rolling_z(prefix, 48).reset_index(drop=True),
    )


def test_linear_residual_coefficients_ignore_rows_outside_fit_mask() -> None:
    n = 20_100
    x = np.column_stack([np.linspace(-1.0, 1.0, n), np.sin(np.arange(n) / 30.0)])
    y = 0.2 + 2.0 * x[:, 0] - 0.5 * x[:, 1]
    fit = np.zeros(n, dtype=bool)
    fit[:20_050] = True
    beta_a, _ = _fit_linear_residual(y, x, fit)
    x_changed = x.copy()
    y_changed = y.copy()
    x_changed[~fit] = 1_000_000.0
    y_changed[~fit] = -1_000_000.0
    beta_b, _ = _fit_linear_residual(y_changed, x_changed, fit)
    np.testing.assert_allclose(beta_a, beta_b, atol=1e-12)
    np.testing.assert_allclose(beta_a, [0.2, 2.0, -0.5], atol=1e-10)


def test_signal_fades_carry_and_flip_is_exact() -> None:
    features = pd.DataFrame(
        {
            "resid_z288": [2.0, 2.0, 0.0, 2.0],
            "carry_z": [-1.5, 1.5, -2.0, 0.2],
            "price_ret_z288": [0.0] * 4,
        }
    )
    spec = {
        "control": "residual_carry_fade",
        "feature": "resid_z288",
        "price_feature": "price_ret_z288",
        "resid_threshold": 1.0,
        "carry_abs": 1.0,
    }
    long_active, short_active = _spec_masks(features, spec)
    np.testing.assert_array_equal(long_active, [True, False, False, False])
    np.testing.assert_array_equal(short_active, [False, True, False, False])
    flip_long, flip_short = _spec_masks(features, spec, flip=True)
    np.testing.assert_array_equal(flip_long, short_active)
    np.testing.assert_array_equal(flip_short, long_active)


def test_fit_feature_threshold_ignores_2023_selection() -> None:
    dates = pd.Series(pd.date_range("2020-10-15", periods=1_000, freq="D"))
    values = np.arange(1_000, dtype=float)
    features = pd.DataFrame({"x": values})
    fit = (dates < pd.Timestamp("2023-01-01")).to_numpy(bool)
    expected = float(np.quantile(values[fit], 0.9))
    assert _fit_feature_threshold(features, dates, "x", 0.9) == expected


def _market(dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }
    )


def _metrics(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "create_time": dates,
            "symbol": "BTCUSDT",
            "sum_open_interest": 1_000.0 + np.arange(len(dates)),
            "sum_open_interest_value": 100_000.0 + np.arange(len(dates)),
            "count_toptrader_long_short_ratio": 1.0,
            "sum_toptrader_long_short_ratio": 1.0,
            "count_long_short_ratio": 1.0,
            "sum_taker_long_short_vol_ratio": 1.0,
        }
    )


def test_loader_physically_excludes_2024_and_delays_oi(tmp_path: Path) -> None:
    dates = pd.date_range("2023-12-31 22:00", periods=48, freq="5min")
    market_path = tmp_path / "market.csv"
    metrics_path = tmp_path / "metrics.csv"
    funding_path = tmp_path / "funding.csv"
    premium_path = tmp_path / "premium.csv"
    _market(dates).to_csv(market_path, index=False)
    _metrics(dates).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "date": ["2023-12-31 16:00:00", "2024-01-01 00:00:00"],
            "funding_rate": [0.0001, 0.0002],
        }
    ).to_csv(funding_path, index=False)
    premium_dates = pd.to_datetime(["2023-12-31 22:59:59", "2023-12-31 23:59:59", "2024-01-01 00:59:59"])
    pd.DataFrame(
        {
            "close_time": (premium_dates.astype("int64") // 1_000_000).astype(np.int64),
            "close": [0.001, 0.002, 0.003],
        }
    ).to_csv(premium_path, index=False)
    cfg = replace(
        Config(),
        market_csv=str(market_path),
        metrics_csv=str(metrics_path),
        funding_csv=str(funding_path),
        premium_csv=str(premium_path),
    )
    loaded, loaded_dates, audit = _load_pre2024(cfg)
    assert loaded_dates.max() < pd.Timestamp("2024-01-01")
    source = pd.to_datetime(loaded["positioning_source_time"], errors="coerce")
    valid = source.notna()
    assert (source.loc[valid].to_numpy() <= (loaded_dates.loc[valid] - pd.Timedelta("5min")).to_numpy()).all()
    assert "file_hashes_full_files_recorded_not_used_for_fit" not in audit
