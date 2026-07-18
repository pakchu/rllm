import numpy as np
import pandas as pd
import pytest

from training.search_liveparity_state_feature_interactions import (
    _clear_bocpd_runtime_cache,
    _runtime_bocpd_output,
    _runtime_kalman_frame,
    completed_hourly_features,
    hourly_state_features,
    immutable_anchors,
    state_bank,
    state_bank_from_hourly,
)


def test_completed_hourly_features_excludes_current_boundary_bar():
    dates = pd.date_range("2022-01-01 00:00:00", periods=25, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(100.0, 125.0),
            "high": np.arange(101.0, 126.0),
            "low": np.arange(99.0, 124.0),
            "close": np.arange(1000.0, 1025.0),
            "quote_asset_volume": np.ones(25) * 10.0,
            "taker_buy_quote": np.ones(25) * 4.0,
        }
    )

    hourly, _ = completed_hourly_features(market)

    assert hourly.index[0] == pd.Timestamp("2022-01-01 01:00:00")
    assert hourly.iloc[0]["open"] == pytest.approx(100.0)
    assert hourly.iloc[0]["close"] == pytest.approx(1011.0)
    assert 1024.0 not in hourly["close"].to_numpy(), "the 01:00 boundary bar belongs to the next hour"
    assert hourly.index.tolist() == [pd.Timestamp("2022-01-01 01:00:00"), pd.Timestamp("2022-01-01 02:00:00")]
    assert hourly.iloc[1]["open"] == pytest.approx(112.0)
    assert hourly.iloc[1]["close"] == pytest.approx(1023.0)


def test_immutable_anchors_respect_cooldown():
    active = np.zeros(400, dtype=bool)
    active[[0, 1, 143, 144, 145, 287, 288]] = True

    anchors = immutable_anchors(active, cooldown=144)

    assert np.flatnonzero(anchors).tolist() == [0, 144, 288]


def test_state_bank_hourly_warm_start_path_matches_batch_builder():
    rows = 35 * 24 * 12
    dates = pd.date_range("2020-07-01", periods=rows, freq="5min")
    phase = np.linspace(0.0, 18.0, rows)
    close = 10_000.0 * np.exp(0.00002 * np.arange(rows) + 0.002 * np.sin(phase))
    quote = 1_000_000.0 * (1.0 + 0.1 * np.cos(phase))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_asset_volume": quote,
            "taker_buy_quote": quote * (0.5 + 0.05 * np.sin(phase)),
        }
    )
    hourly, hourly_features = completed_hourly_features(market)

    batch = state_bank(market, market["date"])
    warm = state_bank_from_hourly(hourly, hourly_features, market["date"])

    for key in ("kalman", "bocpd", "semimarkov"):
        np.testing.assert_array_equal(warm[key], batch[key])


def test_incremental_state_filters_match_clean_full_rebuild():
    rows = 45 * 24
    index = pd.date_range("2020-07-01 01:00", periods=rows, freq="1h")
    phase = np.linspace(0.0, 20.0, rows)
    close = 10_000.0 * np.exp(0.0002 * np.arange(rows) + 0.003 * np.sin(phase))
    hourly = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote": 1_000_000.0 * (1.0 + 0.1 * np.cos(phase)),
            "buy": 500_000.0 * (1.0 + 0.1 * np.sin(phase)),
        },
        index=index,
    )
    features = hourly_state_features(hourly)
    fit = np.zeros(rows, dtype=bool)
    fit[: 30 * 24] = True

    _clear_bocpd_runtime_cache()
    _runtime_kalman_frame(hourly.iloc[:-1], fit[:-1])
    resumed_kalman = _runtime_kalman_frame(hourly, fit).copy()
    _runtime_bocpd_output(
        features.iloc[:-1],
        fit[:-1],
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=336,
    )
    resumed_bocpd = _runtime_bocpd_output(
        features,
        fit,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=336,
    ).copy()

    _clear_bocpd_runtime_cache()
    rebuilt_kalman = _runtime_kalman_frame(hourly, fit)
    rebuilt_bocpd = _runtime_bocpd_output(
        features,
        fit,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=336,
    )

    pd.testing.assert_frame_equal(resumed_kalman, rebuilt_kalman, check_exact=True)
    pd.testing.assert_frame_equal(resumed_bocpd, rebuilt_bocpd, check_exact=True)
