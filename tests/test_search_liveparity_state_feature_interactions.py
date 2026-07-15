import numpy as np
import pandas as pd
import pytest

from training.search_liveparity_state_feature_interactions import completed_hourly_features, immutable_anchors


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
