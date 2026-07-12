import numpy as np
import pandas as pd
import torch

from training.search_chronos2_zero_shot_alpha import (
    anchor_hour_indices,
    causal_hourly_frame,
    forecast_score_streams,
)


def test_hourly_aggregation_excludes_candle_opened_at_hour_boundary():
    dates = pd.date_range("2024-01-01", periods=13, freq="5min")
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.arange(13) + 100.0,
            "high": np.arange(13) + 101.0,
            "low": np.arange(13) + 99.0,
            "close": np.r_[np.arange(12) + 100.5, 999.0],
            "quote_asset_volume": np.ones(13),
            "taker_buy_quote": np.full(13, 0.5),
        }
    )

    hourly = causal_hourly_frame(market)

    assert hourly.loc[pd.Timestamp("2024-01-01 01:00:00"), "close"] == 111.5
    assert hourly.loc[pd.Timestamp("2024-01-01 02:00:00"), "close"] == 999.0


def test_anchor_maps_signal_close_to_completed_hour():
    dates = pd.Series(pd.date_range("2024-01-01", periods=144, freq="5min"))
    hourly_index = pd.date_range("2024-01-01 01:00", periods=12, freq="1h")

    indices = anchor_hour_indices(dates, np.array([143]), hourly_index)

    assert hourly_index[indices[0]] == pd.Timestamp("2024-01-01 12:00:00")


def test_forecast_scores_use_entry_price_and_expected_quantiles():
    quantiles = [0.1, 0.5, 0.9]
    values = torch.zeros((1, 3, 48), dtype=torch.float32)
    values[0, 0, -1] = 9.9
    values[0, 1, :] = torch.linspace(10.0, 10.2, 48)
    values[0, 2, -1] = 10.5

    streams = forecast_score_streams(
        [values],
        np.array([1]),
        np.array([0.0, 10.0]),
        quantiles,
        total_anchors=3,
    )

    assert np.isclose(streams["median_terminal"][1], 0.2)
    assert np.isclose(streams["central_terminal"][1], 0.2)
    assert np.isclose(streams["quantile_mean_terminal"][1], 0.2)
    assert streams["terminal_interval_snr"][1] > 0.0
    assert np.isnan(streams["median_terminal"][[0, 2]]).all()
