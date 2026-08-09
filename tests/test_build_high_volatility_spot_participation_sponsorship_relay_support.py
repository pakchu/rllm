import numpy as np
import pandas as pd

from training import build_high_volatility_spot_participation_sponsorship_relay_support as support


def test_strict_prior_rank_excludes_current() -> None:
    rank = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 2.0]), lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25


def test_primary_and_controls_use_frozen_gates() -> None:
    frame = pd.DataFrame({
        "source_day": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "source_valid": [True, True, True],
        "spot_return": [0.02, -0.02, 0.02],
        "perp_return": [0.01, 0.01, 0.01],
        "spot_quote_volume": [1.0, 1.0, 1.0], "perp_quote_volume": [2.0, 2.0, 2.0],
        "spot_participation_share": [1 / 3, 1 / 3, 1 / 3],
        "spot_participation_rank": [0.8, 0.8, 0.7],
        "btc_realized_variation": [0.1, 0.1, 0.1],
        "btc_variation_rank": [0.7, 0.6, 0.7],
    })
    active, side, _ = support.conditions(frame)
    assert active.tolist() == [True, False, False]
    assert side.tolist() == [1, -1, 1]
    assert support.conditions(frame, "no_btc_variation_gate")[0].tolist() == [True, False, False]
    assert support.conditions(frame, "no_spot_participation_gate")[0].tolist() == [True, False, True]
    assert support.conditions(frame, "no_direction_agreement")[0].tolist() == [True, False, False]
    assert support.conditions(frame, "direction_flip")[1].tolist() == [-1, 1, -1]
    assert support.conditions(frame, "same_clock_forced_long")[1].tolist() == [1, 1, 1]


def test_clock_uses_next_day_0005_and_twelve_hours() -> None:
    frame = pd.DataFrame({
        "source_day": [pd.Timestamp("2024-01-01T00:00:00Z")], "source_valid": [True],
        "spot_return": [-0.02], "perp_return": [-0.01],
        "spot_quote_volume": [1.0], "perp_quote_volume": [2.0],
        "spot_participation_share": [1 / 3], "spot_participation_rank": [0.8],
        "btc_realized_variation": [0.1], "btc_variation_rank": [0.7],
    })
    clock = support.build_clock(frame)
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-02T00:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-02T12:05:00Z")
    assert clock.iloc[0].side == -1


def test_builder_queries_are_source_only() -> None:
    assert "bars_binance_spot" in support.SPOT_QUERY
    assert "bars_binance" in support.PERP_QUERY
    assert "funding_rates_binance" not in support.SPOT_QUERY + support.PERP_QUERY
    assert support.PREREG_SHA == "f6adc656d9de242175df7e6a42a88e1d7012058eababa9e1b0e945f83e439460"
