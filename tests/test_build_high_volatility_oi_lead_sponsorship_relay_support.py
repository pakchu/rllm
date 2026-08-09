import numpy as np
import pandas as pd

from training import build_high_volatility_oi_lead_sponsorship_relay_support as support


def test_strict_prior_rank_excludes_current() -> None:
    rank = support.strict_prior_midrank(pd.Series([1.0, 2.0, 3.0, 2.0]), lookback=2, minimum=2)
    assert np.isnan(rank.iloc[:2]).all()
    assert rank.iloc[2] == 1.0
    assert rank.iloc[3] == 0.25


def test_finite_correlation_rejects_constant_and_measures_lead() -> None:
    assert np.isnan(support._finite_correlation(np.ones(3), np.arange(3)))
    assert support._finite_correlation(np.arange(4), np.arange(4)) == 1.0


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source_day": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "feature_available_time": pd.date_range("2024-01-02", periods=3, tz="UTC"),
        "source_valid": [True, True, True], "day_return": [0.02, -0.02, 0.02],
        "realized_variation": [0.1, 0.1, 0.1], "lead_correlation": [0.2, 0.2, 0.2],
        "directional_lead_score": [0.2, -0.2, 0.2],
        "contemporaneous_correlation": [0.1, 0.1, 0.1],
        "directional_contemporaneous_score": [0.1, -0.1, 0.1],
        "lead_score_rank": [0.8, 0.8, 0.7], "contemporaneous_score_rank": [0.8, 0.7, 0.8],
        "variation_rank": [0.7, 0.6, 0.7],
    })


def test_primary_and_controls_use_frozen_gates() -> None:
    data = frame(); active, side, _ = support.conditions(data)
    assert active.tolist() == [True, False, False]
    assert side.tolist() == [1, -1, 1]
    assert support.conditions(data, "no_variation_gate")[0].tolist() == [True, True, False]
    assert support.conditions(data, "no_lead_score_gate")[0].tolist() == [True, False, True]
    assert support.conditions(data, "contemporaneous_correlation")[0].tolist() == [True, False, True]
    assert support.conditions(data, "direction_flip")[1].tolist() == [-1, 1, -1]
    assert support.conditions(data, "same_clock_forced_long")[1].tolist() == [1, 1, 1]


def test_clock_is_next_day_0005_for_twelve_hours() -> None:
    clock = support.build_clock(frame().iloc[:1])
    assert clock.iloc[0].entry_time == pd.Timestamp("2024-01-02T00:05:00Z")
    assert clock.iloc[0].exit_time == pd.Timestamp("2024-01-02T12:05:00Z")


def test_queries_are_source_only_and_hash_bound() -> None:
    assert "open_interest_binance" in support.OI_QUERY
    assert "bars_binance" in support.BAR_QUERY
    assert "funding_rates_binance" not in support.OI_QUERY + support.BAR_QUERY
    assert support.PREREG_SHA == "c79bb5e0ebdb0f1ee48f2d307f7648ac9b24782871de25f544438925dc43877c"
