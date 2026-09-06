import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_signed_path_hysteresis_onset_support as support


def test_loop_area_tracks_early_directional_displacement() -> None:
    positive_returns = np.r_[np.full(120, 0.001), np.full(359, -0.0001)]
    negative_returns = -positive_returns
    positive_close = 100.0 * np.exp(np.r_[0.0, np.cumsum(positive_returns)])
    negative_close = 100.0 * np.exp(np.r_[0.0, np.cumsum(negative_returns)])
    positive_area, positive_variation = support.normalized_loop_area(positive_close)
    negative_area, negative_variation = support.normalized_loop_area(negative_close)
    assert positive_area > 0
    assert negative_area < 0
    assert positive_variation == pytest.approx(negative_variation)


def test_strict_prior_midrank_excludes_current() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    ranked = support.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranked.iloc[0])
    assert np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0
    assert ranked.iloc[3] == 1.0


def test_reservation_keeps_equal_exit_entry() -> None:
    times = pd.date_range("2023-07-01T00:00:00Z", periods=12, freq="1h")
    panel = pd.DataFrame({
        "decision_time": times,
        "source_valid": True,
        "normalized_area": [1.0, -1.0] * 6,
        "absolute_area_rank": 0.9,
        "realized_variation": 1.0,
        "realized_variation_rank": 0.9,
    })
    clock = support.candidate_clock(panel)
    assert list(clock.decision_time) == [times[1], times[9]]
