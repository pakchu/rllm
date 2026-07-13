from __future__ import annotations

import numpy as np
import pandas as pd

from training import search_conformal_sr_pressure_alpha as conformal
from training.search_positioning_disagreement_alpha import _simulate_no_stop


def test_prior_zscore_uses_only_previous_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 100.0, -999.0])
    first = conformal.prior_zscore(values, window=3, min_periods=2)
    changed = values.copy()
    changed[4] = 999999.0
    second = conformal.prior_zscore(changed, window=3, min_periods=2)
    assert np.allclose(first[:4], second[:4], equal_nan=True)
    assert np.isclose(first[2], 3.0)


def test_conformal_rank_excludes_current_and_adds_warmup_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    upper, lower, history = conformal.rolling_conformal_pvalues(
        values,
        reference_window=3,
        min_history=2,
    )
    assert history.tolist() == [0, 1, 2, 3]
    assert np.isnan(upper[0]) and np.isnan(upper[1])
    assert np.isclose(upper[2], 1.0 / 3.0)
    assert np.isclose(lower[2], 1.0)
    assert np.isclose(upper[3], 1.0 / 4.0)
    assert np.isclose(lower[3], 1.0)


def test_conformal_rank_prefix_is_future_suffix_independent() -> None:
    prefix = np.arange(1.0, 10.0)
    expected = conformal.rolling_conformal_pvalues(
        prefix,
        reference_window=5,
        min_history=3,
    )
    actual = conformal.rolling_conformal_pvalues(
        np.r_[prefix, [1e9, -1e9]],
        reference_window=5,
        min_history=3,
    )
    for left, right in zip(expected, actual, strict=True):
        assert np.allclose(left, right[: len(prefix)], equal_nan=True)


def test_shiryaev_roberts_event_accumulates_then_resets() -> None:
    residual = np.array([1.0, 1.0, 1.0])
    upper = np.array([0.001, 0.001, 0.001])
    lower = np.ones(3)
    side, evidence = conformal.shiryaev_roberts_events(
        residual,
        upper,
        lower,
        boundary=100.0,
        power=0.5,
    )
    assert side.tolist() == [0, 1, 0]
    assert evidence["log_sr_up"][1] >= np.log(100.0)
    assert evidence["log_sr_up"][2] < np.log(100.0)


def test_hourly_components_delay_current_row_oi_and_use_complete_flow_hour() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=24, freq="5min"))
    market = pd.DataFrame(
        {
            "quote_asset_volume": np.ones(24),
            "taker_buy_quote": np.full(24, 0.75),
            "close": np.linspace(100.0, 101.0, 24),
            "open_interest": np.linspace(1000.0, 1023.0, 24),
        }
    )
    first = conformal.hourly_pressure_components(market, dates)
    changed = market.copy()
    changed.loc[11, "open_interest"] = 1e12
    second = conformal.hourly_pressure_components(changed, dates)
    assert first[0].tolist() == [11, 23]
    assert np.isclose(first[1][0], 0.5)
    assert np.allclose(first[3], second[3], equal_nan=True)


def test_fade_and_release_are_exact_opposites() -> None:
    state = pd.DataFrame(
        {
            "decision": [True, True, False],
            "pressure_side": [1, -1, 1],
        }
    )
    fade_long, fade_short = conformal.policy_masks(state, "fade")
    release_long, release_short = conformal.policy_masks(state, "release")
    np.testing.assert_array_equal(fade_long, release_short)
    np.testing.assert_array_equal(fade_short, release_long)


def test_single_tail_control_uses_current_rank_without_sr_memory() -> None:
    state = pd.DataFrame(
        {
            "pressure_residual": [2.0, -2.0, 2.0],
            "upper_p": [0.005, 1.0, 0.02],
            "lower_p": [1.0, 0.005, 1.0],
        }
    )
    long_active, short_active = conformal.single_tail_masks(state, "fade")
    assert long_active.tolist() == [False, True, False]
    assert short_active.tolist() == [True, False, False]


def test_lag_has_no_wraparound() -> None:
    values = np.array([True, False, True, False])
    assert conformal.lag_boolean(values, 2).tolist() == [False, False, True, False]


def test_support_counts_match_nonoverlap_and_split_containment(monkeypatch) -> None:
    monkeypatch.setitem(conformal.WINDOWS, "sample", ("2023-01-01", "2023-01-01 00:40"))
    dates = pd.Series(pd.date_range("2023-01-01", periods=10, freq="5min"))
    active = np.array([True, True, False, False, True, False, True, False, False, False])
    counts = conformal.support_counts(
        dates,
        active,
        np.zeros(10, dtype=bool),
        window="sample",
        hold_bars=2,
    )
    assert counts == {"raw": 4, "strict_executable": 2}


def test_canonical_execution_is_next_open_nonoverlap_and_strict_mdd() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=7, freq="5min"))
    market = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 110.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 80.0, 110.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    long_active = np.array([True, True, False, False, False, False, False])
    result = _simulate_no_stop(
        market,
        dates,
        long_active,
        np.zeros(7, dtype=bool),
        window="sample",
        hold_bars=1,
        stride_bars=1,
        leverage=0.5,
        fee_rate=0.0,
        slippage_rate=0.0,
        windows={"sample": ("2023-01-01", "2023-01-02")},
    )
    assert result["trades"] == 1
    assert np.isclose(result["return_pct"], 5.0)
    assert np.isclose(result["strict_mdd_pct"], 10.0)


def test_loader_keeps_returned_frame_pre2024_and_complete() -> None:
    _, dates = conformal.load_pre2024()
    assert dates.max() < pd.Timestamp("2024-01-01")
    assert dates.diff().dropna().eq(pd.Timedelta("5min")).all()
