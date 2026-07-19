from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import build_premium_compression_breakout_relay_support as support
from training import preregister_premium_compression_breakout_relay as prereg


def _minute_source(rows: int = 60) -> pd.DataFrame:
    date = pd.date_range("2020-01-01", periods=rows, freq="1min", tz="UTC")
    values = np.arange(rows, dtype=float) / 10_000.0
    return pd.DataFrame(
        {
            "date": date,
            "source_close_time": date + pd.Timedelta(seconds=59, milliseconds=999),
            "feature_available_time": date + pd.Timedelta(minutes=1, seconds=1),
            "source_valid": True,
            "premium_open": values,
            "premium_high": values + 0.0001,
            "premium_low": values - 0.0001,
            "premium_close": values + 0.00005,
        }
    )


def test_aggregate_5m_uses_exact_completed_minute_rows() -> None:
    source = _minute_source()
    bars = support.aggregate_5m(source)
    assert len(bars) == 12
    assert bars.loc[0, "open"] == 0.0
    assert bars.loc[0, "high"] == 5.0
    assert bars.loc[0, "low"] == -1.0
    assert bars.loc[0, "close"] == 4.5
    assert bars.loc[0, "decision_time"] == pd.Timestamp("2020-01-01T00:05:00Z")


def test_aggregate_5m_rejects_noncontiguous_minute_grid() -> None:
    source = _minute_source()
    source.loc[10, "date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="not contiguous"):
        support.aggregate_5m(source)


def test_prior_threshold_paths_do_not_change_when_future_is_appended() -> None:
    rows = 80
    times = pd.date_range("2021-01-01", periods=rows, freq="5min", tz="UTC")
    close = np.sin(np.arange(rows) / 3.0) * 5.0
    bars = pd.DataFrame(
        {
            "bar_open_time": times,
            "feature_available_time": times + pd.Timedelta(minutes=5, seconds=1),
            "valid": True,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "decision_time": times + pd.Timedelta(minutes=5),
        }
    )
    policy = replace(
        prereg.Policy(),
        context_bars_5m=4,
        trigger_bars_5m=2,
        prior_window_bars_5m=20,
        prior_min_periods_5m=15,
        prior_nonoverlap_shift_bars_5m=6,
    )
    prefix = support.derive_state(bars.iloc[:60].copy(), policy)
    full = support.derive_state(bars, policy).iloc[:60]
    columns = [
        "context_range",
        "trigger_move",
        "trigger_efficiency",
        "terminal_location",
        "outside_distance",
        "primary_active",
    ]
    pd.testing.assert_frame_equal(
        prefix.loc[:, columns].reset_index(drop=True),
        full.loc[:, columns].reset_index(drop=True),
    )


def test_build_clocks_delays_entry_and_reserves_nonoverlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2020-03-01T00:00:00Z")
    state = pd.DataFrame(
        {
            "context_start_time": [start, start + pd.Timedelta(minutes=5)],
            "decision_time": [start, start + pd.Timedelta(minutes=5)],
            "feature_available_time": [
                start - pd.Timedelta(seconds=1),
                start + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1),
            ],
            "side": [1, -1],
            "context_range": [1.0, 1.0],
            "trigger_move": [2.0, -2.0],
            "trigger_efficiency": [0.9, 0.9],
            "terminal_location": [0.9, -0.9],
            "outside_distance": [1.0, 1.0],
            "primary_active": [True, True],
            "no_compression_active": [True, True],
            "no_terminal_pin_active": [True, True],
            "no_outside_cage_active": [True, True],
        }
    )
    monkeypatch.setattr(
        support,
        "SPLITS",
        {"train": (start, start + pd.Timedelta(days=1))},
    )
    clocks = support.build_clocks(state)
    assert len(clocks) == 1
    assert clocks.loc[0, "entry_time"] == start + pd.Timedelta(minutes=10)
    assert clocks.loc[0, "exit_time"] == start + pd.Timedelta(minutes=70)
    assert clocks.loc[0, "side"] == 1


def test_novelty_counts_exact_and_near_matches() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"])
    )
    other = pd.DatetimeIndex(
        pd.to_datetime(["2023-01-01T00:00:00Z", "2023-01-02T00:30:00Z"])
    )
    row = support._novelty(primary, other, near_minutes=60)
    assert row["exact_intersection"] == 1
    assert row["near_primary_events"] == 2
    assert row["near_primary_share"] == 1.0


def test_novelty_denominator_is_limited_to_explicit_comparator_coverage() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-01-01T00:00:00Z",
                "2023-01-01T02:00:00Z",
                "2025-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ]
        )
    )
    other = pd.DatetimeIndex(
        pd.to_datetime(
            ["2023-01-01T00:30:00Z", "2023-01-01T02:30:00Z"]
        )
    )
    row = support._novelty(
        primary,
        other,
        near_minutes=60,
        coverage_start="2023-01-01T00:00:00Z",
        coverage_end="2024-01-01T00:00:00Z",
    )
    assert row["primary_full"] == 4
    assert row["primary_events"] == 2
    assert row["near_primary_events"] == 2
    assert row["near_primary_share"] == 1.0
