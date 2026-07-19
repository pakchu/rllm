from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from training import build_options_perpetual_demand_relay_support as support
from training import preregister_options_perpetual_demand_relay as prereg


def _premium_minutes(start: str = "2023-06-20T00:00:00Z") -> pd.DataFrame:
    date = pd.date_range(start, periods=60, freq="1min")
    base = np.arange(60, dtype=float) / 10_000.0
    return pd.DataFrame(
        {
            "date": date,
            "source_close_time": date + pd.Timedelta(seconds=59, milliseconds=999),
            "feature_available_time": date + pd.Timedelta(minutes=1, seconds=1),
            "source_valid": True,
            "premium_open": base,
            "premium_high": base + 0.0002,
            "premium_low": base - 0.0001,
            "premium_close": base + 0.00005,
        }
    )


def _joint(periods: int = 7) -> pd.DataFrame:
    signal = pd.date_range("2023-06-20T01:00:00Z", periods=periods, freq="1h")
    return pd.DataFrame(
        {
            "signal_time": signal,
            "feature_available_time": signal + pd.Timedelta(seconds=1),
            "joint_valid": True,
            "log_bvol_dvol_ratio": np.arange(periods, dtype=float),
            "premium_move_bp": np.arange(1, periods + 1, dtype=float),
            "premium_path_range_bp": 10.0,
            "premium_efficiency": np.arange(1, periods + 1, dtype=float) / 10.0,
        }
    )


def _clock_state(decisions: list[str]) -> pd.DataFrame:
    decision = pd.to_datetime(decisions, utc=True)
    rows = len(decision)
    return pd.DataFrame(
        {
            "decision_time": decision,
            "feature_available_time": decision + pd.Timedelta(seconds=1),
            "side": np.resize(np.array([1, -1]), rows),
            "primary_onset": True,
            "no_vol_disagreement_onset": True,
            "no_premium_efficiency_onset": True,
            "dvol_poor_mirror_onset": True,
            "log_bvol_dvol_ratio": -1.0,
            "premium_move_bp": 5.0,
            "premium_path_range_bp": 10.0,
            "premium_efficiency": 0.5,
            "prior_ratio_q20": -0.5,
            "prior_ratio_q80": 0.5,
            "prior_move_abs_q80_bp": 4.0,
            "prior_efficiency_q70": 0.4,
        }
    )


def test_premium_hour_uses_exact_completed_60_minute_path() -> None:
    frame = support.aggregate_premium_hourly(_premium_minutes())
    assert len(frame) == 1
    assert frame.loc[0, "signal_time"] == pd.Timestamp("2023-06-20T01:00:00Z")
    assert frame.loc[0, "feature_available_time"] == pd.Timestamp(
        "2023-06-20T01:00:01Z"
    )
    assert np.isclose(frame.loc[0, "premium_move_bp"], 59.5)
    assert np.isclose(frame.loc[0, "premium_path_range_bp"], 180.0)
    assert np.isclose(frame.loc[0, "premium_efficiency"], 59.5 / 180.0)


def test_premium_hour_fails_closed_when_one_minute_is_invalid() -> None:
    source = _premium_minutes()
    source.loc[17, "source_valid"] = False
    frame = support.aggregate_premium_hourly(source)
    assert not bool(frame.loc[0, "premium_valid"])
    assert frame.loc[
        0,
        ["premium_move_bp", "premium_path_range_bp", "premium_efficiency"],
    ].isna().all()


def test_strict_prior_thresholds_exclude_current_and_future_values() -> None:
    policy = replace(
        prereg.Policy(),
        prior_window_hours=4,
        prior_min_periods_hours=3,
    )
    prefix = _joint(6)
    current_extreme = prefix.copy()
    current_extreme.loc[3, "log_bvol_dvol_ratio"] = -1_000.0
    state = support.derive_state(current_extreme, policy)
    assert state.loc[3, "prior_ratio_q20"] == 0.4
    assert state.loc[3, "prior_move_abs_q80_bp"] == 2.6
    assert state.loc[3, "prior_efficiency_q70"] == 0.24
    extended = pd.concat(
        [current_extreme, _joint(2).assign(
            signal_time=pd.date_range(
                "2023-06-20T07:00:00Z", periods=2, freq="1h"
            )
        )],
        ignore_index=True,
    )
    full = support.derive_state(extended, policy).iloc[: len(current_extreme)]
    pd.testing.assert_frame_equal(
        state[
            [
                "prior_ratio_q20",
                "prior_ratio_q80",
                "prior_move_abs_q80_bp",
                "prior_efficiency_q70",
                "primary_onset",
                "no_vol_disagreement_onset",
                "no_premium_efficiency_onset",
                "dvol_poor_mirror_onset",
            ]
        ],
        full[
            [
                "prior_ratio_q20",
                "prior_ratio_q80",
                "prior_move_abs_q80_bp",
                "prior_efficiency_q70",
                "primary_onset",
                "no_vol_disagreement_onset",
                "no_premium_efficiency_onset",
                "dvol_poor_mirror_onset",
            ]
        ],
    )


def test_thresholds_require_declared_minimum_joint_valid_count() -> None:
    policy = replace(
        prereg.Policy(),
        prior_window_hours=4,
        prior_min_periods_hours=3,
    )
    joint = _joint(5)
    joint.loc[1, "joint_valid"] = False
    state = support.derive_state(joint, policy)
    assert np.isnan(state.loc[3, "prior_ratio_q20"])
    assert state.loc[4, "prior_valid_count"] == 3
    assert np.isfinite(state.loc[4, "prior_ratio_q20"])


def test_global_nonoverlap_reserves_split_crossing_window() -> None:
    state = _clock_state(
        [
            "2023-12-31T12:00:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-02T13:00:00Z",
        ]
    )
    clocks = support.build_clocks(state)
    assert len(clocks) == 1
    assert clocks.loc[0, "decision_time"] == pd.Timestamp("2024-01-02T13:00:00Z")
    assert clocks.loc[0, "split"] == "test"
    assert clocks.loc[0, "entry_time"] == pd.Timestamp("2024-01-02T13:05:00Z")
    assert clocks.loc[0, "exit_time"] == pd.Timestamp("2024-01-03T13:05:00Z")


def test_half_open_window_may_exit_exactly_at_split_end() -> None:
    state = _clock_state(["2023-12-31T00:00:00Z"])
    state["feature_available_time"] = state["decision_time"] - pd.Timedelta(
        seconds=1
    )
    policy = replace(prereg.Policy(), entry_delay_minutes=0, hold_hours=24)
    clocks = support.build_clocks(state, policy=policy)
    assert len(clocks) == 1
    assert clocks.loc[0, "exit_time"] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_novelty_denominator_uses_explicit_common_coverage_only() -> None:
    primary = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-07-01T01:00:00Z",
                "2023-07-02T01:00:00Z",
                "2025-01-01T01:00:00Z",
                "2026-01-01T01:00:00Z",
            ],
            utc=True,
        )
    )
    other = pd.DatetimeIndex(
        pd.to_datetime(
            ["2023-07-01T01:30:00Z", "2023-07-02T01:30:00Z"], utc=True
        )
    )
    row = support._novelty(
        primary,
        other,
        60,
        coverage_start="2023-07-01T00:00:00Z",
        coverage_end="2024-01-01T00:00:00Z",
    )
    assert row["primary_full"] == 4
    assert row["primary_events"] == 2
    assert row["near_primary_events"] == 2
    assert row["near_primary_share"] == 1.0


def test_clock_schema_cannot_retain_outcomes() -> None:
    forbidden = {"price", "return", "pnl", "funding", "btc_close", "btc_open"}
    assert forbidden.isdisjoint(support.CLOCK_COLUMNS)
