from __future__ import annotations

import pandas as pd

from training import build_intrinsic_volume_flow_handoff_relay_support as s


def _referenced(days: list[str], sides: list[str], directional: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_day": pd.to_datetime(days, utc=True),
            "side": sides,
            "directional_return": directional,
            "reference_ready": [True] * len(days),
            "flow_pass": [True] * len(days),
            "anchor_time": pd.to_datetime(days, utc=True) + pd.Timedelta(hours=8),
            "entry_time": pd.to_datetime(days, utc=True)
            + pd.Timedelta(hours=8, minutes=5),
            "exit_time": pd.to_datetime(days, utc=True)
            + pd.Timedelta(hours=14, minutes=5),
        }
    )


def test_handoff_after_three_anchor_state_is_primary_when_price_lags() -> None:
    frame = _referenced(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        ["LONG", "LONG", "LONG", "SHORT"],
        [0.1, 0.1, 0.1, -0.01],
    )
    result = s.annotate_handoff_state(frame)
    event = result.iloc[-1]
    assert event["prior_state_side"] == "LONG"
    assert event["prior_state_run_length"] == 3
    assert bool(event["handoff"])
    assert bool(event["price_lag"])
    assert bool(event["primary"])


def test_reused_ivlir_primary_label_cannot_leak_into_ivfhr_identity() -> None:
    frame = _referenced(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        ["LONG", "LONG", "LONG", "SHORT"],
        [0.1, 0.1, 0.1, -0.01],
    )
    frame["primary"] = False
    result = s.annotate_handoff_state(frame)
    assert list(result.columns).count("primary") == 1
    assert bool(result.iloc[-1]["primary"])


def test_calendar_gap_resets_state_instead_of_skipping_day() -> None:
    frame = _referenced(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-05"],
        ["LONG", "LONG", "LONG", "SHORT"],
        [0.1, 0.1, 0.1, -0.01],
    )
    event = s.annotate_handoff_state(frame).iloc[-1]
    assert not bool(event["calendar_consecutive"])
    assert event["prior_state_run_length"] == 0
    assert not bool(event["handoff"])
    assert not bool(event["primary"])


def test_price_confirmation_rejects_follow_flow_primary() -> None:
    frame = _referenced(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        ["SHORT", "SHORT", "SHORT", "LONG"],
        [0.1, 0.1, 0.1, 0.01],
    )
    event = s.annotate_handoff_state(frame).iloc[-1]
    assert bool(event["handoff"])
    assert not bool(event["price_lag"])
    assert not bool(event["primary"])


def test_clock_uses_decision_at_entry_and_contains_no_outcome_fields() -> None:
    frame = _referenced(
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        ["LONG", "LONG", "LONG", "SHORT"],
        [0.1, 0.1, 0.1, -0.01],
    )
    features = s.annotate_handoff_state(frame)
    clocks = s.build_clocks(features, features)
    primary = clocks["primary"]
    assert list(primary.columns) == s.CLOCK_COLUMNS
    assert primary["decision_time"].equals(primary["entry_time"])
    assert not {"open", "close", "return", "pnl"}.intersection(primary.columns)


def test_longest_run_and_gap_statistics_are_chronological() -> None:
    clock = pd.DataFrame(
        {
            "clock_name": ["primary"] * 3,
            "source_day": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-13"], utc=True
            ),
            "decision_time": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-13"], utc=True
            ),
            "entry_time": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-13"], utc=True
            ),
            "exit_time": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-13"], utc=True
            )
            + pd.Timedelta(hours=6),
            "side": ["LONG", "LONG", "SHORT"],
        }
    )
    stats = s.clock_stats(clock)
    assert stats["maximum_same_side_run"] == 2
    assert stats["maximum_calendar_gap_days"] == 10.0
