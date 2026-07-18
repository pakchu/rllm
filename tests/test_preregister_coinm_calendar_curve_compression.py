from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_coinm_calendar_curve_compression as module


def test_support_source_columns_exclude_post_entry_ohlc() -> None:
    assert "front_high" not in module.SOURCE_COLUMNS
    assert "front_low" not in module.SOURCE_COLUMNS
    assert "next_high" not in module.SOURCE_COLUMNS
    assert "next_low" not in module.SOURCE_COLUMNS
    assert "front_open" not in module.SOURCE_COLUMNS
    assert "next_open" not in module.SOURCE_COLUMNS


def test_parse_feature_valid_rejects_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="unexpected feature_valid"):
        module.parse_feature_valid(pd.Series(["true", "yes"]))


def test_causal_rolling_state_is_prior_only_and_pair_reset() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 10.0, 20.0, 30.0])
    pairs = pd.Series(["a", "a", "a", "b", "b", "b"])
    state = module.causal_rolling_state(values, pairs, window=2, min_periods=2)
    assert state.loc[2, "center"] == 1.5
    assert np.isnan(state.loc[3, "center"])
    assert state.loc[5, "center"] == 15.0


def _clock_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2023-01-01", periods=6, freq="5min")
    source = pd.DataFrame(
        {
            "signal_bar_open_utc": times,
            "feature_available_time_utc": times + pd.Timedelta("5min"),
            "trade_earliest_time_utc": times + pd.Timedelta("5min"),
            "front_symbol": ["F"] * 6,
            "next_symbol": ["N"] * 6,
            "front_hours_to_delivery": [1000.0] * 6,
            "next_hours_to_delivery": [2000.0] * 6,
        }
    )
    state = pd.DataFrame(
        {
            "pair": ["F|N"] * 6,
            "curve": [0.0, 0.001, 0.0030, 0.0026, 0.0025, 0.0024],
            "center": [0.0] * 6,
            "z": [0.0, 1.0, 3.0, 2.6, 2.5, 2.4],
            "source_valid": [True] * 6,
            "two_bar_liquid": [False, False, True, True, True, True],
        }
    )
    return source, state


def test_candidate_clock_anchors_on_completed_confirmation_bar() -> None:
    source, state = _clock_fixture()
    active, side = module.candidate_clock(source, state)
    assert np.flatnonzero(active).tolist() == [3]
    assert side[3] == -1


def test_candidate_clock_requires_shock_crossing_and_residual_sign() -> None:
    source, state = _clock_fixture()
    state.loc[1, "z"] = 2.1
    active, _ = module.candidate_clock(source, state)
    assert not active.any()

    source, state = _clock_fixture()
    state.loc[3, "curve"] = -0.0026
    active, _ = module.candidate_clock(source, state)
    assert not active.any()


def test_schedule_enters_after_confirmation_and_uses_opposite_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, state = _clock_fixture()
    extra = module.CANDIDATE.hold_bars + 2
    tail_times = pd.date_range(source.signal_bar_open_utc.iloc[-1] + pd.Timedelta("5min"), periods=extra, freq="5min")
    tail_source = pd.DataFrame(
        {
            "signal_bar_open_utc": tail_times,
            "feature_available_time_utc": tail_times + pd.Timedelta("5min"),
            "trade_earliest_time_utc": tail_times + pd.Timedelta("5min"),
            "front_symbol": ["F"] * extra,
            "next_symbol": ["N"] * extra,
            "front_hours_to_delivery": [1000.0] * extra,
            "next_hours_to_delivery": [2000.0] * extra,
        }
    )
    tail_state = pd.DataFrame(
        {
            "pair": ["F|N"] * extra,
            "curve": [0.0] * extra,
            "center": [0.0] * extra,
            "z": [0.0] * extra,
            "source_valid": [True] * extra,
            "two_bar_liquid": [True] * extra,
        }
    )
    source = pd.concat([source, tail_source], ignore_index=True)
    state = pd.concat([state, tail_state], ignore_index=True)
    active = np.zeros(len(source), dtype=bool)
    sides = np.zeros(len(source), dtype=np.int8)
    active[3] = True
    sides[3] = -1
    schedule = module.nonoverlapping_schedule(
        source,
        state,
        active,
        sides,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-02-01"),
    )
    assert len(schedule) == 1
    row = schedule.iloc[0]
    assert pd.Timestamp(row.entry_time) == source.loc[3, "trade_earliest_time_utc"]
    assert row.front_side == 1
    assert row.next_side == -1
    assert row.front_side + row.next_side == 0


def test_support_clock_does_not_filter_entry_with_post_entry_metadata() -> None:
    source, state = _clock_fixture()
    active = np.zeros(len(source), dtype=bool)
    sides = np.zeros(len(source), dtype=np.int8)
    active[3] = True
    sides[3] = -1
    state.loc[4:, "pair"] = "future-change"
    state.loc[4:, "source_valid"] = False
    schedule = module.nonoverlapping_schedule(
        source,
        state,
        active,
        sides,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-02-01"),
    )
    assert len(schedule) == 1


def test_subwindow_is_filtered_from_parent_schedule_without_rescheduling() -> None:
    parent = pd.DataFrame(
        {
            "entry_time": [
                "2022-12-31 23:00:00",
                "2023-01-01 11:00:00",
                "2023-07-01 01:00:00",
            ]
        }
    )
    first_half = module.schedule_window(
        parent, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01")
    )
    assert first_half["entry_time"].tolist() == ["2023-01-01 11:00:00"]


def test_empty_support_gates_fail_closed() -> None:
    empty = module.period_support(pd.DataFrame())
    support = {
        "windows": {
            name: empty
            for name in (
                "fit",
                "fit_2020_partial",
                "fit_2021",
                "fit_2022",
                "select_2023",
                "select_2023_h1",
                "select_2023_h2",
            )
        }
    }
    gates = module.support_gates(support)
    assert not any(gates.values())


def test_manifest_hash_ignores_created_at() -> None:
    payload = {"a": 1, "created_at": "x"}
    other = {"a": 1, "created_at": "y"}
    left = module.canonical_hash({k: v for k, v in payload.items() if k != "created_at"})
    right = module.canonical_hash({k: v for k, v in other.items() if k != "created_at"})
    assert left == right
