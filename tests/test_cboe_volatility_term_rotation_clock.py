from __future__ import annotations

from datetime import date

import pytest

from training import cboe_volatility_term_rotation_clock as clock


def _row(day: str, front: float, broad: float, vix: float = 20.0) -> clock.SourceRow:
    return clock.SourceRow(
        observation_date=date.fromisoformat(day),
        vix9d_close=vix * front,
        vix_close=vix,
        vix3m_close=vix / broad,
    )


def test_strict_prior_midrank_excludes_current() -> None:
    assert clock.strict_prior_midrank(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert clock.strict_prior_midrank(4.0, [1.0, 2.0, 3.0]) == 1.0


def test_future_source_change_cannot_change_earlier_rank() -> None:
    rows = [
        _row(f"2021-01-{day:02d}", 1.0 + day / 100.0, 1.0 + day / 200.0)
        for day in range(1, 21)
    ]
    first = clock.build_features(rows, lookback_observations=10, minimum_history=5)
    changed = rows[:-1] + [_row("2021-01-20", 9.0, 9.0)]
    second = clock.build_features(changed, lookback_observations=10, minimum_history=5)
    assert first[-2] == second[-2]


def test_decision_time_tracks_new_york_dst() -> None:
    assert clock.decision_time(date(2021, 1, 4)).isoformat() == "2021-01-04T14:35:00+00:00"
    assert clock.decision_time(date(2021, 7, 6)).isoformat() == "2021-07-06T13:35:00+00:00"


def test_primary_low_score_is_long_and_high_score_is_short() -> None:
    days = [date(2021, 1, day) for day in range(1, 10)]
    rows = [
        clock.SourceRow(day, 10.0 + i, 10.0, 10.0 / (1.0 + i / 10.0))
        for i, day in enumerate(days)
    ]
    events = clock.build_events(
        rows,
        lookback_observations=4,
        minimum_history=3,
        lower_tail=0.25,
    )
    assert events
    assert all(event.side == "SHORT" for event in events)


def test_schedule_uses_next_two_source_dates_and_delay_shifts_one_release() -> None:
    rows = [
        _row("2021-01-04", 1.0, 1.0),
        _row("2021-01-05", 1.1, 1.1),
        _row("2021-01-06", 1.2, 1.2),
        _row("2021-01-07", 1.3, 1.3),
        _row("2021-01-08", 1.4, 1.4),
        _row("2021-01-11", 1.5, 1.5),
        _row("2021-01-12", 1.6, 1.6),
    ]
    primary = clock.build_events(
        rows, lookback_observations=3, minimum_history=3, lower_tail=0.49
    )
    delayed = clock.build_events(
        rows,
        lookback_observations=3,
        minimum_history=3,
        lower_tail=0.49,
        release_delay=1,
    )
    assert primary[0].entry_time == "2021-01-08T14:35:00+00:00"
    assert primary[0].exit_time == "2021-01-11T14:35:00+00:00"
    assert delayed[0].entry_time == primary[0].exit_time
    assert delayed[0].clock_mode == "one_release_delay"


def test_invalid_release_delay_rejected() -> None:
    with pytest.raises(ValueError, match="zero or one"):
        clock.build_events([], release_delay=2)
