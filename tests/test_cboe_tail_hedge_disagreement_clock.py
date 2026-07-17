from __future__ import annotations

from datetime import date, timedelta

import pytest

from training import cboe_tail_hedge_disagreement_clock as clock


def _row(day: date, index: int) -> clock.SourceRow:
    return clock.SourceRow(
        observation_date=day,
        skew_close=110.0 + index,
        vvix_close=80.0 + 3.0 * index,
        vix_close=30.0 - index,
    )


def test_strict_prior_midrank_excludes_current() -> None:
    assert clock.strict_prior_midrank(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert clock.strict_prior_midrank(4.0, [1.0, 2.0, 3.0]) == 1.0


def test_future_source_change_cannot_change_earlier_feature() -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(12)]
    first = clock.build_features(rows, lookback_observations=4, minimum_history=3)
    changed = rows[:-1] + [clock.SourceRow(rows[-1].observation_date, 500.0, 500.0, 5.0)]
    second = clock.build_features(changed, lookback_observations=4, minimum_history=3)
    assert first[-2] == second[-2]


def test_nested_pressure_rank_is_strictly_prior() -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(12)]
    features = clock.build_features(rows, lookback_observations=4, minimum_history=3)
    assert features[2].skew_rank is None
    assert features[3].skew_rank is not None
    assert features[5].hidden_pressure_rank is None
    assert features[6].hidden_pressure_rank is not None


def test_decision_time_tracks_new_york_dst() -> None:
    assert clock.decision_time(date(2021, 1, 4)).isoformat() == "2021-01-04T14:35:00+00:00"
    assert clock.decision_time(date(2021, 7, 6)).isoformat() == "2021-07-06T13:35:00+00:00"


def _high_pressure_feature(index: int, day: date) -> clock.FeatureRow:
    return clock.FeatureRow(
        source_index=index,
        observation_date=day,
        skew_level=0.2,
        vvix_relative=1.5,
        vix_level=2.5,
        skew_rank=0.9,
        vvix_relative_rank=0.9,
        vix_level_rank=0.1,
        hidden_pressure=0.8,
        hidden_pressure_rank=0.9,
    )


def test_primary_is_short_only_and_uses_next_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(14)]
    monkeypatch.setattr(
        clock,
        "build_features",
        lambda *_args, **_kwargs: [_high_pressure_feature(6, rows[6].observation_date)],
    )
    events = clock.build_events(rows, upper_tail=0.49)
    assert events
    assert all(event.side == "SHORT" for event in events)
    first = events[0]
    signal_index = next(
        i for i, row in enumerate(rows) if row.observation_date.isoformat() == first.observation_date
    )
    assert first.entry_time == clock.decision_time(rows[signal_index + 1].observation_date).isoformat()
    assert first.exit_time == clock.decision_time(rows[signal_index + 2].observation_date).isoformat()


def test_release_placebos_shift_source_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(25)]
    monkeypatch.setattr(
        clock,
        "build_features",
        lambda *_args, **_kwargs: [_high_pressure_feature(6, rows[6].observation_date)],
    )
    primary = clock.build_events(rows, upper_tail=0.49)
    delayed = clock.build_events(
        rows,
        upper_tail=0.49,
        release_delay=1,
    )
    placebo = clock.build_events(
        rows,
        upper_tail=0.49,
        release_delay=7,
    )
    assert delayed[0].entry_time == primary[0].exit_time
    assert delayed[0].clock_mode == "one_release_delay"
    assert placebo[0].clock_mode == "seven_release_placebo"


def test_invalid_release_delay_rejected() -> None:
    with pytest.raises(ValueError, match="zero, one, or seven"):
        clock.build_events([], release_delay=2)
