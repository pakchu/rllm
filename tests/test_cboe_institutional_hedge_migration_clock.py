from __future__ import annotations

from datetime import date, timedelta

import pytest

from training import cboe_institutional_hedge_migration_clock as clock


def _row(day: date, index: int) -> clock.SourceRow:
    return clock.SourceRow(
        observation_date=day,
        total_volume=10_000 + 100 * index,
        index_call_volume=2_000 - 10 * index,
        index_put_volume=2_000 + 20 * index,
        index_volume=4_000 + 10 * index,
        equity_call_volume=3_000 + 30 * index,
        equity_put_volume=1_500 - 5 * index,
        vix_call_volume=800 + 20 * index,
        vix_put_volume=400 - 5 * index,
    )


def test_strict_prior_midrank_excludes_current() -> None:
    assert clock.strict_prior_midrank(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert clock.strict_prior_midrank(4.0, [1.0, 2.0, 3.0]) == 1.0


def test_delta_and_level_ranks_are_strictly_prior() -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(10)]
    features = clock.build_features(rows, lookback_observations=4, minimum_history=3)
    assert features[2].institutional_gap_rank is None
    assert features[3].institutional_gap_rank is not None
    assert features[3].delta_institutional_gap_rank is None
    assert features[4].delta_institutional_gap_rank is not None


def test_future_source_change_cannot_change_earlier_feature() -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(12)]
    first = clock.build_features(rows, lookback_observations=4, minimum_history=3)
    changed = rows[:-1] + [
        clock.SourceRow(
            observation_date=rows[-1].observation_date,
            total_volume=100_000,
            index_call_volume=100,
            index_put_volume=50_000,
            index_volume=60_000,
            equity_call_volume=30_000,
            equity_put_volume=100,
            vix_call_volume=20_000,
            vix_put_volume=100,
        )
    ]
    second = clock.build_features(changed, lookback_observations=4, minimum_history=3)
    assert first[-2] == second[-2]


def test_decision_time_tracks_new_york_dst() -> None:
    assert clock.decision_time(date(2021, 1, 4)).isoformat() == (
        "2021-01-04T14:35:00+00:00"
    )
    assert clock.decision_time(date(2021, 7, 6)).isoformat() == (
        "2021-07-06T13:35:00+00:00"
    )


def _high_feature(index: int, day: date) -> clock.FeatureRow:
    return clock.FeatureRow(
        source_index=index,
        observation_date=day,
        institutional_gap=1.0,
        vix_call_pressure=1.0,
        index_share=-0.5,
        delta_institutional_gap=0.2,
        delta_vix_call_pressure=0.2,
        delta_index_share=0.2,
        institutional_gap_rank=0.9,
        vix_call_pressure_rank=0.9,
        index_share_rank=0.9,
        delta_institutional_gap_rank=0.8,
        delta_vix_call_pressure_rank=0.8,
        delta_index_share_rank=0.8,
    )


def test_primary_is_short_only_and_uses_next_source_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(14)]
    monkeypatch.setattr(
        clock,
        "build_features",
        lambda *_args, **_kwargs: [_high_feature(6, rows[6].observation_date)],
    )
    events = clock.build_events(rows)
    assert len(events) == 1
    event = events[0]
    assert event.side == "SHORT"
    assert event.entry_time == clock.decision_time(rows[7].observation_date).isoformat()
    assert event.exit_time == clock.decision_time(rows[8].observation_date).isoformat()


def test_component_control_uses_its_preregistered_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(14)]
    feature = _high_feature(6, rows[6].observation_date)
    feature = clock.FeatureRow(
        **{
            **feature.__dict__,
            "delta_institutional_gap_rank": 0.69,
            "delta_vix_call_pressure_rank": 0.90,
            "delta_index_share_rank": 0.90,
        }
    )
    monkeypatch.setattr(clock, "build_features", lambda *_args, **_kwargs: [feature])
    assert not clock.build_events(rows, mode="institutional_gap_only")
    assert clock.build_events(rows, mode="vix_call_pressure_only")


def test_release_placebos_shift_source_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    start = date(2021, 1, 1)
    rows = [_row(start + timedelta(days=i), i) for i in range(25)]
    monkeypatch.setattr(
        clock,
        "build_features",
        lambda *_args, **_kwargs: [_high_feature(6, rows[6].observation_date)],
    )
    primary = clock.build_events(rows)
    delayed = clock.build_events(rows, release_delay=1)
    placebo = clock.build_events(rows, release_delay=7)
    assert delayed[0].entry_time == primary[0].exit_time
    assert delayed[0].clock_mode == "one_release_delay"
    assert placebo[0].clock_mode == "seven_release_placebo"


def test_invalid_release_delay_rejected() -> None:
    with pytest.raises(ValueError, match="zero, one, or seven"):
        clock.build_events([], release_delay=2)
