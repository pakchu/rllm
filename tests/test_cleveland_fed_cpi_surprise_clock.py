from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

from training import cleveland_fed_cpi_surprise_clock as clock


def _row(
    *,
    release: str = "2023-03-10T13:30:00+00:00",
    headline_surprise: str = "0.08",
    core_surprise: str = "0.06",
) -> clock.SourceRow:
    headline = Decimal(headline_surprise)
    core = Decimal(core_surprise)
    composite = (headline + core) / Decimal(2)
    parsed_release = datetime.fromisoformat(release).astimezone(timezone.utc)
    return clock.SourceRow(
        reference_month="2023-02-01",
        release_time=parsed_release,
        latest_nowcast_date=date(2023, 3, 9),
        headline_nowcast_mom_pct=Decimal("0.30"),
        core_nowcast_mom_pct=Decimal("0.40"),
        headline_actual_mom_pct=Decimal("0.30") + headline,
        core_actual_mom_pct=Decimal("0.40") + core,
        headline_surprise_pct=headline,
        core_surprise_pct=core,
        composite_surprise_pct=composite,
        surprise_sign_concordant=headline * core > 0,
    )


def test_primary_requires_concordance_and_uses_opposite_risk_direction() -> None:
    hot = clock.build_events([_row()])
    cool = clock.build_events([_row(headline_surprise="-0.08", core_surprise="-0.06")])
    mixed = clock.build_events([_row(headline_surprise="0.20", core_surprise="-0.06")])
    assert hot[0].side == -1
    assert cool[0].side == 1
    assert mixed == []


def test_threshold_is_inclusive_and_component_control_is_separate() -> None:
    exact = _row(headline_surprise="0.06", core_surprise="0.04")
    assert len(clock.build_events([exact], threshold_pct="0.05")) == 1
    mixed = _row(headline_surprise="0.20", core_surprise="-0.06")
    assert clock.build_events([mixed]) == []
    headline = clock.build_events([mixed], mode="headline_only")
    no_concordance = clock.build_events([mixed], mode="composite_no_concordance")
    assert headline[0].side == -1
    assert no_concordance[0].side == -1


def test_execution_is_release_plus_five_minutes_to_same_day_1600_et() -> None:
    event = clock.build_events([_row()])[0]
    assert event.signal_time == "2023-03-10T13:30:00+00:00"
    assert event.entry_time == "2023-03-10T13:35:00+00:00"
    assert event.exit_time == "2023-03-10T21:00:00+00:00"


def test_seven_day_placebo_preserves_new_york_wall_clock_across_dst() -> None:
    event = clock.build_events([_row()], delay_days=7)[0]
    assert event.clock_mode == "seven_day_placebo"
    assert event.entry_time == "2023-03-17T12:35:00+00:00"
    assert event.exit_time == "2023-03-17T20:00:00+00:00"


def test_frozen_source_and_selected_source_only_counts() -> None:
    rows = clock.load_source()
    events = clock.build_events(rows)
    assert len(rows) == 60
    assert len(events) == 40
    stage1 = [event for event in events if "2020" <= event.entry_time[:4] <= "2022"]
    sealed = [event for event in events if event.entry_time.startswith("2023")]
    assert len(stage1) == 26
    assert sum(event.side == 1 for event in stage1) == 10
    assert sum(event.side == -1 for event in stage1) == 16
    assert len(sealed) == 8
    assert all(event.side == 1 for event in sealed)
