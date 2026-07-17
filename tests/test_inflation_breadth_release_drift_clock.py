from __future__ import annotations

from decimal import Decimal

import pandas as pd

from training import inflation_breadth_release_drift_clock as clock


def _row(month: str, release: str, headline: str, core: str) -> clock.SourceRow:
    return clock.SourceRow(
        reference_month=month,
        release_time=pd.Timestamp(release),
        headline_yoy_pct=Decimal(headline),
        core_yoy_pct=Decimal(core),
        release_url=f"https://www.bls.gov/{month}",
    )


def test_primary_requires_concordant_nonzero_changes() -> None:
    rows = [
        _row("a", "2021-01-01T13:30:00Z", "2.0", "2.0"),
        _row("b", "2021-02-01T13:30:00Z", "1.9", "1.8"),
        _row("c", "2021-03-01T13:30:00Z", "2.1", "1.7"),
        _row("d", "2021-04-01T12:30:00Z", "2.2", "1.9"),
        _row("e", "2021-05-01T12:30:00Z", "2.2", "2.0"),
    ]
    events = clock.build_events(rows)
    assert [event.reference_month for event in events] == ["b", "d"]
    assert [event.side for event in events] == [1, -1]
    assert events[0].entry_time == "2021-02-01T13:35:00+00:00"
    assert events[0].exit_time == "2021-02-08T13:35:00+00:00"


def test_component_controls_are_source_only_and_one_sided() -> None:
    rows = [
        _row("a", "2021-01-01T13:30:00Z", "2.0", "2.0"),
        _row("b", "2021-02-01T13:30:00Z", "1.9", "2.1"),
    ]
    headline = clock.build_events(rows, component="headline")
    core = clock.build_events(rows, component="core")
    assert [event.side for event in headline] == [1]
    assert [event.side for event in core] == [-1]
    assert clock.build_events(rows) == []


def test_one_release_delay_uses_next_complete_release_clock() -> None:
    rows = [
        _row("a", "2021-01-01T13:30:00Z", "2.0", "2.0"),
        _row("b", "2021-02-01T13:30:00Z", "1.9", "1.9"),
        _row("c", "2021-03-01T13:30:00Z", "2.0", "1.8"),
    ]
    delayed = clock.build_one_release_delay(rows)
    assert len(delayed) == 1
    assert delayed[0].reference_month == "c"
    assert delayed[0].side == 1
    assert delayed[0].signal_time == "2021-03-01T13:30:00+00:00"


def test_frozen_source_distribution() -> None:
    rows = clock.load_source()
    frame = clock.events_frame(clock.build_events(rows))
    stage1 = frame.loc[
        frame["entry_time"].ge(pd.Timestamp("2020-01-01T00:00:00Z"))
        & frame["entry_time"].lt(pd.Timestamp("2023-01-01T00:00:00Z"))
    ]
    stage2 = frame.loc[
        frame["entry_time"].ge(pd.Timestamp("2023-01-01T00:00:00Z"))
        & frame["entry_time"].lt(pd.Timestamp("2024-01-01T00:00:00Z"))
    ]
    assert (len(stage1), int(stage1["side"].eq(1).sum()), int(stage1["side"].eq(-1).sum())) == (20, 8, 12)
    assert (len(stage2), int(stage2["side"].eq(1).sum()), int(stage2["side"].eq(-1).sum())) == (7, 7, 0)
