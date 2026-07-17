from __future__ import annotations

from decimal import Decimal

import pandas as pd

from training import eia_petroleum_stock_breadth_clock as clock


def _row(
    release: str,
    available: str,
    crude: str,
    gasoline: str,
    distillate: str,
    *,
    complete: bool = True,
) -> clock.SourceRow:
    return clock.SourceRow(
        release_date=release,
        available_time=pd.Timestamp(available),
        commercial_crude_change_mmbbl=Decimal(crude),
        gasoline_change_mmbbl=Decimal(gasoline),
        distillate_change_mmbbl=Decimal(distillate),
        archive_page_url=f"https://www.eia.gov/{release}",
        table1_csv_url=f"https://www.eia.gov/{release}/table1.csv",
        source_complete=complete,
    )


def test_primary_requires_concordant_nonzero_changes() -> None:
    rows = [
        _row("a", "2021-01-07T13:00:00Z", "1", "2", "3"),
        _row("b", "2021-01-14T13:00:00Z", "-1", "-2", "-3"),
        _row("c", "2021-01-21T13:00:00Z", "1", "-2", "3"),
        _row("d", "2021-01-28T13:00:00Z", "0", "1", "1"),
        _row("e", "2021-02-04T13:00:00Z", "1", "1", "1", complete=False),
    ]
    events = clock.build_events(rows)
    assert [event.release_date for event in events] == ["a", "b"]
    assert [event.side for event in events] == [1, -1]
    assert events[0].entry_time == "2021-01-07T13:05:00+00:00"
    assert events[0].exit_time == "2021-01-10T13:05:00+00:00"


def test_mechanism_controls_remain_source_only() -> None:
    rows = [_row("a", "2021-01-07T13:00:00Z", "1", "-2", "-3")]
    assert [event.side for event in clock.build_events(rows, mode="crude_only")] == [1]
    assert [
        event.side
        for event in clock.build_events(rows, mode="refined_products_only")
    ] == [-1]
    assert clock.build_events(rows) == []


def test_one_release_delay_uses_next_complete_issue() -> None:
    rows = [
        _row("a", "2021-01-07T13:00:00Z", "1", "1", "1"),
        _row("b", "2021-01-14T13:00:00Z", "1", "-1", "1", complete=False),
        _row("c", "2021-01-21T13:00:00Z", "-1", "1", "-1"),
    ]
    delayed = clock.build_one_release_delay(rows)
    assert len(delayed) == 1
    assert delayed[0].release_date == "c"
    assert delayed[0].side == 1


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
    assert (
        len(stage1),
        int(stage1["side"].eq(1).sum()),
        int(stage1["side"].eq(-1).sum()),
    ) == (37, 13, 24)
    assert (
        len(stage2),
        int(stage2["side"].eq(1).sum()),
        int(stage2["side"].eq(-1).sum()),
    ) == (13, 6, 7)
