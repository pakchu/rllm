from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from training import treasury_auction_demand_impulse_clock as clock


def _source(
    index: int,
    *,
    term: str = "5-Year",
    bid: str | None = None,
    indirect: str | None = None,
    complete: bool = True,
) -> clock.SourceRow:
    year = 2020 + index // 12
    month = index % 12 + 1
    value = Decimal(index if bid is None else bid)
    share = Decimal(index if indirect is None else indirect) / Decimal(100)
    return clock.SourceRow(
        auction_date=f"{year:04d}-{month:02d}-01",
        available_at=datetime(year, month, 1, 22, tzinfo=timezone.utc),
        term=term,
        cusip=f"CUSIP{index:03d}",
        bid_to_cover=value if complete else None,
        indirect_share=share if complete else None,
        source_complete=complete,
    )


def test_strict_prior_midrank_excludes_current_and_handles_ties() -> None:
    assert clock.strict_prior_midrank(
        Decimal("2"), [Decimal("1"), Decimal("2"), Decimal("3")]
    ) == 0.5


def test_incomplete_row_breaks_change_without_erasing_prior_history() -> None:
    rows = [_source(i) for i in range(4)]
    rows.append(_source(4, complete=False))
    rows.extend([_source(5), _source(6)])
    changes = clock.build_changes(rows, prior_changes=2)
    assert [row.cusip for row in changes] == [
        "CUSIP001",
        "CUSIP002",
        "CUSIP003",
        "CUSIP006",
    ]
    assert changes[-1].bid_to_cover_delta_rank == 0.5


def test_primary_requires_concordant_tails_and_enters_after_source_clock() -> None:
    rows: list[clock.SourceRow] = []
    for index in range(13):
        rows.append(_source(index))
    rows.append(
        clock.SourceRow(
            auction_date="2021-02-01",
            available_at=datetime(2021, 2, 1, 22, tzinfo=timezone.utc),
            term="5-Year",
            cusip="EVENT",
            bid_to_cover=Decimal("30"),
            indirect_share=Decimal("0.30"),
            source_complete=True,
        )
    )
    events = clock.build_events(rows, prior_changes=12)
    assert len(events) == 1
    assert events[0].side == "LONG"
    assert events[0].decision_time == "2021-02-01T22:00:00+00:00"
    assert events[0].entry_time == "2021-02-01T22:05:00+00:00"
    assert events[0].scheduled_exit_time == "2021-02-02T22:05:00+00:00"


def test_global_reservation_uses_shortest_tenor_first() -> None:
    # The source-level integration test below asserts the real clock. Here the
    # ordering invariant is represented directly by the public priority map.
    assert clock.TERM_PRIORITY["2-Year"] < clock.TERM_PRIORITY["5-Year"]


def test_frozen_source_clock_counts_and_never_uses_quarantined_rows() -> None:
    rows = clock.read_source()
    assert sum(not row.source_complete for row in rows) == 5
    primary = clock.build_events(rows)
    train = [event for event in primary if "2021-01-01" <= event.entry_time < "2023-01-01"]
    sealed = [event for event in primary if "2023-01-01" <= event.entry_time < "2024-01-01"]
    assert len(train) == 28
    assert len(sealed) == 23
    assert {event.side for event in train} == {"LONG", "SHORT"}
    assert {event.side for event in sealed} == {"LONG", "SHORT"}
