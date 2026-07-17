from __future__ import annotations

from datetime import datetime, timedelta, timezone

from training import overnight_rrp_flow_release_clock as clock


UTC = timezone.utc


def _rows(count: int = 140) -> list[clock.SourceRow]:
    start = datetime(2020, 1, 2, 18, 30, tzinfo=UTC)
    rows = []
    for index in range(count):
        available = start + timedelta(days=index)
        amount = int((100 + index % 11) * 1_000_000_000)
        rows.append(
            clock.SourceRow(
                operation_date=available.date().isoformat(),
                available_at=available,
                amount_usd=amount,
                source_complete=True,
            )
        )
    return rows


def test_strict_prior_rank_excludes_current_value() -> None:
    assert clock.strict_prior_midrank(3.0, [1.0, 2.0, 3.0, 4.0]) == 0.625


def test_current_change_is_ranked_before_append() -> None:
    rows = _rows()
    features = clock.build_features(rows, baseline_operations=2, rank_operations=20)
    ranked = [row for row in features if row.residual_rank is not None]
    assert ranked
    first = ranked[0]
    prior = [
        row.residual_innovation
        for row in features
        if row.source_index < first.source_index and row.residual_innovation is not None
    ][-20:]
    assert len(prior) == 20
    assert first.residual_rank == clock.strict_prior_midrank(
        first.residual_innovation,  # type: ignore[arg-type]
        prior,  # type: ignore[arg-type]
    )


def test_quarantine_resets_local_baselines_without_emitting() -> None:
    rows = _rows(30)
    rows[10] = clock.SourceRow(
        operation_date=rows[10].operation_date,
        available_at=rows[10].available_at,
        amount_usd=None,
        source_complete=False,
    )
    features = clock.build_features(rows, baseline_operations=5, rank_operations=20)
    by_index = {row.source_index: row for row in features}
    assert 10 not in by_index
    assert by_index[11].residual_innovation is None
    assert by_index[15].residual_innovation is None
    assert by_index[16].residual_innovation is not None
    assert by_index[11].one_day_delta is None
    assert by_index[12].one_day_delta is not None


def test_event_enters_after_source_and_exits_on_next_operation_clock() -> None:
    rows = _rows()
    events = clock.build_events(
        rows, baseline_operations=2, rank_operations=20, lower_tail=0.25
    )
    assert events
    for event in events:
        source_index = next(
            index for index, row in enumerate(rows) if row.operation_date == event.operation_date
        )
        assert clock._parse_utc(event.entry_time) == rows[source_index].available_at + timedelta(minutes=5)
        assert clock._parse_utc(event.scheduled_exit_time) == rows[source_index + 1].available_at + timedelta(minutes=5)


def test_primary_direction_is_liquidity_release_long() -> None:
    rows = _rows(140)
    # Force a very small current take-up after enough history.
    target = 130
    rows[target] = clock.SourceRow(
        operation_date=rows[target].operation_date,
        available_at=rows[target].available_at,
        amount_usd=0,
        source_complete=True,
    )
    events = clock.build_events(
        rows, baseline_operations=2, rank_operations=20, lower_tail=0.25
    )
    selected = [event for event in events if event.operation_date == rows[target].operation_date]
    assert len(selected) == 1
    assert selected[0].side == "LONG"
