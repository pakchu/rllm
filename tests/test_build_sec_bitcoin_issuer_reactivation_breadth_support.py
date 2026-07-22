from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from training import build_sec_bitcoin_issuer_reactivation_breadth_support as support


UTC = timezone.utc


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def event(ready: datetime, issuer: str, accession: str) -> support.IssuerEvent:
    return support.IssuerEvent(
        ready=ready,
        issuer=issuer,
        accessions=(accession,),
        forms=("8-K",),
    )


def row(accepted: str, accession: str, issuer: str, *, amendment: bool = False):
    return {
        "acceptance_datetime": accepted,
        "accession": accession,
        "amendment": amendment,
        "ciks": [issuer],
        "form": "8-K/A" if amendment else "8-K",
    }


def test_source_rows_deduplicate_accessions_and_collapse_same_issuer_batch() -> None:
    rows = [
        row("2020-01-01T00:00:00Z", "a", "10"),
        row("2020-01-01T00:00:00Z", "a", "10"),
        row("2020-01-01T00:00:00Z", "b", "10"),
        row("2020-01-01T00:00:00Z", "c", "20"),
        row("2020-01-02T00:00:00Z", "d", "30", amendment=True),
    ]
    events, stats = support.source_events_from_rows(rows)
    assert stats["unique_accessions"] == 4
    assert stats["eligible_accessions"] == 3
    assert stats["issuer_ready_events"] == 2
    assert events[0].accessions == ("a", "b")
    assert events[0].ready == dt("2020-01-01T01:00:00")


def test_classification_uses_prior_atomic_batch_and_inclusive_365_day_gap() -> None:
    base = dt("2018-01-01T00:00:00")
    events = [
        event(base, "1", "a"),
        event(base + timedelta(days=10), "1", "b"),
        event(base + timedelta(days=375), "1", "c"),
        event(base + timedelta(days=375), "2", "d"),
        event(base + timedelta(days=740), "1", "e"),
    ]
    result = support.classify_events(events)
    assert [item.accessions for item in result["birth"]] == [("a",), ("d",)]
    assert [item.accessions for item in result["repeat"]] == [("b",)]
    assert [item.accessions for item in result["reactivation"]] == [
        ("c",),
        ("e",),
    ]
    assert result["reactivation"][0].gap_seconds == 365 * 86400


def test_breadth_crosses_once_and_window_is_left_open() -> None:
    start = dt("2020-01-01T00:00:00")
    events = [
        event(start, "1", "a"),
        event(start + timedelta(days=1), "2", "b"),
        event(start + timedelta(days=2), "3", "c"),
        event(start + timedelta(days=3), "4", "d"),
        event(start + timedelta(days=7), "5", "e"),
        event(start + timedelta(days=9, minutes=1), "6", "f"),
        event(start + timedelta(days=10, minutes=1), "7", "g"),
    ]
    signals = support.breadth_signals(events, control="primary", threshold=3)
    assert [signal.signal_ready for signal in signals] == [
        start + timedelta(days=2),
        start + timedelta(days=9, minutes=1),
        start + timedelta(days=10, minutes=1),
    ]
    assert signals[0].breadth_issuers == ("1", "2", "3")


def test_same_ready_batch_emits_one_signal_without_intra_batch_order() -> None:
    ready = dt("2020-01-01T00:00:00")
    signals = support.breadth_signals(
        [event(ready, str(i), chr(96 + i)) for i in range(1, 5)],
        control="primary",
        threshold=3,
    )
    assert len(signals) == 1
    assert signals[0].breadth_issuers == ("1", "2", "3", "4")
    assert signals[0].trigger_accessions == ("a", "b", "c", "d")


def test_schedule_waits_one_bar_and_skips_overlap_and_split_crossing() -> None:
    first = support.BreadthSignal(
        control="primary",
        signal_ready=dt("2020-01-01T00:00:00"),
        threshold=3,
        trigger_accessions=("c",),
        breadth_accessions=("a", "b", "c"),
        breadth_issuers=("1", "2", "3"),
    )
    overlap = support.BreadthSignal(
        control="primary",
        signal_ready=dt("2020-01-02T00:01:00"),
        threshold=3,
        trigger_accessions=("f",),
        breadth_accessions=("d", "e", "f"),
        breadth_issuers=("4", "5", "6"),
    )
    crossing = support.BreadthSignal(
        control="primary",
        signal_ready=dt("2022-12-30T00:00:00"),
        threshold=3,
        trigger_accessions=("i",),
        breadth_accessions=("g", "h", "i"),
        breadth_issuers=("7", "8", "9"),
    )
    scheduled = support.schedule_nonoverlap([first, overlap, crossing])
    assert len(scheduled) == 1
    assert scheduled[0].entry_time == dt("2020-01-01T00:05:00")
    assert scheduled[0].exit_time == dt("2020-01-06T00:05:00")


def test_single_component_is_report_only_not_a_specificity_failure() -> None:
    ready = dt("2020-01-01T00:00:00")
    primary = support.schedule_nonoverlap(
        [
            support.BreadthSignal(
                control="primary",
                signal_ready=ready,
                threshold=3,
                trigger_accessions=("c",),
                breadth_accessions=("a", "b", "c"),
                breadth_issuers=("1", "2", "3"),
            )
        ]
    )
    schedules = {name: [] for name in (
        "primary",
        "first_ever_birth_breadth",
        "any_mention_breadth",
        "repeat_filer_breadth",
        "single_reactivation",
        "stale_30d",
        "year_cik_permutation",
        "threshold_two",
        "threshold_four",
    )}
    schedules["primary"] = primary
    schedules["single_reactivation"] = primary
    result = support._specificity_results(schedules)
    assert result["single_reactivation_proximity_is_report_only"] is True
    assert "single_reactivation" not in result["checks"]


def test_overlap_metrics_use_exact_and_symmetric_near_windows() -> None:
    base = dt("2020-01-01T00:00:00")
    signal = support.ScheduledSignal(
        control="primary",
        signal_id="x",
        split="train",
        signal_ready=base,
        entry_time=base,
        exit_time=base + support.HOLD,
        threshold=3,
        trigger_accessions=("c",),
        breadth_accessions=("a", "b", "c"),
        breadth_issuers=("1", "2", "3"),
    )
    metrics = support.overlap_metrics(
        [signal], [base, base + timedelta(hours=11), base + timedelta(hours=13)]
    )
    assert metrics["exact_intersection"] == 1
    assert metrics["primary_to_comparator_near_containment"] == 1.0
    assert metrics["comparator_to_primary_near_containment"] == pytest.approx(2 / 3)


def test_clock_rows_contain_no_market_or_outcome_fields() -> None:
    schedules = support.build_control_schedules([])
    rows = support._clock_rows(schedules)
    assert rows == []
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd"}
    assert not forbidden.intersection(name.lower() for name in support.CLOCK_COLUMNS)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        support._repository_path("../outside.json")
