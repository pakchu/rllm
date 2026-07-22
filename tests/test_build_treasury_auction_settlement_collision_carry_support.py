from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from training import build_treasury_auction_settlement_collision_carry_support as support


UTC = timezone.utc


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def auction(
    auction_day: str,
    issue_day: str,
    result: str,
    term: str,
    cusip: str,
) -> support.Auction:
    return support.Auction(
        auction_date=date.fromisoformat(auction_day),
        issue_date=date.fromisoformat(issue_day),
        result_available=dt(result),
        term=term,
        cusip=cusip,
    )


def panel_row(day: str, result: str, term: str, cusip: str, complete: str = "true"):
    return {
        "auction_date": day,
        "result_available_at_utc": result,
        "original_security_term": term,
        "cusip": cusip,
        "source_complete": complete,
    }


def raw_row(day: str, issue: str, term: str, cusip: str):
    return {
        "auctionDate": day,
        "issueDate": issue,
        "cusip": cusip,
        "securityType": "Bond" if term in {"20-Year", "30-Year"} else "Note",
        "originalSecurityTerm": term,
        "reopening": "No",
    }


def test_join_materializes_only_pre2024_panel_keys_and_complete_rows() -> None:
    panel = [
        panel_row("2023-01-02", "2023-01-02T22:00:00Z", "5-Year", "A"),
        panel_row("2023-01-03", "2023-01-03T22:00:00Z", "10-Year", "B", "false"),
    ]
    raw = [
        raw_row("2023-01-02", "2023-01-05", "5-Year", "A"),
        raw_row("2023-01-03", "2023-01-05", "10-Year", "B"),
        raw_row("2025-01-02", "2025-01-05", "5-Year", "X"),
    ]
    rows, stats = support.join_source_rows(panel, raw)
    assert [row.cusip for row in rows] == ["A"]
    assert stats["raw_transport_rows_parsed"] == 3
    assert stats["raw_post_2023_transport_rows_parsed_for_key_filter"] == 1
    assert stats["raw_transport_rows_outside_pre2024_panel"] == 1
    assert stats["source_incomplete_rows_excluded"] == 1
    assert stats["post_2023_rows_materialized_into_tascc"] == 0


def test_primary_requires_belly_and_long_on_same_issue_date() -> None:
    rows = [
        auction("2020-01-01", "2020-01-03", "2020-01-02T22:00:00", "5-Year", "A"),
        auction("2020-01-02", "2020-01-03", "2020-01-02T22:00:00", "10-Year", "B"),
        auction("2020-02-01", "2020-02-03", "2020-02-02T22:00:00", "5-Year", "C"),
        auction("2020-03-01", "2020-03-03", "2020-03-02T22:00:00", "2-Year", "D"),
        auction("2020-03-02", "2020-03-03", "2020-03-02T22:00:00", "3-Year", "E"),
    ]
    signals = support.settlement_signals(rows, control="primary", mode="primary")
    assert len(signals) == 1
    assert signals[0].calendar_date == date(2020, 1, 3)
    assert signals[0].terms == ("10-Year", "5-Year")
    assert signals[0].signal_time == dt("2020-01-03T00:00:00")


def test_late_result_skips_settlement_but_result_clock_remains_causal() -> None:
    rows = [
        auction("2020-01-01", "2020-01-03", "2020-01-03T22:00:00", "5-Year", "A"),
        auction("2020-01-02", "2020-01-03", "2020-01-02T22:00:00", "10-Year", "B"),
    ]
    assert support.settlement_signals(rows, control="primary", mode="primary") == []
    result = support.settlement_signals(
        rows, control="result_time_clock", mode="result_time"
    )
    assert len(result) == 1
    assert result[0].signal_time == dt("2020-01-03T22:00:00")


def test_source_controls_have_frozen_geometry() -> None:
    rows = [
        auction("2020-01-01", "2020-01-03", "2020-01-02T22:00:00", "5-Year", "A"),
        auction("2020-01-02", "2020-01-03", "2020-01-02T22:00:00", "10-Year", "B"),
        auction("2020-02-01", "2020-02-03", "2020-02-02T22:00:00", "7-Year", "C"),
    ]
    schedules = support.build_control_schedules(rows)
    assert set(schedules) == {
        "primary",
        "belly_settlement_calendar",
        "long_settlement_calendar",
        "any_multitenor_settlement",
        "single_tenor_settlement",
        "auction_date_collision",
        "term_year_permutation",
        "result_time_clock",
        "settlement_plus_7d",
    }
    assert len(schedules["primary"]) == 1
    assert len(schedules["belly_settlement_calendar"]) == 2
    assert len(schedules["long_settlement_calendar"]) == 1
    assert len(schedules["any_multitenor_settlement"]) == 1
    assert len(schedules["single_tenor_settlement"]) == 1


def test_schedule_waits_one_bar_and_accepts_exact_nonoverlap_boundary() -> None:
    first = support.BasketSignal(
        control="primary",
        calendar_date=date(2020, 1, 1),
        latest_result_available=dt("2019-12-31T22:00:00"),
        signal_time=dt("2020-01-01T00:00:00"),
        terms=("10-Year", "5-Year"),
        cusips=("A", "B"),
    )
    second = support.BasketSignal(
        control="primary",
        calendar_date=date(2020, 1, 4),
        latest_result_available=dt("2020-01-03T22:00:00"),
        signal_time=dt("2020-01-04T00:00:00"),
        terms=("10-Year", "7-Year"),
        cusips=("C", "D"),
    )
    scheduled = support.schedule_nonoverlap([first, second])
    assert len(scheduled) == 2
    assert scheduled[0].entry_time == dt("2020-01-01T00:05:00")
    assert scheduled[0].exit_time == scheduled[1].entry_time
    assert {row.split for row in scheduled} == {"train"}


def test_split_crossing_is_skipped() -> None:
    signal = support.BasketSignal(
        control="primary",
        calendar_date=date(2022, 12, 30),
        latest_result_available=dt("2022-12-29T22:00:00"),
        signal_time=dt("2022-12-30T00:00:00"),
        terms=("10-Year", "5-Year"),
        cusips=("A", "B"),
    )
    assert support.schedule_nonoverlap([signal]) == []


def test_component_controls_are_report_only_for_specificity() -> None:
    empty = {name: [] for name in support.build_control_schedules([])}
    result = support._specificity(empty)
    assert result["passed"] is True
    assert "result_time_clock" in result["component_and_superset_controls_report_only"]
    assert set(result["checks"]) == {"auction_date_collision", "term_year_permutation"}


def test_overlap_metrics_use_exact_and_twelve_hour_windows() -> None:
    base = dt("2020-01-01T00:00:00")
    row = support.ScheduledSignal(
        control="primary",
        signal_id="x",
        split="train",
        calendar_date=date(2020, 1, 1),
        latest_result_available=base - timedelta(hours=2),
        signal_time=base,
        entry_time=base,
        exit_time=base + support.HOLD,
        terms=("10-Year", "5-Year"),
        cusips=("A", "B"),
    )
    metrics = support.overlap_metrics(
        [row], [base, base + timedelta(hours=11), base + timedelta(hours=13)]
    )
    assert metrics["exact_intersection"] == 1
    assert metrics["primary_to_comparator_near_containment"] == 1.0
    assert metrics["comparator_to_primary_near_containment"] == pytest.approx(2 / 3)


def test_clock_schema_contains_no_market_or_outcome_fields() -> None:
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd"}
    assert not forbidden.intersection(name.lower() for name in support.CLOCK_COLUMNS)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        support._repository_path("../outside.json")
