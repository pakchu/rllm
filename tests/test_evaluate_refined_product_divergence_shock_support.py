from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from training import evaluate_refined_product_divergence_shock_support as rpds


UTC = timezone.utc


def _row(
    release: date,
    *,
    crude: str,
    gasoline: str,
    distillate: str,
    complete: bool = True,
    consistent: bool = True,
) -> rpds.SourceRow:
    return rpds.SourceRow(
        release_date=release,
        available_time=datetime.combine(
            release + timedelta(days=1), datetime.min.time(), UTC
        )
        + timedelta(hours=13),
        source_complete=complete,
        published_difference_consistent=consistent,
        crude=Decimal(crude),
        gasoline=Decimal(gasoline),
        distillate=Decimal(distillate),
    )


def test_primary_uses_only_the_frozen_refined_crude_divergence() -> None:
    long_row = _row(date(2021, 1, 6), crude="-1", gasoline="2", distillate="3")
    short_row = _row(date(2021, 1, 13), crude="4", gasoline="-2", distillate="-3")
    concordant = _row(date(2021, 1, 20), crude="1", gasoline="2", distillate="3")
    disagreement = _row(date(2021, 1, 27), crude="-1", gasoline="2", distillate="-3")
    zero = _row(date(2021, 2, 3), crude="-1", gasoline="0", distillate="3")
    controls = rpds.build_raw_events(
        [long_row, short_row, concordant, disagreement, zero]
    )
    assert [event.side for event in controls["primary"]] == [1, -1]
    assert [event.side for event in controls["epsb_concordance_48h"]] == [1]
    assert all(event.release_date != zero.release_date for event in controls["primary"])


def test_one_release_delay_uses_next_eligible_release_and_quarantine_clears() -> None:
    first = _row(date(2021, 1, 6), crude="-1", gasoline="2", distillate="3")
    next_eligible = _row(date(2021, 1, 13), crude="1", gasoline="2", distillate="-3")
    second = _row(date(2021, 1, 20), crude="4", gasoline="-2", distillate="-3")
    quarantined = _row(
        date(2021, 1, 27),
        crude="1",
        gasoline="2",
        distillate="3",
        complete=False,
        consistent=False,
    )
    after_quarantine = _row(date(2021, 2, 3), crude="1", gasoline="2", distillate="-3")
    delayed = rpds.build_raw_events(
        [first, next_eligible, second, quarantined, after_quarantine]
    )["one_release_delay"]
    assert len(delayed) == 1
    assert delayed[0].origin_release_date == first.release_date
    assert delayed[0].release_date == next_eligible.release_date
    assert delayed[0].side == 1


def test_scheduler_enforces_nonoverlap_without_score_priority() -> None:
    first_row = _row(date(2021, 1, 6), crude="-1", gasoline="2", distillate="3")
    second_row = _row(date(2021, 1, 7), crude="4", gasoline="-2", distillate="-3")
    first = rpds._event(first_row, control="primary", side=1)
    second = rpds._event(second_row, control="primary", side=-1)
    assert rpds.schedule_events([second, first]) == [first]


def test_tolerant_matching_is_one_to_one() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    left = [
        rpds.ComparatorClock("a", start, start + rpds.HOLD, 1),
        rpds.ComparatorClock(
            "a", start + timedelta(hours=12), start + timedelta(hours=60), -1
        ),
    ]
    right = [
        rpds.ComparatorClock(
            "b", start + timedelta(hours=1), start + timedelta(hours=2), 1
        )
    ]
    assert rpds.maximum_tolerant_matches(left, right) == 1


def test_exposure_correlation_tracks_signed_occupied_paths() -> None:
    start = rpds.COMPARISON_START
    left = [rpds.ComparatorClock("a", start, start + timedelta(hours=1), 1)]
    same = [rpds.ComparatorClock("b", start, start + timedelta(hours=1), 1)]
    opposite = [rpds.ComparatorClock("c", start, start + timedelta(hours=1), -1)]
    assert rpds.signed_exposure_correlation(left, same) == pytest.approx(1.0)
    assert rpds.signed_exposure_correlation(left, opposite) == pytest.approx(-1.0)


def test_support_failure_short_circuits_comparator_access() -> None:
    called = False

    def forbidden_loader() -> tuple[dict[str, list[rpds.ComparatorClock]], int]:
        nonlocal called
        called = True
        raise AssertionError("comparator should remain sealed")

    novelty, rows_read = rpds.maybe_evaluate_novelty([], False, forbidden_loader)
    assert called is False
    assert rows_read == 0
    assert novelty["evaluated"] is False


def test_source_projection_materializes_only_frozen_fields() -> None:
    values = [f"forbidden-{index}" for index in range(len(rpds.SOURCE_COLUMNS))]
    replacements = {
        "release_date": "2021-01-06",
        "available_time_utc": "2021-01-07T13:00:00Z",
        "source_complete": "True",
        "published_difference_consistent": "True",
        "commercial_crude_change_mmbbl": "-1",
        "gasoline_change_mmbbl": "2",
        "distillate_change_mmbbl": "3",
    }
    for column, value in replacements.items():
        values[rpds.SOURCE_COLUMNS.index(column)] = value
    projected = rpds._project_source_record((",".join(values) + "\n").encode())
    assert projected == {
        column: replacements[column]
        for column in rpds.SOURCE_COLUMNS
        if column in replacements
    }
    assert not any("forbidden" in value for value in projected.values())


def test_comparator_validation_rejects_out_of_grid_and_bad_decision_clock() -> None:
    start = rpds.COMPARISON_START
    with pytest.raises(RuntimeError, match="clock order"):
        rpds._validate_comparator_times(
            family="live",
            split="train",
            decision_time=start + timedelta(hours=2),
            entry_time=start + timedelta(hours=1),
            exit_time=start + timedelta(hours=3),
            release_date=None,
        )
    with pytest.raises(RuntimeError, match="declared split"):
        rpds._validate_comparator_times(
            family="live",
            split="selection",
            decision_time=rpds.COMPARISON_END,
            entry_time=rpds.COMPARISON_END,
            exit_time=rpds.COMPARISON_END + timedelta(hours=1),
            release_date=None,
        )


def test_comparator_validation_allows_only_complete_epsb_history_rows() -> None:
    history_entry = rpds.COMPARISON_START - timedelta(days=3)
    assert (
        rpds._validate_comparator_times(
            family="epsb",
            split=None,
            decision_time=history_entry - rpds.BAR,
            entry_time=history_entry,
            exit_time=rpds.COMPARISON_START - timedelta(hours=1),
            release_date=(history_entry - rpds.BAR).date() - timedelta(days=1),
        )
        is False
    )
    with pytest.raises(RuntimeError, match="straddles"):
        rpds._validate_comparator_times(
            family="epsb",
            split=None,
            decision_time=rpds.COMPARISON_START - timedelta(hours=1),
            entry_time=rpds.COMPARISON_START - timedelta(minutes=55),
            exit_time=rpds.COMPARISON_START + timedelta(hours=1),
            release_date=(rpds.COMPARISON_START - timedelta(hours=1)).date()
            - timedelta(days=1),
        )
