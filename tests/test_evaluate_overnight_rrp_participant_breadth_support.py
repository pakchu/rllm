from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from training import evaluate_overnight_rrp_participant_breadth_support as orpb


UTC = timezone.utc


def _source_row(
    index: int,
    *,
    complete: bool = True,
    start: date = date(2021, 1, 1),
) -> orpb.SourceRow:
    operation_date = start + timedelta(days=index)
    return orpb.SourceRow(
        operation_date=operation_date,
        available_time=datetime.combine(operation_date, datetime.min.time(), UTC)
        + timedelta(hours=18, minutes=15),
        amount_usd=1_000_000_000 + index * 1_000_000 if complete else None,
        participating_counterparties=50 + index if complete else None,
        accepted_counterparties=40 + index if complete else None,
        source_complete=complete,
    )


def test_source_projection_materializes_only_frozen_fields() -> None:
    values = [f"forbidden-{index}" for index in range(len(orpb.SOURCE_COLUMNS))]
    replacements = {
        "operation_date": "2021-01-04",
        "result_available_at_utc": "2021-01-04T18:15:00+00:00",
        "total_amount_accepted_usd": "1000000000",
        "participating_counterparties": "50",
        "accepted_counterparties": "40",
        "source_complete": "true",
    }
    for column, value in replacements.items():
        values[orpb.SOURCE_COLUMNS.index(column)] = value

    projected = orpb._project_source_record((",".join(values) + "\n").encode())

    assert projected == {
        column: replacements[column]
        for column in orpb.SOURCE_COLUMNS
        if column in replacements
    }
    assert not any("forbidden" in value for value in projected.values())


def test_quarantine_clears_the_exact_21_operation_window() -> None:
    rows = [_source_row(index) for index in range(22)]
    rows.append(_source_row(22, complete=False))
    rows.extend(_source_row(index) for index in range(23, 45))

    features = orpb.build_features(rows)

    assert [feature.source_index for feature in features] == [21, 44]
    assert orpb._feature_windows_are_prior_only(features, rows)


def test_regression_excludes_current_row_and_ranks_against_prior_residuals() -> None:
    prior_amount = [float(index) for index in range(orpb.LOOKBACK)]
    prior_breadth = [2.0 + 3.0 * amount for amount in prior_amount]

    feature = orpb._regression_feature(
        prior_amount,
        prior_breadth,
        current_a=100.0,
        current_b=303.0,
    )

    assert feature.residual == pytest.approx(1.0)
    assert feature.rank == 1.0
    assert feature.rank is not None
    assert orpb._tail_side(feature.rank) == -1


def test_primary_controls_and_one_release_delay_preserve_causal_clock() -> None:
    rows = [_source_row(index) for index in range(24)]
    feature = orpb.FeatureRow(
        source_index=21,
        operation_date=rows[21].operation_date,
        available_time=rows[21].available_time,
        amount_log=1.0,
        accepted_log=1.0,
        participating_log=1.0,
        primary=orpb.RegressionFeature(residual=1.0, rank=1.0),
        participating=orpb.RegressionFeature(residual=0.0, rank=0.5),
        amount_rank=0.5,
        accepted_rank=0.5,
    )

    clocks = orpb.build_clock_rows(rows, [feature])
    by_control = {row.control: row for row in clocks}

    assert set(by_control) == {
        "primary",
        "direction_flip",
        "one_release_delay",
        "deterministic_random_side",
    }
    primary = by_control["primary"]
    delayed = by_control["one_release_delay"]
    assert primary.side == -1
    assert primary.entry_time == rows[21].available_time + orpb.BAR
    assert primary.exit_time == rows[22].available_time + orpb.BAR
    assert delayed.origin_operation_date == rows[21].operation_date
    assert delayed.operation_date == rows[22].operation_date
    assert delayed.entry_time == rows[22].available_time + orpb.BAR
    assert delayed.exit_time == rows[23].available_time + orpb.BAR


def test_support_failure_short_circuits_all_comparator_access() -> None:
    called = False

    def forbidden_loader() -> orpb.ComparatorBundle:
        nonlocal called
        called = True
        raise AssertionError("comparator should remain sealed")

    novelty, rows_read = orpb.maybe_evaluate_novelty(
        [], [], [], False, forbidden_loader
    )

    assert called is False
    assert rows_read == 0
    assert novelty["evaluated"] is False


def test_independent_replay_rejects_corrupted_residual_and_rank() -> None:
    rows, _ = orpb.load_source()
    features = orpb.build_features(rows)
    clocks = orpb.build_clock_rows(rows, features)
    baseline = orpb.independent_replay_checks(features, clocks, rows)
    assert all(baseline.values())

    first = features[0]
    assert first.primary.residual is not None
    assert first.primary.rank is not None
    corrupted_residual = replace(
        first,
        primary=replace(first.primary, residual=first.primary.residual + 0.01),
    )
    residual_checks = orpb.independent_replay_checks(
        [corrupted_residual, *features[1:]], clocks, rows
    )
    assert residual_checks["prior_only_ols_replay_exact"] is False

    corrupted_rank = replace(
        first,
        primary=replace(first.primary, rank=1.0 - first.primary.rank),
    )
    rank_checks = orpb.independent_replay_checks(
        [corrupted_rank, *features[1:]], clocks, rows
    )
    assert rank_checks["prior_only_rank_replay_exact"] is False


def test_tolerant_matching_is_one_to_one() -> None:
    start = orpb.COMPARISON_START
    left = [
        orpb.ComparatorClock("left", start, start + orpb.BAR, 1),
        orpb.ComparatorClock(
            "left", start + timedelta(hours=4), start + timedelta(hours=5), -1
        ),
    ]
    right = [
        orpb.ComparatorClock(
            "right", start + timedelta(hours=1), start + timedelta(hours=2), 1
        )
    ]

    assert orpb.maximum_tolerant_matches(left, right) == 1


def test_signed_exposure_correlation_uses_the_full_frozen_grid() -> None:
    start = orpb.COMPARISON_START
    left = [orpb.ComparatorClock("left", start, start + timedelta(hours=1), 1)]
    same = [orpb.ComparatorClock("same", start, start + timedelta(hours=1), 1)]
    opposite = [orpb.ComparatorClock("opposite", start, start + timedelta(hours=1), -1)]

    assert orpb.signed_exposure_correlation(left, same) == pytest.approx(1.0)
    assert orpb.signed_exposure_correlation(left, opposite) == pytest.approx(-1.0)
