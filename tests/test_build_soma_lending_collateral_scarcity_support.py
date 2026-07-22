from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from training import build_soma_lending_collateral_scarcity_support as support


UTC = timezone.utc


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def operation(index: int, day: str = "2020-01-01") -> support.OperationRow:
    return support.OperationRow(
        operation_id=f"SL-{index}",
        operation_date=day,
        available_at=dt(day + "T00:00:00"),
        total_submitted=Decimal(100 + index),
        total_accepted=Decimal(80 + index),
    )


def details_for(row: support.OperationRow, index: int) -> list[support.DetailRow]:
    return [
        support.DetailRow(
            operation_id=row.operation_id,
            operation_date=row.operation_date,
            available_at=row.available_at,
            cusip="A",
            submitted=Decimal(60 + index),
            accepted=Decimal(50 + index),
            weighted_rate=Decimal("0.05") + Decimal(index) / Decimal(10000),
            actual_available=Decimal(1000),
            outstanding_loans=Decimal(100 + index),
        ),
        support.DetailRow(
            operation_id=row.operation_id,
            operation_date=row.operation_date,
            available_at=row.available_at,
            cusip="B",
            submitted=Decimal(40),
            accepted=Decimal(30),
            weighted_rate=Decimal("0.04"),
            actual_available=Decimal(1000),
            outstanding_loans=Decimal(50),
        ),
    ]


def state_row(
    index: int,
    state: int,
    *,
    ready: bool = True,
    available: datetime | None = None,
) -> support.StateRow:
    value = 0.75 if state > 0 else -0.75 if state < 0 else 0.0
    return support.StateRow(
        operation_id=f"SL-{index}",
        operation_date="2020-01-02",
        available_at=available or dt("2020-01-02T00:00:00"),
        rank_ready=ready,
        state=state,
        score=value,
        u_demand_intensity=value if ready else None,
        u_weighted_fee=value if ready else None,
        u_carry_intensity=value if ready else None,
        u_demand_breadth=value if ready else None,
    )


def test_feature_builder_uses_four_exact_weak_components() -> None:
    op = operation(0)
    rows = details_for(op, 0)
    feature = support.build_features([op], rows)[0]
    assert feature.complete is True
    assert feature.demand_intensity == Decimal("0.05")
    assert feature.weighted_fee == Decimal("0.04625")
    assert feature.carry_intensity == Decimal("0.075")
    assert feature.demand_breadth == Decimal(1)


def test_feature_builder_rejects_join_or_total_drift() -> None:
    op = operation(0)
    rows = details_for(op, 0)
    with pytest.raises(RuntimeError, match="totals stopped reconciling"):
        support.build_features([op], [*rows[:-1], support.replace(rows[-1], submitted=Decimal(41))])
    with pytest.raises(RuntimeError, match="join identity mismatch"):
        support.build_features([op], [support.replace(row, operation_date="2020-01-02") for row in rows])


def test_midrank_is_strict_prior_and_tie_deterministic() -> None:
    prior = [Decimal(index) for index in range(support.HISTORY)]
    assert support.midrank_unit(Decimal(252), prior) == 1.0
    ties = [Decimal(5)] * support.HISTORY
    assert support.midrank_unit(Decimal(5), ties) == 0.0
    with pytest.raises(RuntimeError, match="history length"):
        support.midrank_unit(Decimal(1), prior[:-1])


def test_rank_builder_excludes_current_value() -> None:
    features = []
    base = dt("2019-01-01T00:00:00")
    for index in range(253):
        value = Decimal(index + 1)
        features.append(
            support.FeatureRow(
                operation_id=f"SL-{index}",
                operation_date=(base + timedelta(days=index)).date().isoformat(),
                available_at=base + timedelta(days=index),
                complete=True,
                demand_intensity=value,
                weighted_fee=value,
                carry_intensity=value,
                demand_breadth=value,
            )
        )
    rows = support.build_rank_rows(features)
    assert sum(row.rank_ready for row in rows) == 1
    assert rows[-1].u_demand_intensity == 1.0
    assert rows[-1].state == 1


def test_rank_builder_batches_equal_availability_without_cross_contamination() -> None:
    base = dt("2019-01-01T00:00:00")
    features = [
        support.FeatureRow(
            operation_id=f"SL-{index}",
            operation_date=(base + timedelta(days=index)).date().isoformat(),
            available_at=base + timedelta(days=index),
            complete=True,
            demand_intensity=Decimal(0),
            weighted_fee=Decimal(0),
            carry_intensity=Decimal(0),
            demand_breadth=Decimal(0),
        )
        for index in range(support.HISTORY)
    ]
    shared_time = base + timedelta(days=support.HISTORY)
    for suffix in ("A", "B"):
        features.append(
            support.FeatureRow(
                operation_id=f"SL-{suffix}",
                operation_date=shared_time.date().isoformat(),
                available_at=shared_time,
                complete=True,
                demand_intensity=Decimal(100),
                weighted_fee=Decimal(100),
                carry_intensity=Decimal(100),
                demand_breadth=Decimal(100),
            )
        )
    rows = support.build_rank_rows(features)
    assert rows[-2].u_demand_intensity == 1.0
    assert rows[-1].u_demand_intensity == 1.0


def test_primary_state_requires_vote_and_magnitude() -> None:
    assert support.primary_state(
        dict(zip(support.COMPONENTS, [0.8, 0.7, 0.6, -0.1]))
    )[0] == 1
    assert support.primary_state(
        dict(zip(support.COMPONENTS, [-0.8, -0.7, -0.6, 0.1]))
    )[0] == -1
    assert support.primary_state(
        dict(zip(support.COMPONENTS, [0.6, 0.6, 0.1, -0.1]))
    )[0] == 0
    assert support.primary_state(
        dict(zip(support.COMPONENTS, [0.3, 0.3, 0.3, 0.3]))
    )[0] == 0


def test_missing_rank_row_breaks_transition_continuity() -> None:
    rows = [
        state_row(0, 0),
        state_row(1, 1),
        state_row(2, 0, ready=False),
        state_row(3, -1),
        state_row(4, 0),
        state_row(5, -1),
    ]
    candidates = support.candidates_from_states(rows)
    assert [(row.operation_id, row.side) for row in candidates] == [
        ("SL-1", -1),
        ("SL-5", 1),
    ]


def test_schedule_waits_one_bar_skips_overlap_and_split_crossing() -> None:
    first = support.Candidate(
        "A", "2020-01-01", dt("2020-01-01T00:00:00"), -1, 1, 0.6, 0.6, 0.6, 0.6, 0.6
    )
    overlap = support.replace(
        first,
        operation_id="B",
        signal_time=dt("2020-01-02T00:00:00"),
    )
    boundary = support.replace(
        first,
        operation_id="C",
        signal_time=dt("2020-01-04T00:00:00"),
    )
    rows = support.schedule("primary", [first, overlap, boundary])
    assert [row.operation_id for row in rows] == ["A", "C"]
    assert rows[0].entry_time == dt("2020-01-01T00:05:00")
    assert rows[0].exit_time == rows[1].entry_time
    crossing = support.replace(
        first,
        operation_id="X",
        signal_time=dt("2022-12-31T00:00:00"),
    )
    assert support.schedule("primary", [crossing]) == []


def test_source_controls_and_exact_clock_controls_are_complete() -> None:
    rows = [state_row(0, 0), state_row(1, 1)]
    clocks = support.build_control_clocks(rows)
    assert set(clocks) == set(support.CONTROL_NAMES)
    assert len(clocks["primary"]) == 1
    assert clocks["exact_direction_flip"][0].side == 1
    assert clocks["constant_long"][0].entry_time == clocks["primary"][0].entry_time
    assert clocks["constant_short"][0].side == -1


def test_clock_validation_rejects_duplicate_or_mislabeled_rows() -> None:
    clocks = support.build_control_clocks([state_row(0, 0), state_row(1, 1)])
    row = clocks["primary"][0]
    assert support._clock_valid("primary", [row]) is True
    assert support._clock_valid("primary", [row, row]) is False
    assert support._clock_valid(
        "primary", [support.replace(row, split="selection")]
    ) is False
    assert support._clock_valid(
        "primary", [support.replace(row, control="constant_long")]
    ) is False


def test_permutation_preserves_each_year_component_multiset() -> None:
    rows = [
        support.replace(
            state_row(index, 1),
            operation_date=f"2020-01-{index + 1:02d}",
            u_demand_intensity=float(index),
            u_weighted_fee=float(index + 10),
            u_carry_intensity=float(index + 20),
            u_demand_breadth=float(index + 30),
        )
        for index in range(5)
    ]
    permuted = support.permuted_state_rows(rows)
    for component in support.COMPONENTS:
        assert sorted(getattr(row, f"u_{component}") for row in rows) == sorted(
            getattr(row, f"u_{component}") for row in permuted
        )


def test_one_to_one_and_novelty_metrics_are_deterministic() -> None:
    base = dt("2020-01-01T00:05:00")
    primary = [
        support.Scheduled(
            "primary", str(index), "2020-01-01", base + timedelta(days=4 * index),
            base + timedelta(days=4 * index), base + timedelta(days=4 * index + 3),
            "train", 1 if index % 2 == 0 else -1, 1, 0.6, 0.6, 0.6, 0.6, 0.6
        )
        for index in range(3)
    ]
    comparator = [
        support.ComparatorEvent(
            base + timedelta(days=4 * index, hours=12),
            base + timedelta(days=4 * index + 3, hours=12),
            1 if index % 2 == 0 else -1,
        )
        for index in range(3)
    ]
    metrics = support.novelty_metrics(primary, comparator)
    assert metrics["one_day_one_to_one_matches"] == 3
    assert metrics["slcs_one_day_containment"] == 1.0
    assert metrics["exact_entry_intersection"] == 0
    assert metrics["signed_5m_occupied_exposure_correlation"] > 0.7


def test_comparator_side_parser_accepts_frozen_encodings_only() -> None:
    assert support._side("LONG") == 1
    assert support._side("SHORT") == -1
    assert support._side("1") == 1
    assert support._side(-1) == -1
    with pytest.raises(RuntimeError, match="unknown SLCS comparator side"):
        support._side("BUY")


def test_frozen_source_and_comparator_clocks_replay_without_outcomes() -> None:
    operations, details = support.load_source()
    assert len(operations) == 1259
    assert len(details) == 182616
    assert sum(
        current.available_at == previous.available_at
        for previous, current in zip(operations, operations[1:])
    ) == 10
    groups = support._comparator_groups()
    assert len(groups) == 30
    assert sum(len(rows) for rows in groups.values()) == 4123
    assert {row.side for rows in groups.values() for row in rows} == {-1, 1}


def test_clock_schema_contains_no_market_or_outcome_fields() -> None:
    forbidden = {"open", "high", "low", "close", "return", "pnl", "cagr", "mdd", "funding"}
    assert not forbidden.intersection(column.lower() for column in support.CLOCK_COLUMNS)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        support._repository_path("../outside.json")
