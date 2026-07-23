from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from training import build_radial_quote_handoff_relay_support as support
from training import preregister_residual_notional_centroid_migration as rncm


D = Decimal


def signal_row(
    position: int,
    *,
    near_sign: int = 0,
    far_sign: int = 0,
    near_intensity: int = 0,
    far_intensity: int = 0,
    near_efficiency: str = "1",
    far_efficiency: str = "1",
    near_threshold: int | None = 2,
    far_threshold: int | None = 2,
    complete: bool = True,
) -> support.SignalRow:
    def pair(sign: int, intensity: int) -> tuple[Decimal, Decimal]:
        value = D(intensity) * D(sign)
        return value, value

    near = pair(near_sign, near_intensity)
    far = pair(far_sign, far_intensity)
    date = support.GRID_START + position * support.BAR
    source = support.SourceRow(
        position=position,
        date=date,
        available_at=date + support.BAR,
        complete=complete,
        net=(*near, *far),
        path=tuple(abs(value) for value in (*near, *far)),
        efficiency=(
            D(near_efficiency),
            D(near_efficiency),
            D(far_efficiency),
            D(far_efficiency),
        ),
    )
    return support.SignalRow(
        source=source,
        near_sign=near_sign if complete else 0,
        far_sign=far_sign if complete else 0,
        near_intensity=D(near_intensity) if complete else None,
        far_intensity=D(far_intensity) if complete else None,
        near_efficiency=D(near_efficiency) if complete else None,
        far_efficiency=D(far_efficiency) if complete else None,
        near_threshold=(
            D(near_threshold) if near_threshold is not None else None
        ),
        far_threshold=D(far_threshold) if far_threshold is not None else None,
    )


def candidate(
    signal_time: datetime,
    *,
    position: int,
    side: int = 1,
    age: int = 1,
    control: str = "primary",
) -> support.Candidate:
    return support.Candidate(
        control=control,
        signal_position=position,
        signal_time=signal_time,
        side=side,
        confirmation_age=age,
    )


def valid_synthetic_payload() -> dict:
    scenarios = {
        name: {
            "grid_rows": support.GRID_ROWS,
            "complete_rows": (
                support.GRID_ROWS - 315
                if name == "missing_rows"
                else support.GRID_ROWS
            ),
            "scheduled_snapshot_slots": support.GRID_ROWS * 10,
            "raw_confirmations": 0,
            "accepted_events": 0,
            "race_audit": {
                field: 0 for field in support.RACE_AUDIT_FIELDS
            },
            "passed": True,
        }
        for name in support.NULL_SCENARIOS
    }
    payload = {
        "protocol_version": f"{support.PROTOCOL_VERSION}_synthetic_v1",
        "candidate": support.POLICY_ID,
        "preregistration": {
            "path": str(support.PREREGISTRATION),
            "sha256": support.PREREGISTRATION_SHA256,
            "manifest_hash": "m" * 64,
            "policy_hash": "p" * 64,
        },
        "support_source": {
            "path": str(support.SCRIPT_PATH),
            "sha256": support.sha256_file(support.SCRIPT_PATH),
        },
        "dependencies": support._frozen_dependency_payload(),
        "synthetic_null": {
            "protocol": "RQHR-72 hash-bound synthetic moving-band null",
            "candidate": support.POLICY_ID,
            "source_rows_read": 0,
            "real_rqhr_columns_read": 0,
            "comparator_rows_read": 0,
            "market_or_outcome_rows_read": 0,
            "parameters": {
                "grid_rows": support.GRID_ROWS,
                "snapshots_per_non_suppressed_bar": 10,
                "strict_prior_window_rows": support.PRIOR_WINDOW,
                "strict_prior_minimum_values": support.PRIOR_MINIMUM,
                "near_quantile": "39/40",
                "far_quantile": "9/10",
            },
            "scenarios": scenarios,
            "passed": True,
            "failure_action": (
                "real source may be opened only by frozen support builder"
            ),
        },
        "real_source_rows_read": 0,
        "real_rqhr_columns_read": 0,
        "comparator_rows_read": 0,
        "market_or_outcome_rows_read": 0,
        "passed": True,
    }
    payload["manifest_hash"] = support.canonical_hash(payload)
    return payload


def test_strict_prior_nearest_rank_excludes_current_without_interpolation() -> None:
    values = [D(value) for value in range(1, 101)] + [D(10_000)]
    threshold = support.strict_prior_thresholds(
        values,
        window=100,
        minimum=100,
        quantile=(39, 40),
    )
    assert threshold[99] is None
    assert threshold[100] == D(98)

    short = [D(0), D(10), D(20), D(30), D(999)]
    threshold = support.strict_prior_thresholds(
        short,
        window=4,
        minimum=4,
        quantile=(9, 10),
    )
    assert threshold[4] == D(30)


def test_strict_prior_window_is_calendar_rows_not_compressed_values() -> None:
    values = [D(1), D(2), D(3), D(4), None, None, None, None, D(9)]
    threshold = support.strict_prior_thresholds(
        values,
        window=4,
        minimum=2,
        quantile=(1, 2),
    )
    assert threshold[-1] is None


@pytest.mark.parametrize("quantile", [(39, 40), (9, 10)])
def test_fenwick_thresholds_match_naive_strict_prior_reference(
    quantile: tuple[int, int],
) -> None:
    values = [
        None if index % 5 == 0 else D((index * 7) % 13)
        for index in range(57)
    ]
    observed = support.strict_prior_thresholds(
        values,
        window=11,
        minimum=5,
        quantile=quantile,
    )
    expected: list[Decimal | None] = []
    numerator, denominator = quantile
    for position in range(len(values)):
        prior = sorted(
            value
            for value in values[max(0, position - 11) : position]
            if value is not None
        )
        if len(prior) < 5:
            expected.append(None)
            continue
        rank = (numerator * len(prior) + denominator - 1) // denominator
        expected.append(prior[rank - 1])
    assert observed == expected


@pytest.mark.parametrize(
    ("net", "path", "efficiency", "message"),
    [
        ("3", "2.999999", "1", "below absolute net"),
        ("1", "0", "0", "zero path"),
        ("0", "0", "0.1", "zero path"),
        ("3", "6", "0.4", "efficiency algebra"),
        ("1", "1", "1.1", "outside"),
    ],
)
def test_source_algebra_fails_closed(
    net: str,
    path: str,
    efficiency: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        support.validate_algebra(D(net), D(path), D(efficiency))


def test_source_algebra_accepts_exact_and_tolerance_bound_values() -> None:
    support.validate_algebra(D(3), D(6), D("0.5"))
    support.validate_algebra(
        D(3),
        D(6),
        D("0.500000000004"),
    )


def test_primary_relay_confirms_at_age_one_and_six() -> None:
    age_one = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(1, near_sign=1, near_intensity=3, far_sign=0),
        signal_row(
            2,
            near_sign=1,
            near_intensity=1,
            far_sign=1,
            far_intensity=3,
        ),
    ]
    events, audit = support.relay_candidates(age_one, control="primary")
    assert [(row.signal_position, row.side, row.confirmation_age) for row in events] == [
        (2, 1, 1)
    ]
    assert audit["confirmations"] == 1

    age_six = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(1, near_sign=1, near_intensity=3),
        *[
            signal_row(position, near_sign=1, near_intensity=1)
            for position in range(2, 7)
        ],
        signal_row(
            7,
            near_sign=1,
            near_intensity=1,
            far_sign=1,
            far_intensity=3,
        ),
    ]
    events, _ = support.relay_candidates(age_six, control="primary")
    assert len(events) == 1
    assert events[0].confirmation_age == 6


def test_primary_timeout_blocks_age_seven_confirmation() -> None:
    rows = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(1, near_sign=1, near_intensity=3),
        *[
            signal_row(position, near_sign=1, near_intensity=1)
            for position in range(2, 8)
        ],
        signal_row(
            8,
            near_sign=1,
            near_intensity=1,
            far_sign=1,
            far_intensity=3,
        ),
    ]
    events, audit = support.relay_candidates(rows, control="primary")
    assert events == []
    assert audit["timeouts"] == 1


@pytest.mark.parametrize(
    ("current_near", "expected_confirmations"),
    [
        ((D(-7), D(1)), 0),
        ((D(-8), D(1)), 0),
        ((D(-5), D(1)), 1),
    ],
)
def test_primary_confirmation_requires_strict_cumulative_arm_sign(
    current_near: tuple[Decimal, Decimal],
    expected_confirmations: int,
) -> None:
    arm_previous = signal_row(0, near_sign=1, near_intensity=1)
    arm = signal_row(1, near_sign=1, near_intensity=3)
    confirmation = signal_row(
        2,
        near_sign=0,
        near_intensity=4,
        far_sign=1,
        far_intensity=3,
    )
    source = replace(
        confirmation.source,
        net=(*current_near, *confirmation.source.net[2:]),
        path=(
            abs(current_near[0]),
            abs(current_near[1]),
            *confirmation.source.path[2:],
        ),
    )
    confirmation = replace(confirmation, source=source)

    events, audit = support.relay_candidates(
        [arm_previous, arm, confirmation],
        control="primary",
    )
    assert len(events) == expected_confirmations
    assert audit["confirmations"] == expected_confirmations


def test_primary_ambiguity_and_incomplete_rows_retire_without_rearm() -> None:
    ambiguous = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(1, near_sign=1, near_intensity=10),
        signal_row(
            2,
            near_sign=-1,
            near_intensity=1,
            far_sign=1,
            far_intensity=3,
        ),
    ]
    events, audit = support.relay_candidates(ambiguous, control="primary")
    assert events == []
    assert audit["arms"] == 1
    assert audit["ambiguities"] == 1
    assert audit["terminal_consumed_rows"] == 1

    incomplete = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(1, near_sign=1, near_intensity=3),
        signal_row(2, complete=False),
    ]
    events, audit = support.relay_candidates(incomplete, control="primary")
    assert events == []
    assert audit["incomplete_cancellations"] == 1


def test_same_sign_far_blocks_arm_and_previous_equality_does_not_cross() -> None:
    blocked = [
        signal_row(0, near_sign=1, near_intensity=1),
        signal_row(
            1,
            near_sign=1,
            near_intensity=3,
            far_sign=1,
            far_intensity=3,
        ),
    ]
    _, audit = support.relay_candidates(blocked, control="primary")
    assert audit["arms"] == 0

    equality = [
        signal_row(0, near_sign=1, near_intensity=2),
        signal_row(1, near_sign=1, near_intensity=3),
    ]
    _, audit = support.relay_candidates(equality, control="primary")
    assert audit["arms"] == 0


def test_no_efficiency_and_reverse_controls_keep_other_rules() -> None:
    low_efficiency = [
        signal_row(0, near_sign=1, near_intensity=1, near_efficiency="0"),
        signal_row(1, near_sign=1, near_intensity=3, near_efficiency="0"),
        signal_row(
            2,
            near_sign=1,
            near_intensity=1,
            near_efficiency="0",
            far_sign=1,
            far_intensity=3,
            far_efficiency="0",
        ),
    ]
    primary, _ = support.relay_candidates(low_efficiency, control="primary")
    no_eff, _ = support.relay_candidates(
        low_efficiency,
        control="no_efficiency_relay",
        use_efficiency=False,
    )
    assert primary == []
    assert len(no_eff) == 1

    reverse = [
        signal_row(0, far_sign=1, far_intensity=1),
        signal_row(1, far_sign=1, far_intensity=3),
        signal_row(
            2,
            far_sign=1,
            far_intensity=1,
            near_sign=1,
            near_intensity=3,
        ),
    ]
    events, _ = support.relay_candidates(
        reverse,
        control="far_to_near_reverse_relay",
        arm_leg="far",
        confirmation_leg="near",
    )
    assert len(events) == 1
    assert events[0].side == 1


def test_scheduler_allows_adjacency_suppresses_overlap_and_contains_quarter() -> None:
    first_signal = support.GRID_START + support.BAR
    first = candidate(first_signal, position=0)
    overlapping = candidate(
        first_signal + timedelta(hours=5),
        position=60,
    )
    adjacent = candidate(
        first_signal + support.HOLD,
        position=72,
    )
    accepted = support.schedule("primary", [first, overlapping, adjacent])
    assert [row.signal_position for row in accepted] == [0, 72]
    assert accepted[1].entry_time == accepted[0].exit_time

    crossing = candidate(
        datetime(2023, 3, 31, 20, 0, tzinfo=timezone.utc),
        position=25_000,
    )
    assert support.schedule("primary", [crossing]) == []


def test_scheduler_and_clock_validation_reject_post_2023_terminal_signal() -> None:
    terminal = candidate(
        support.GRID_END,
        position=support.GRID_ROWS - 1,
    )
    assert support.schedule("primary", [terminal]) == []

    invalid = support.Scheduled(
        control="primary",
        signal_position=support.GRID_ROWS - 1,
        signal_time=support.GRID_END,
        entry_time=support.GRID_END + support.BAR,
        exit_time=support.GRID_END + support.BAR + support.HOLD,
        quarter="2024-Q1",
        side=1,
        confirmation_age=1,
    )
    assert support._clock_valid("primary", [invalid]) is False


def test_side_controls_reuse_exact_intervals_and_hash_parity() -> None:
    primary = support.schedule(
        "primary",
        [
            candidate(support.GRID_START + support.BAR, position=0),
            candidate(
                support.GRID_START + support.BAR + support.HOLD,
                position=72,
                side=-1,
            ),
        ],
    )
    assert len(primary) == 2
    for control in (
        "deterministic_random_side",
        "exact_direction_flip",
        "constant_long",
        "constant_short",
    ):
        controlled = support.side_control(primary, control=control)
        assert [
            (
                row.signal_position,
                row.signal_time,
                row.entry_time,
                row.exit_time,
                row.quarter,
                row.confirmation_age,
            )
            for row in controlled
        ] == [
            (
                row.signal_position,
                row.signal_time,
                row.entry_time,
                row.exit_time,
                row.quarter,
                row.confirmation_age,
            )
            for row in primary
        ]

    random = support.side_control(
        primary,
        control="deterministic_random_side",
    )
    expected = []
    for row in primary:
        digest = hashlib.sha256(
            (
                "RQHR-72|deterministic_random_side|"
                f"{row.entry_time.isoformat()}"
            ).encode()
        ).digest()
        expected.append(1 if digest[0] % 2 == 0 else -1)
    assert [row.side for row in random] == expected
    assert [row.side for row in support.side_control(
        primary, control="exact_direction_flip"
    )] == [-row.side for row in primary]


def test_stale_controls_drop_incomplete_and_out_of_range_destinations() -> None:
    rows = [
        signal_row(0),
        signal_row(1),
        signal_row(2, complete=False),
    ]
    primary = [
        candidate(rows[1].source.available_at, position=1),
        candidate(rows[2].source.available_at, position=2),
    ]
    assert support.stale_candidates(
        primary,
        rows,
        lag=1,
        control="one_bar_stale",
    ) == []


def test_snapshot_aggregation_and_missing_predicate_are_exact() -> None:
    values = np.array(
        [
            np.arange(10, dtype=float),
            np.ones(10, dtype=float),
        ]
    )
    net, path, efficiency = support.aggregate_snapshot_skews(values)
    assert net.tolist() == [9.0, 0.0]
    assert path.tolist() == [9.0, 0.0]
    assert efficiency.tolist() == [1.0, 0.0]

    rows = support.synthetic_source_rows("missing_rows", rows=1_012)
    incomplete = [row.position for row in rows if not row.complete]
    assert incomplete == [0, 1, 2, 1009, 1010, 1011]


@pytest.mark.parametrize(
    "scenario",
    [
        "smooth_symmetric",
        "tick_rounded_anchor",
        "stepped_asymmetric",
        "discrete_asymmetric_ladder",
    ],
)
def test_synthetic_snapshot_geometry_matches_hash_bound_rncm_source(
    scenario: str,
) -> None:
    positions = np.arange(32, dtype=np.float64)
    average = support._synthetic_average_quotes(scenario, positions)
    inner_bid, inner_ask = average[1]
    if scenario == "discrete_asymmetric_ladder":
        frozen = rncm.synthetic_discrete_ladder_panel(rows=len(positions))
    else:
        frozen = rncm.synthetic_fixed_book_panel(
            rows=len(positions),
            scenario=scenario,
        )
    for distance in range(2, 6):
        bid, ask = average[distance]
        observed = np.log(ask / inner_ask) - np.log(inner_bid / bid)
        np.testing.assert_array_equal(
            observed,
            frozen[f"skew_{distance}_median"].to_numpy(float),
        )


def test_far_tuple_permutation_preserves_near_availability_and_quarter() -> None:
    rows = []
    dates = [
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 1, 2, tzinfo=timezone.utc),
        datetime(2023, 1, 3, tzinfo=timezone.utc),
        datetime(2023, 4, 1, tzinfo=timezone.utc),
        datetime(2023, 4, 2, tzinfo=timezone.utc),
    ]
    for position, date in enumerate(dates):
        complete = position != 2
        rows.append(
            support.SourceRow(
                position=position,
                date=date,
                available_at=date + support.BAR,
                complete=complete,
                net=(D(position + 1), D(position + 2), D(position + 10), D(position + 20)),
                path=(D(position + 1), D(position + 2), D(position + 10), D(position + 20)),
                efficiency=(D(1), D(1), D(1), D(1)),
            )
        )
    permuted = support.permute_far_tuples(rows)
    assert not permuted[2].complete
    assert all(
        current.available_at == original.available_at
        and current.net[:2] == original.net[:2]
        for current, original in zip(permuted, rows, strict=True)
    )
    for quarter in ("2023-Q1", "2023-Q2"):
        original_far = sorted(
            (row.net[2], row.net[3])
            for row in rows
            if row.complete and support._quarter(row.date) == quarter
        )
        permuted_far = sorted(
            (row.net[2], row.net[3])
            for row in permuted
            if row.complete and support._quarter(row.date) == quarter
        )
        assert original_far == permuted_far


def scheduled_clock(control: str, *, immediate: bool) -> list[support.Scheduled]:
    output = []
    for index in range(120):
        entry = support.GRID_START + index * timedelta(days=3) + 2 * support.BAR
        signal = entry - support.BAR
        position = int((signal - support.GRID_START - support.BAR) / support.BAR)
        output.append(
            support.Scheduled(
                control=control,
                signal_position=position,
                signal_time=signal,
                entry_time=entry,
                exit_time=entry + support.HOLD,
                quarter=support._quarter(entry),
                side=1 if index % 2 == 0 else -1,
                confirmation_age=0 if immediate else 2,
            )
        )
    return output


def test_support_gate_passes_exact_well_distributed_clock() -> None:
    immediate = {"simultaneous_near_far", "near_only", "far_only"}
    clocks = {
        control: scheduled_clock(control, immediate=control in immediate)
        for control in support.CONTROL_NAMES
    }
    summary = support.source_support_summary(clocks["primary"], clocks=clocks)
    assert summary["passed"] is True
    assert all(summary["checks"].values())
    assert summary["accepted_events"] == 120
    assert summary["long_events"] == 60
    assert summary["short_events"] == 60


def test_support_gate_rejects_out_of_window_primary_event() -> None:
    immediate = {"simultaneous_near_far", "near_only", "far_only"}
    clocks = {
        control: scheduled_clock(control, immediate=control in immediate)
        for control in support.CONTROL_NAMES
    }
    clocks["primary"].append(
        support.Scheduled(
            control="primary",
            signal_position=support.GRID_ROWS - 1,
            signal_time=support.GRID_END,
            entry_time=support.GRID_END + support.BAR,
            exit_time=support.GRID_END + support.BAR + support.HOLD,
            quarter="2024-Q1",
            side=1,
            confirmation_age=2,
        )
    )
    summary = support.source_support_summary(
        clocks["primary"],
        clocks=clocks,
    )
    assert summary["checks"]["exact_2023_window"] is False
    assert summary["passed"] is False


def valid_comparator_raw() -> dict:
    return {
        "quarter": "q1",
        "signal_position": 0,
        "entry_position": 1,
        "exit_position": 73,
        "signal_date": "2023-01-01 00:00:00",
        "entry_date": "2023-01-01 00:05:00",
        "exit_date": "2023-01-01 06:05:00",
        "side": 1,
        "branch": "test",
        "hold_bars": 72,
    }


def test_comparator_row_and_group_validation_fail_closed() -> None:
    event = support._comparator_event(valid_comparator_raw())
    assert event.entry_time == support.GRID_START + support.BAR

    bad = valid_comparator_raw()
    bad["side"] = 0
    with pytest.raises(ValueError, match="side"):
        support._comparator_event(bad)

    bad = valid_comparator_raw()
    bad["entry_date"] = "2023-01-01 00:10:00"
    with pytest.raises(ValueError, match="position/date"):
        support._comparator_event(bad)

    with pytest.raises(ValueError, match="duplicate"):
        support._validate_comparator_group("x", [event, event])
    overlapping = support.ComparatorEvent(
        entry_time=event.entry_time + support.BAR,
        exit_time=event.exit_time + support.BAR,
        side=-1,
    )
    with pytest.raises(ValueError, match="overlapping"):
        support._validate_comparator_group("x", [event, overlapping])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("protocol", "wrong", "protocol"),
        ("outcomes_opened_for_pdf10", True, "boundary opened"),
        ("event_count", 590, "event count"),
        ("event_clock_sha256", "0" * 64, "clock hash"),
        ("canonical_fields", ["side"], "canonical field"),
        ("serialization", "wrong", "serialization"),
        ("quarter_boundary_policy", "wrong", "quarter policy"),
    ],
)
def test_comparator_header_declarations_fail_closed(
    field: str,
    replacement: object,
    message: str,
) -> None:
    spec = support.COMPARATOR_SPECS[1]
    payload = {
        "protocol": spec["protocol"],
        **spec["closed_flags"],
        "event_count": spec["expected_rows"],
        "event_clock_sha256": spec["clock_hash"],
        "canonical_fields": spec["canonical_fields"],
        "serialization": spec["serialization"],
        "quarter_boundary_policy": spec["quarter_boundary_policy"],
    }
    payload[field] = replacement
    with pytest.raises(ValueError, match=message):
        support._validate_comparator_header(payload, spec)


def test_comparator_rows_read_are_audited_before_header_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "protocol": "wrong",
        "closed": False,
        "event_count": 2,
        "event_clock_sha256": "h",
        "events": [{"row": 1}, {"row": 2}],
    }
    (tmp_path / "clock.json").write_text(json.dumps(payload), encoding="utf-8")
    spec = {
        "group": "test:primary",
        "path": Path("clock.json"),
        "sha256": "unused",
        "producer": Path("producer.py"),
        "producer_sha256": "unused",
        "protocol": "expected",
        "closed_flags": {"closed": False},
        "expected_rows": 2,
        "selection_end_exclusive": None,
        "canonical_fields": None,
        "clock_hash": "h",
        "kind": "embedded_full_dict",
    }
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(support, "COMPARATOR_SPECS", (spec,))
    monkeypatch.setattr(support, "_verify_hash", lambda *_args: "unused")
    with pytest.raises(support.ComparatorValidationError) as exc_info:
        support.load_comparator_groups()
    assert exc_info.value.rows_read == 2
    assert "protocol mismatch" in str(exc_info.value)


def test_pdf_projection_hash_is_exactly_six_fields_and_branch_sensitive() -> None:
    row = valid_comparator_raw()
    projection = support._pdf_projection([row])
    assert tuple(projection[0]) == (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "branch",
        "hold_bars",
    )
    changed = dict(row)
    changed["branch"] = "different"
    assert support.canonical_hash(projection) != support.canonical_hash(
        support._pdf_projection([changed])
    )


def test_common_window_counts_without_clipping() -> None:
    contained = [
        support.ComparatorEvent(
            support.GRID_START + (index + 1) * timedelta(days=2),
            support.GRID_START + (index + 1) * timedelta(days=2) + support.HOLD,
            1,
        )
        for index in range(10)
    ]
    extra = [
        support.ComparatorEvent(
            support.GRID_START - timedelta(days=2),
            support.GRID_START - timedelta(days=1),
            1,
        ),
        support.ComparatorEvent(
            support.GRID_END + timedelta(days=1),
            support.GRID_END + timedelta(days=2),
            1,
        ),
        support.ComparatorEvent(
            support.GRID_END - timedelta(hours=1),
            support.GRID_END + timedelta(hours=1),
            -1,
        ),
    ]
    selected, counts = support._window_filter("x", [*contained, *extra])
    assert selected == contained
    assert counts == {
        "total_raw_rows_parsed": 13,
        "fully_contained_rows_used": 10,
        "rows_before_window": 1,
        "rows_after_window": 1,
        "rows_crossing_boundary": 1,
    }


def test_synthetic_payload_tampering_fails_closed() -> None:
    payload = valid_synthetic_payload()
    payload["synthetic_null"]["scenarios"]["smooth_symmetric"][
        "raw_confirmations"
    ] = 1
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = support.canonical_hash(core)
    with pytest.raises(RuntimeError, match="synthetic scenario failed"):
        support.validate_synthetic_artifact_payload(payload)


def test_minimal_self_hashed_synthetic_payload_is_rejected() -> None:
    payload = valid_synthetic_payload()
    payload["synthetic_null"]["scenarios"]["smooth_symmetric"].pop(
        "race_audit"
    )
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = support.canonical_hash(core)
    with pytest.raises(RuntimeError, match="synthetic scenario failed"):
        support.validate_synthetic_artifact_payload(payload)


def test_fresh_synthetic_replay_must_match_seal_before_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = valid_synthetic_payload()
    replayed = valid_synthetic_payload()
    sealed["synthetic_null"]["scenarios"]["smooth_symmetric"][
        "race_audit"
    ]["arms"] = 1
    core = {key: value for key, value in sealed.items() if key != "manifest_hash"}
    sealed["manifest_hash"] = support.canonical_hash(core)
    called = False

    def forbidden_source() -> object:
        nonlocal called
        called = True
        raise AssertionError("source loader must not run")

    monkeypatch.setattr(
        support,
        "load_registration",
        lambda: {"manifest_hash": "m", "policy_hash": "p"},
    )
    monkeypatch.setattr(support, "validate_frozen_dependencies", lambda: {})
    monkeypatch.setattr(
        support,
        "load_sealed_synthetic_artifact",
        lambda _path: sealed,
    )
    monkeypatch.setattr(support, "build_synthetic_artifact", lambda: replayed)
    with pytest.raises(RuntimeError, match="fresh frozen replay"):
        support.build_support_report(
            source_loader=forbidden_source,
        )
    assert called is False


def test_full_synthetic_artifact_schema_survives_json_round_trip() -> None:
    payload = json.loads(json.dumps(valid_synthetic_payload(), sort_keys=True))
    support.validate_synthetic_artifact_payload(payload)


def test_synthetic_artifact_path_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="path is frozen"):
        support._validate_config(
            support.Config(synthetic_output="results/alternate-null.json")
        )


def test_source_failure_short_circuits_comparator_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support,
        "load_registration",
        lambda: {"manifest_hash": "m", "policy_hash": "p"},
    )
    monkeypatch.setattr(support, "validate_frozen_dependencies", lambda: {})
    synthetic = valid_synthetic_payload()
    monkeypatch.setattr(
        support,
        "load_sealed_synthetic_artifact",
        lambda _path: synthetic,
    )
    monkeypatch.setattr(
        support,
        "build_synthetic_artifact",
        lambda: synthetic,
    )
    source = support.synthetic_source_rows("smooth_symmetric", rows=20)
    comparator_called = False

    def source_loader() -> tuple[list[support.SourceRow], dict]:
        return source, {
            "rows": len(source),
            "complete_rows": len(source),
            "incomplete_rows": 0,
            "rqhr_columns_read": list(support.RQHR_COLUMNS),
            "forbidden_columns_read": [],
            "post_2023_rows_read": 0,
            "market_or_outcome_rows_read": 0,
        }

    def comparator_loader() -> object:
        nonlocal comparator_called
        comparator_called = True
        raise AssertionError("comparator must not be opened")

    report, clocks = support.build_support_report(
        source_loader=source_loader,
        comparator_loader=comparator_loader,
    )
    assert clocks
    assert report["source_support"]["passed"] is False
    assert report["novelty"]["evaluated"] is False
    assert report["comparator_rows_read"] == 0
    assert report["market_or_outcome_rows_read"] == 0
    assert comparator_called is False


def test_support_builder_static_data_access_boundary() -> None:
    path = Path(support.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    load_source = functions["load_real_source"]
    source_calls = [
        node
        for node in ast.walk(load_source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd"
    ]
    assert [node.func.attr for node in source_calls] == ["read_csv"]
    all_pandas_loads = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"pd", "pandas"}
        and node.func.attr.startswith("read_")
    ]
    assert all_pandas_loads == ["read_csv"]
    project_imports = {
        (function.name, node.module)
        for function in functions.values()
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.split(".", 1)[0] == "training"
    }
    assert project_imports == {
        (
            "_replay_pdf_comparator",
            "training",
        )
    }
    assert tuple(support.RQHR_COLUMNS) == (
        "date",
        "skew_2_net",
        "skew_2_path",
        "skew_2_efficiency",
        "skew_3_net",
        "skew_3_path",
        "skew_3_efficiency",
        "skew_4_net",
        "skew_4_path",
        "skew_4_efficiency",
        "skew_5_net",
        "skew_5_path",
        "skew_5_efficiency",
        "source_complete",
        "source_available_at",
    )


def test_atomic_artifact_write_is_no_clobber_and_directory_synced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    synced: list[Path] = []
    monkeypatch.setattr(
        support,
        "_fsync_directory",
        lambda path: synced.append(path),
    )
    path = tmp_path / "out" / "artifact.json"
    payload = b"{}\n"
    assert support._write_or_verify(path, payload) == "created"
    assert support._write_or_verify(path, payload) == "verified_existing"
    assert synced == [tmp_path / "out"]
    with pytest.raises(RuntimeError, match="differs"):
        support._write_or_verify(path, b'{"changed":true}\n')
