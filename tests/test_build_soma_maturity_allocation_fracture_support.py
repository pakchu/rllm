from __future__ import annotations

import csv
import gzip
import io
import json
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import build_soma_maturity_allocation_fracture_support as s

UTC = timezone.utc


def test_preregistration_bindings_validate_before_source_access() -> None:
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert payload["source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False


def test_protocol_guard_uses_exactly_two_git_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*arguments: str) -> SimpleNamespace:
        calls.append(arguments)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(s, "_git_check", fake_git)
    s._assert_protocol_committed()
    assert len(calls) == 2
    assert calls[0][:2] == ("ls-files", "--error-unmatch")
    assert calls[1][:3] == ("diff", "--quiet", "HEAD")


def _synthetic_source(
    count: int = 4,
    *,
    start: date = date(2019, 1, 2),
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    operations: list[dict[str, str]] = []
    details: list[dict[str, str]] = []
    for index in range(count):
        operation_date = start + timedelta(days=index)
        available = datetime.combine(
            operation_date + timedelta(days=1),
            datetime.min.time(),
            tzinfo=UTC,
        )
        operation_id = f"op-{index:04d}"
        submitted = [100 + index % 7, 80 + index % 5, 60 + index % 3]
        accepted = [80 + index % 7, 50 + index % 5, 30 + index % 3]
        available_weights = [
            70 + index % 4,
            65 + index % 6,
            55 + index % 5,
        ]
        operations.append(
            {
                "operation_id": operation_id,
                "operation_date": operation_date.isoformat(),
                "available_at_utc": available.isoformat(),
                "total_par_submitted": str(sum(submitted)),
                "total_par_accepted": str(sum(accepted)),
            }
        )
        for atom, days in enumerate((365, 730, 1_095)):
            maturity = operation_date + timedelta(days=days)
            details.append(
                {
                    "operation_id": operation_id,
                    "operation_date": operation_date.isoformat(),
                    "available_at_utc": available.isoformat(),
                    "cusip": f"C{index:04d}{atom}",
                    "security_description": (
                        f"T 00.000 {maturity.strftime('%m/%d/%y')}"
                    ),
                    "par_submitted": str(submitted[atom]),
                    "par_accepted": str(accepted[atom]),
                    "actual_available_to_borrow": str(
                        available_weights[atom]
                    ),
                }
            )
    return operations, details


def _feature(index: int, value: int) -> s.OperationFeature:
    available = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)
    return s.OperationFeature(
        operation_id=f"feature-{index:04d}",
        operation_date=available.date(),
        available_at=available,
        available_at_text=available.isoformat(),
        primary=Fraction(value),
        submitted_inventory_tilt=Fraction(value),
        submitted_award_tilt=Fraction(value),
        award_inventory_tilt=Fraction(value),
        aggregate_demand_intensity=Fraction(value),
        detail_rows=3,
    )


def _signal(
    *,
    control: str = "primary",
    index: int = 0,
    decision: datetime | None = None,
    operation_date: date | None = None,
    side: str = "LONG",
    tail: str = "LOW",
    segment: int = 0,
) -> s.Signal:
    decision_time = decision or datetime(2020, 2, 1, tzinfo=UTC)
    return s.Signal(
        control=control,
        signal_id=f"{control}-{index}",
        parent_signal_id="",
        operation_id=f"operation-{index}",
        operation_date=operation_date or decision_time.date(),
        decision_time=decision_time,
        segment=segment,
        side=side,
        tail=tail,
    )


def _set_coherent_passing_source(report: dict[str, Any]) -> None:
    operation_counts = {
        "warmup": 126,
        "train": 740,
        "selection": 240,
    }
    operation_counts["full"] = sum(operation_counts.values())
    coverage = {
        window: {
            "description_rows_valid": operation_counts[window],
            "description_rows_total": operation_counts[window],
            "description_parser_coverage": 1.0,
            "complete_operations": operation_counts[window],
            "operation_rows": operation_counts[window],
            "complete_operation_share": 1.0,
            "singleton_batches": operation_counts[window],
            "availability_batches": operation_counts[window],
            "single_operation_batch_share": 1.0,
        }
        for window in ("full", "warmup", "train", "selection")
    }
    rank_selectivity = {
        "train": {
            "rank_ready": 740,
            "LOW": 74,
            "HIGH": 74,
            "LOW_share": 0.1,
            "HIGH_share": 0.1,
        },
        "selection": {
            "rank_ready": 240,
            "LOW": 24,
            "HIGH": 24,
            "LOW_share": 0.1,
            "HIGH_share": 0.1,
        },
    }

    def event_stats(split: str) -> dict[str, Any]:
        years = (
            range(2020, 2023)
            if split == "train"
            else range(2023, 2024)
        )
        rows = []
        index = 0
        for year in years:
            for month in range(1, 13):
                for day in (1, 10):
                    entry = datetime(year, month, day, tzinfo=UTC)
                    rows.append(
                        s.Scheduled(
                            control="primary",
                            signal_id=f"{split}-{index}",
                            parent_signal_id="",
                            operation_id="",
                            operation_date=entry.date(),
                            decision_time=entry,
                            entry_time=entry,
                            exit_time=entry + timedelta(hours=72),
                            segment=0,
                            side="LONG" if index % 2 == 0 else "SHORT",
                            tail="LOW" if index % 2 == 0 else "HIGH",
                            split=split,
                        )
                    )
                    index += 1
        return s._event_stats(rows, split)

    internal = {}
    for control in s.prereg.SOURCE_CONTROL_ORDER[1:]:
        internal[control] = {}
        for split, entries in (("train", 30), ("selection", 10)):
            internal[control][split] = {
                "entries": entries,
                "LONG": entries // 2,
                "SHORT": entries // 2,
                "exact_entry_jaccard": 0.0,
                "same_entry_same_side_reproduction": 0.0,
                "signed_occupancy_pearson": 0.0,
            }
    report["source"] = {
        "operation_rows": operation_counts["full"],
        "detail_rows": operation_counts["full"],
        "valid_description_rows": operation_counts["full"],
        "operation_features": operation_counts["full"],
        "complete_operations": operation_counts["full"],
        "invalid_operations": 0,
        "invalid_reason_counts": {},
        "availability_batches": operation_counts["full"],
        "singleton_batches": operation_counts["full"],
        "invalid_or_multi_operation_batches": 0,
    }
    report["support"] = {
        "coverage": coverage,
        "rank_selectivity": rank_selectivity,
        "primary_event_support": {
            "train": event_stats("train"),
            "selection": event_stats("selection"),
        },
        "internal_component_distinctness": internal,
    }
    report["clock"]["rows_by_control"]["primary"] = 96
    for control in s.prereg.SOURCE_CONTROL_ORDER[1:]:
        report["clock"]["rows_by_control"][control] = 40
    report["clock"]["rows"] = sum(
        report["clock"]["rows_by_control"].values()
    )
    report["source_checks"] = {
        stage: {name: True for name in names}
        for stage, names in s._expected_source_check_keys().items()
    }
    report["source_passed"] = True
    report["novelty_authorized"] = True


def _set_coherent_passing_novelty(report: dict[str, Any]) -> None:
    preregistration = s.prereg.build_manifest()
    report["comparator_failure"] = None
    report["novelty_checks"] = {
        name: True
        for name in s._expected_novelty_check_keys(preregistration)
    }
    novelty: dict[str, dict[str, dict[str, object]]] = {}
    rows_decoded = 0
    metric_evidence = {
        "exact_entry_jaccard": 0.0,
        "same_entry_same_side_reproduction": 0.0,
        "within_24h_matches": 0,
        "candidate_24h_containment": 0.0,
        "comparator_24h_containment": 0.0,
        "signed_occupancy_pearson": 0.0,
    }
    for contract in preregistration["novelty_contract"]["comparators"]:
        comparator_id = str(contract["id"])
        selected_groups = set(contract["selected_groups"])
        novelty[comparator_id] = {}
        for group in contract["allowed_groups"]:
            selected = group in selected_groups
            raw_rows = int(
                contract["minimum_contained_rows_each_group"]
            )
            evidence: dict[str, object] = {
                "selected_for_metrics": selected,
                "raw_rows": raw_rows,
                "contained_rows": raw_rows,
                "before_rows": 0,
                "after_rows": 0,
                "crossing_rows": 0,
            }
            if selected:
                evidence.update(metric_evidence)
            novelty[comparator_id][group] = evidence
            rows_decoded += raw_rows
    report["novelty"] = novelty
    report["comparator_rows_decoded"] = rows_decoded
    report["novelty_passed"] = True
    report["decision"] = "advance_to_economic_evaluator_freeze"
    report["first_failing_stage"] = None
    report["first_failing_check"] = None


def test_source_stream_builds_exact_fraction_features() -> None:
    operations, details = _synthetic_source()
    source = s.build_source(operations, details)
    assert len(source.operations) == 4
    assert len(source.features) == 4
    assert source.detail_rows == 12
    assert source.valid_description_rows == 12
    assert source.complete_operations == 4
    assert source.batches == 4
    assert source.singleton_batches == 4
    assert source.invalid_reason_counts == {}
    first = source.features[0]
    assert first.valid is True
    assert isinstance(first.primary, Fraction)
    assert isinstance(first.submitted_inventory_tilt, Fraction)
    assert isinstance(first.submitted_award_tilt, Fraction)
    assert first.primary == (
        first.submitted_inventory_tilt
        + first.submitted_award_tilt
    )


def test_source_attributable_failures_invalidate_whole_operation() -> None:
    operations, details = _synthetic_source(count=1)
    broken = [dict(row) for row in details]
    broken[0]["operation_date"] = "2019-01-03"
    source = s.build_source(operations, broken)
    assert source.complete_operations == 0
    assert source.features[0].valid is False
    assert source.invalid_reason_counts == {
        "detail_join_value_mismatch": 1,
    }

    duplicate = details + [dict(details[0])]
    with pytest.raises(s.SourceContractFailure, match="duplicate"):
        s.build_source(operations, duplicate)

    unjoined = [dict(row) for row in details]
    unjoined[0]["operation_id"] = "unknown-operation"
    with pytest.raises(s.SourceContractFailure, match="does not join"):
        s.build_source(operations, unjoined)

    malformed_timestamp = [dict(operations[0])]
    malformed_timestamp[0]["available_at_utc"] = "not-a-timestamp"
    with pytest.raises(
        s.SourceContractFailure,
        match="noncanonical timestamp",
    ):
        s.build_source(malformed_timestamp, details)

    broken = [dict(row) for row in details]
    broken[0]["security_description"] = "T BAD"
    source = s.build_source(operations, broken)
    assert source.valid_description_rows == 2
    assert source.complete_operations == 0
    assert source.invalid_reason_counts == {
        "security_description_parser": 1,
    }

    broken_operations = [dict(operations[0])]
    broken_operations[0]["total_par_submitted"] = "999"
    source = s.build_source(broken_operations, details)
    assert source.complete_operations == 0
    assert source.invalid_reason_counts == {
        "operation_total_reconciliation": 1,
    }


def test_multiple_attributable_failures_accumulate_and_reset_batches() -> None:
    operations, original_details = _synthetic_source(count=7)
    details = [dict(row) for row in original_details]
    details[0]["operation_date"] = "2019-01-03"
    details[1]["security_description"] = "T BAD"
    details[3]["par_submitted"] = "1e3"
    operations[2]["total_par_submitted"] = "999"
    for detail in details[9:12]:
        detail["actual_available_to_borrow"] = "0"
    operations[4]["total_par_accepted"] = "1e3"

    source = s.build_source(operations, details)

    assert source.detail_rows == 21
    assert source.valid_description_rows == 20
    assert source.complete_operations == 2
    assert source.singleton_batches == 2
    assert source.invalid_reason_counts == {
        "detail_join_value_mismatch": 1,
        "invalid_detail_par_submitted": 1,
        "invalid_operation_total_accepted": 1,
        "nonpositive_centroid_weight": 1,
        "operation_total_reconciliation": 1,
        "security_description_parser": 1,
    }
    assert [feature.valid for feature in source.features] == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    raw, audit, segmented = s.build_raw_signals(source.features)
    assert all(not rows for rows in raw.values())
    assert audit["train"]["rank_ready"] == 0
    assert [feature.segment for feature in segmented[:5]] == [
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    assert segmented[5].segment == segmented[6].segment == 5

    report, _ = s.build_support_from_rows(operations, details)
    assert report["source_passed"] is False
    assert report["first_failing_stage"] == (
        "schema_join_uniqueness_reconciliation"
    )
    assert report["source"]["invalid_operations"] == 5
    assert report["source"]["invalid_or_multi_operation_batches"] == 5
    assert report["source"]["invalid_reason_counts"] == (
        source.invalid_reason_counts
    )
    assert report["support"]["coverage"]["warmup"] == {
        "description_rows_valid": 20,
        "description_rows_total": 21,
        "description_parser_coverage": 20 / 21,
        "complete_operations": 2,
        "operation_rows": 7,
        "complete_operation_share": 2 / 7,
        "singleton_batches": 2,
        "availability_batches": 7,
        "single_operation_batch_share": 2 / 7,
    }
    assert report["comparator_rows_decoded"] == 0


def test_rank_midpoint_tail_and_first_ready_baseline() -> None:
    features = [_feature(index, index) for index in range(126)]
    features.extend(
        [
            _feature(126, -100),
            _feature(127, 60),
            _feature(128, -100),
            _feature(129, 1_000),
        ]
    )
    raw, audit, segmented = s.build_raw_signals(features)
    assert len(segmented) == 130
    assert audit["train"] == {
        "rank_ready": 4,
        "LOW": 2,
        "HIGH": 1,
    }
    for control in s.prereg.SOURCE_CONTROL_ORDER:
        assert [item.tail for item in raw[control]] == ["LOW", "HIGH"]
        assert [item.side for item in raw[control]] == ["LONG", "SHORT"]


def test_multi_operation_batch_resets_history() -> None:
    features = [_feature(index, index) for index in range(126)]
    duplicate_time = features[-1].available_at + timedelta(days=1)
    first = _feature(126, -100)
    second = _feature(127, 1_000)
    first = s.replace(first, available_at=duplicate_time)
    second = s.replace(second, available_at=duplicate_time)
    raw, audit, segmented = s.build_raw_signals(
        [*features, first, second]
    )
    assert all(not rows for rows in raw.values())
    assert audit["train"]["rank_ready"] == 0
    assert segmented[-1].segment == -1
    assert segmented[-2].segment == -1


def test_global_reservation_precedes_split_containment() -> None:
    crossing = _signal(
        index=1,
        decision=datetime(2019, 12, 31, 23, 55, tzinfo=UTC),
        operation_date=date(2019, 12, 31),
    )
    overlapped = _signal(
        index=2,
        decision=datetime(2020, 1, 1, 0, 10, tzinfo=UTC),
        operation_date=date(2020, 1, 1),
    )
    accepted, retained, reasons = s.schedule_signals(
        "primary",
        [crossing, overlapped],
    )
    assert len(accepted) == 1
    assert retained == []
    assert reasons == {
        "outside_or_crossing_split": 1,
        "overlap_suppressed": 1,
    }


def test_primary_side_and_delay_controls_are_parent_bound() -> None:
    parent_signal = _signal(
        index=1,
        decision=datetime(2020, 2, 1, tzinfo=UTC),
    )
    primary_all, primary_retained, _ = s.schedule_signals(
        "primary",
        [parent_signal],
    )
    parent_feature = _feature(0, 0)
    parent_feature = s.replace(
        parent_feature,
        operation_id=parent_signal.operation_id,
        available_at=parent_signal.decision_time,
        operation_date=parent_signal.operation_date,
        segment=0,
    )
    successor = _feature(1, 1)
    successor = s.replace(
        successor,
        available_at=datetime(2020, 2, 2, tzinfo=UTC),
        operation_date=date(2020, 2, 2),
        segment=0,
    )
    controls, diagnostics = s._derive_primary_side_controls(
        primary_all,
        primary_retained,
        [parent_feature, successor],
    )
    for control in s.prereg.OUTCOME_CONTROL_ORDER:
        assert controls[control]
        assert controls[control][0].parent_signal_id == parent_signal.signal_id
    assert controls["exact_direction_flip"][0].side == "SHORT"
    assert diagnostics["one_operation_delay"][
        "missing_same_segment_successor"
    ] == 0


def test_random_side_uses_preregistered_independent_digest() -> None:
    digest = s.hashlib.sha256(
        b"SMAF-72|parent-0|RANDOM_SIDE"
    ).hexdigest()
    assert digest == (
        "a4f2ae667345ff14901801a7a3bcaceb"
        "d56aa9941ace4966c0df83e160618331"
    )
    assert s._random_side("parent-0") == "SHORT"
    assert s._side_control_id(
        "deterministic_random_side",
        "parent-0",
    ) != digest


def test_deterministic_clock_is_canonical_symbolic_only() -> None:
    signal = _signal()
    _, primary, _ = s.schedule_signals("primary", [signal])
    clocks = {control: [] for control in s.CONTROL_ORDER}
    clocks["primary"] = primary
    first = s.deterministic_clock_bytes(clocks)
    second = s.deterministic_clock_bytes(clocks)
    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(handle, encoding="utf-8")
            )
        )
    assert rows[0]["control"] == "primary"
    assert list(rows[0]) == list(s.CLOCK_COLUMNS)
    header = ",".join(rows[0]).lower()
    assert all(token not in header for token in s.FORBIDDEN_CLOCK_TOKENS)


def test_synthetic_support_never_opens_comparator_or_outcomes() -> None:
    operations, details = _synthetic_source(count=10)
    report, clock = s.build_support_from_rows(operations, details)
    assert report["source_passed"] is False
    assert report["novelty_authorized"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["outcomes_opened"] is False
    assert report["economic_evaluator_authorized"] is False
    assert report["outcome_boundary"] == {
        "btc_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "forward_return_rows_loaded": 0,
        "pnl_cagr_mdd_opened": False,
        "network_calls": 0,
        "protocol_git_subprocess_calls": 0,
        "model_or_gpu_calls": 0,
    }
    assert report["clock"]["sha256"] == s.hashlib.sha256(clock).hexdigest()
    assert tuple(report["source_checks"]) == (
        "frozen_identity_and_exact_header",
        "schema_join_uniqueness_reconciliation",
        "parser_coverage_and_complete_operations",
        "singleton_causal_batches",
        "rank_coverage_and_tail_selectivity",
        "primary_event_support",
        "internal_component_distinctness",
    )
    s.validate_report(report)


def test_source_contract_failure_serializes_terminal_empty_report() -> None:
    clocks = {control: [] for control in s.CONTROL_ORDER}
    clock = s.deterministic_clock_bytes(clocks)
    error = s.SourceContractFailure(
        "parser_coverage_and_complete_operations",
        "security_description_parser",
        17,
        "bad description",
    )
    report = s.build_source_failure_report(
        error=error,
        clock_bytes=clock,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        protocol_git_subprocess_calls=2,
    )
    assert report["source_passed"] is False
    assert report["first_failing_stage"] == (
        "parser_coverage_and_complete_operations"
    )
    assert report["source"]["decoded_rows_before_failure"] == 17
    assert report["clock"]["rows"] == 0
    assert report["comparator_rows_decoded"] == 0
    assert report["decision"] == (
        "retire_SMAF_72_unchanged_before_outcomes"
    )
    s.validate_report(report)


def test_global_source_failure_preserves_decoded_row_progress() -> None:
    operations, details = _synthetic_source(count=2)
    operations[1]["available_at_utc"] = "not-a-timestamp"
    with pytest.raises(s.SourceContractFailure) as captured:
        s.build_source(operations, details)
    assert captured.value.rows_decoded == 2

    clocks = {control: [] for control in s.CONTROL_ORDER}
    clock = s.deterministic_clock_bytes(clocks)
    report = s.build_source_failure_report(
        error=captured.value,
        clock_bytes=clock,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        protocol_git_subprocess_calls=0,
    )
    assert report["source"]["decoded_rows_before_failure"] == 2
    s.validate_report(report)


def test_source_failure_short_circuits_comparator_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, details = _synthetic_source(count=10)
    source = s.build_source(operations, details)
    clocks, rank_audit, _, diagnostics = s.build_clocks(source.features)
    clock = s.deterministic_clock_bytes(clocks)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("comparator accessed after source failure")

    monkeypatch.setattr(s, "evaluate_novelty", forbidden)
    report = s.build_report(
        source=source,
        clocks=clocks,
        rank_audit=rank_audit,
        schedule_diagnostics=diagnostics,
        preregistration=s.prereg.build_manifest(),
        clock_bytes=clock,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        protocol_git_subprocess_calls=0,
        comparator_loader=True,
    )
    assert report["source_passed"] is False
    assert report["comparator_rows_decoded"] == 0


def test_exact_24h_match_and_occupancy_metrics() -> None:
    candidate_signals = [
        _signal(
            index=index,
            decision=datetime(2020, 2, 1, tzinfo=UTC)
            + timedelta(days=4 * index),
        )
        for index in range(3)
    ]
    _, candidate, _ = s.schedule_signals("primary", candidate_signals)
    comparator_signals = [
        _signal(
            control="submitted_inventory_tilt",
            index=index,
            decision=item.decision_time + timedelta(hours=23),
        )
        for index, item in enumerate(candidate_signals)
    ]
    _, comparator, _ = s.schedule_signals(
        "submitted_inventory_tilt",
        comparator_signals,
    )
    assert s._within_24h_matches(candidate, comparator) == 3
    assert s._entry_jaccard(candidate, comparator) == 0.0
    correlation = s._occupancy_correlation(
        candidate,
        comparator,
        s.COMMON_START,
        s.TRAIN_END,
    )
    assert correlation is not None
    assert -1.0 <= correlation <= 1.0


def test_comparator_validates_full_vocabulary_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clock.csv.gz"
    allowed = ("primary", "unselected")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["control", "entry_time", "exit_time", "side"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "control": "primary",
                "entry_time": "2021-01-01T00:00:00Z",
                "exit_time": "2021-01-04T00:00:00Z",
                "side": "LONG",
            }
        )
        writer.writerow(
            {
                "control": "unselected",
                "entry_time": "2021-02-01T00:00:00Z",
                "exit_time": "2021-02-04T00:00:00Z",
                "side": "INVALID",
            }
        )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s.prereg, "REPOSITORY_ROOT", tmp_path)
    contract = {
        "id": "TEST",
        "path": path.name,
        "read_csv": {
            "usecols": [
                "control",
                "entry_time",
                "exit_time",
                "side",
            ]
        },
        "allowed_groups": list(allowed),
        "selected_groups": ["primary"],
        "side_map": {"LONG": "LONG", "SHORT": "SHORT"},
        "minimum_contained_rows_each_group": 1,
    }
    with pytest.raises(
        s.ComparatorContractFailure,
        match="unknown side",
    ) as captured:
        s.read_comparator(contract)
    assert captured.value.rows_decoded == 2


def test_comparator_rejects_raw_group_ordering_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clock.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["control", "entry_time", "exit_time", "side"],
            lineterminator="\n",
        )
        writer.writeheader()
        for entry, exit_time in (
            ("2021-01-04T00:00:00Z", "2021-01-07T00:00:00Z"),
            ("2021-01-01T00:00:00Z", "2021-01-04T00:00:00Z"),
        ):
            writer.writerow(
                {
                    "control": "primary",
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side": "LONG",
                }
            )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s.prereg, "REPOSITORY_ROOT", tmp_path)
    contract = {
        "id": "TEST",
        "path": path.name,
        "read_csv": {
            "usecols": [
                "control",
                "entry_time",
                "exit_time",
                "side",
            ]
        },
        "allowed_groups": ["primary"],
        "selected_groups": ["primary"],
        "side_map": {"LONG": "LONG", "SHORT": "SHORT"},
        "minimum_contained_rows_each_group": 1,
    }
    with pytest.raises(
        s.ComparatorContractFailure,
        match="ordering regression",
    ) as captured:
        s.read_comparator(contract)
    assert captured.value.code == "comparator_ordering"
    assert captured.value.rows_decoded == 2


def test_novelty_reports_containment_for_every_comparator_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = _signal(
        decision=datetime(2021, 6, 1, tzinfo=UTC),
        operation_date=date(2021, 6, 1),
    )
    _, candidate, _ = s.schedule_signals("primary", [signal])
    assert len(candidate) == 1

    def comparator_row(
        group: str,
        entry: datetime,
        exit_time: datetime,
    ) -> s.Scheduled:
        return s.replace(
            candidate[0],
            control=group,
            entry_time=entry,
            exit_time=exit_time,
        )

    selected = [
        comparator_row(
            "selected",
            datetime(2019, 12, 20, tzinfo=UTC),
            datetime(2019, 12, 23, tzinfo=UTC),
        ),
        comparator_row(
            "selected",
            datetime(2019, 12, 31, tzinfo=UTC),
            datetime(2020, 1, 2, tzinfo=UTC),
        ),
        comparator_row(
            "selected",
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 4, tzinfo=UTC),
        ),
        comparator_row(
            "selected",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 4, tzinfo=UTC),
        ),
    ]
    unselected = [
        comparator_row(
            "unselected",
            datetime(2022, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 4, tzinfo=UTC),
        ),
        comparator_row(
            "unselected",
            datetime(2023, 12, 31, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ),
    ]
    monkeypatch.setattr(
        s,
        "read_comparator",
        lambda contract: (
            {"selected": selected, "unselected": unselected},
            6,
        ),
    )
    monkeypatch.setattr(s, "_entry_jaccard", lambda left, right: 0.0)
    monkeypatch.setattr(
        s,
        "_exact_entry_same_side",
        lambda left, right: 0.0,
    )
    monkeypatch.setattr(
        s,
        "_within_24h_matches",
        lambda left, right: 0,
    )
    monkeypatch.setattr(
        s,
        "_occupancy_correlation",
        lambda left, right, start, end: 0.0,
    )
    comparator_contract = {
        "id": "TEST",
        "allowed_groups": ["selected", "unselected"],
        "selected_groups": ["selected"],
        "minimum_contained_rows_each_group": 1,
    }
    preregistration = {
        "novelty_contract": {
            "thresholds_each_group": {
                "exact_entry_jaccard_max": 0.20,
                "same_entry_same_side_reproduction_max": 0.30,
                "candidate_24h_containment_max": 0.40,
                "comparator_24h_containment_max": 0.40,
                "absolute_signed_occupancy_pearson_max": 0.35,
            },
            "comparators": [comparator_contract],
        }
    }

    report, checks, rows_decoded = s.evaluate_novelty(
        candidate,
        preregistration,
    )

    assert rows_decoded == 6
    assert report["TEST"]["selected"]["selected_for_metrics"] is True
    assert {
        key: report["TEST"]["selected"][key]
        for key in (
            "raw_rows",
            "contained_rows",
            "before_rows",
            "after_rows",
            "crossing_rows",
        )
    } == {
        "raw_rows": 4,
        "contained_rows": 1,
        "before_rows": 1,
        "after_rows": 1,
        "crossing_rows": 1,
    }
    assert report["TEST"]["unselected"] == {
        "selected_for_metrics": False,
        "raw_rows": 2,
        "contained_rows": 1,
        "before_rows": 0,
        "after_rows": 0,
        "crossing_rows": 1,
    }
    assert checks
    assert all(":selected:" in name for name in checks)


def test_report_tampering_fails_closed() -> None:
    operations, details = _synthetic_source(count=10)
    report, _ = s.build_support_from_rows(operations, details)
    report["outcome_boundary"]["btc_market_rows_loaded"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        s.validate_report(report)


def test_rehashed_contradictory_advance_decision_fails_closed() -> None:
    operations, details = _synthetic_source(count=10)
    report, _ = s.build_support_from_rows(operations, details)
    assert report["source_passed"] is False
    report["decision"] = "advance_to_economic_evaluator_freeze"
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    report["manifest_hash"] = s.canonical_hash(core)
    with pytest.raises(RuntimeError, match="decision/check mismatch"):
        s.validate_report(report)


def test_rehashed_fabricated_check_maps_and_evidence_fail_closed() -> None:
    operations, details = _synthetic_source(count=10)
    base, _ = s.build_support_from_rows(operations, details)

    def advance_report() -> dict[str, Any]:
        report = json.loads(json.dumps(base))
        report.update(
            {
                "source_passed": True,
                "novelty_authorized": True,
                "comparator_rows_decoded": 0,
                "comparator_failure": None,
                "novelty": {},
                "novelty_passed": True,
                "decision": "advance_to_economic_evaluator_freeze",
                "first_failing_stage": None,
                "first_failing_check": None,
            }
        )
        return report

    def rehash(report: dict[str, Any]) -> None:
        core = {
            key: value
            for key, value in report.items()
            if key != "manifest_hash"
        }
        report["manifest_hash"] = s.canonical_hash(core)

    fabricated_source = advance_report()
    fabricated_source["source_checks"] = {
        stage: {"fabricated": True}
        for stage in s.SOURCE_STAGE_ORDER
    }
    fabricated_source["novelty_checks"] = {"fabricated": True}
    rehash(fabricated_source)
    with pytest.raises(RuntimeError, match="check schema mismatch"):
        s.validate_report(fabricated_source)

    fabricated_novelty = advance_report()
    _set_coherent_passing_source(fabricated_novelty)
    fabricated_novelty["novelty_checks"] = {"fabricated": True}
    rehash(fabricated_novelty)
    with pytest.raises(RuntimeError, match="novelty check schema"):
        s.validate_report(fabricated_novelty)

    empty_evidence = advance_report()
    _set_coherent_passing_source(empty_evidence)
    empty_evidence["novelty_checks"] = {
        name: True
        for name in s._expected_novelty_check_keys(
            s.prereg.build_manifest()
        )
    }
    rehash(empty_evidence)
    with pytest.raises(RuntimeError, match="comparator evidence schema"):
        s.validate_report(empty_evidence)


def test_complete_advance_report_schema_validates() -> None:
    operations, details = _synthetic_source(count=10)
    report, _ = s.build_support_from_rows(operations, details)
    _set_coherent_passing_source(report)
    _set_coherent_passing_novelty(report)
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    report["manifest_hash"] = s.canonical_hash(core)

    s.validate_report(report)


def test_rehashed_metric_check_contradictions_fail_closed() -> None:
    operations, details = _synthetic_source(count=10)

    source_report, _ = s.build_support_from_rows(operations, details)
    _set_coherent_passing_source(source_report)
    _set_coherent_passing_novelty(source_report)
    source_report["support"]["rank_selectivity"]["train"] = {
        "rank_ready": 0,
        "LOW": 0,
        "HIGH": 0,
        "LOW_share": None,
        "HIGH_share": None,
    }
    source_core = {
        key: value
        for key, value in source_report.items()
        if key != "manifest_hash"
    }
    source_report["manifest_hash"] = s.canonical_hash(source_core)
    with pytest.raises(RuntimeError, match="source metric/check"):
        s.validate_report(source_report)

    novelty_report, _ = s.build_support_from_rows(operations, details)
    _set_coherent_passing_source(novelty_report)
    _set_coherent_passing_novelty(novelty_report)
    first_contract = s.prereg.build_manifest()["novelty_contract"][
        "comparators"
    ][0]
    first_group = first_contract["selected_groups"][0]
    novelty_report["novelty"][first_contract["id"]][first_group][
        "exact_entry_jaccard"
    ] = 1.0
    novelty_core = {
        key: value
        for key, value in novelty_report.items()
        if key != "manifest_hash"
    }
    novelty_report["manifest_hash"] = s.canonical_hash(novelty_core)
    with pytest.raises(RuntimeError, match="novelty metric/check"):
        s.validate_report(novelty_report)

    internal_report, _ = s.build_support_from_rows(operations, details)
    _set_coherent_passing_source(internal_report)
    _set_coherent_passing_novelty(internal_report)
    first_control = s.prereg.SOURCE_CONTROL_ORDER[1]
    internal_report["support"]["internal_component_distinctness"][
        first_control
    ]["train"]["exact_entry_jaccard"] = -0.1
    internal_core = {
        key: value
        for key, value in internal_report.items()
        if key != "manifest_hash"
    }
    internal_report["manifest_hash"] = s.canonical_hash(internal_core)
    with pytest.raises(RuntimeError, match="internal metric outside range"):
        s.validate_report(internal_report)

    negative_novelty, _ = s.build_support_from_rows(operations, details)
    _set_coherent_passing_source(negative_novelty)
    _set_coherent_passing_novelty(negative_novelty)
    negative_novelty["novelty"][first_contract["id"]][first_group][
        "same_entry_same_side_reproduction"
    ] = -0.1
    negative_core = {
        key: value
        for key, value in negative_novelty.items()
        if key != "manifest_hash"
    }
    negative_novelty["manifest_hash"] = s.canonical_hash(negative_core)
    with pytest.raises(
        RuntimeError,
        match="comparator metric outside range",
    ):
        s.validate_report(negative_novelty)


def test_month_concentration_gate_uses_integer_counts() -> None:
    rows = []
    month_starts = [
        datetime(2020 + index // 12, index % 12 + 1, 1, tzinfo=UTC)
        for index in range(18)
    ]
    for index, month_start in enumerate(month_starts):
        repetitions = 3 if index == 0 else 1
        for repetition in range(repetitions):
            entry = month_start + timedelta(days=3 * repetition)
            rows.append(
                s.Scheduled(
                    control="primary",
                    signal_id=f"event-{index}-{repetition}",
                    parent_signal_id="",
                    operation_id="",
                    operation_date=entry.date(),
                    decision_time=entry,
                    entry_time=entry,
                    exit_time=entry + timedelta(hours=72),
                    segment=0,
                    side="LONG" if len(rows) % 2 == 0 else "SHORT",
                    tail="LOW",
                    split="train",
                )
            )
    boundary = s._event_stats(rows, "train")
    assert boundary["events"] == 20
    assert s._event_checks(boundary, "train")[
        "train:maximum_month_share"
    ] is True

    extra = s.replace(
        rows[0],
        signal_id="event-extra",
        entry_time=datetime(2020, 1, 20, tzinfo=UTC),
        exit_time=datetime(2020, 1, 23, tzinfo=UTC),
    )
    above = s._event_stats([*rows, extra], "train")
    above["maximum_month_share"] = 0.0
    assert s._event_checks(above, "train")[
        "train:maximum_month_share"
    ] is False


def test_rehashed_impossible_count_and_domain_evidence_fails_closed() -> None:
    operations, details = _synthetic_source(count=10)

    def fresh() -> dict[str, Any]:
        report, _ = s.build_support_from_rows(operations, details)
        _set_coherent_passing_source(report)
        _set_coherent_passing_novelty(report)
        return report

    def rehash(report: dict[str, Any]) -> None:
        core = {
            key: value
            for key, value in report.items()
            if key != "manifest_hash"
        }
        report["manifest_hash"] = s.canonical_hash(core)

    rank_report = fresh()
    rank_report["support"]["rank_selectivity"]["train"].update(
        {
            "rank_ready": 741,
            "LOW": 74,
            "HIGH": 74,
            "LOW_share": 74 / 741,
            "HIGH_share": 74 / 741,
        }
    )
    rehash(rank_report)
    with pytest.raises(RuntimeError, match="rank tail count"):
        s.validate_report(rank_report)

    clock_report = fresh()
    first_control = s.prereg.SOURCE_CONTROL_ORDER[1]
    clock_report["clock"]["rows_by_control"][first_control] = 39
    clock_report["clock"]["rows"] -= 1
    rehash(clock_report)
    with pytest.raises(RuntimeError, match="internal clock/event"):
        s.validate_report(clock_report)

    subperiod_report = fresh()
    for label in subperiod_report["support"]["primary_event_support"][
        "train"
    ]["subperiod_counts"]:
        subperiod_report["support"]["primary_event_support"]["train"][
            "subperiod_counts"
        ][label] = 1_000_000
    rehash(subperiod_report)
    with pytest.raises(RuntimeError, match="subperiod count invalid"):
        s.validate_report(subperiod_report)

    gap_report = fresh()
    gap_report["support"]["primary_event_support"]["train"][
        "maximum_elapsed_entry_gap_days"
    ] = -1.0
    rehash(gap_report)
    with pytest.raises(RuntimeError, match="gap evidence invalid"):
        s.validate_report(gap_report)

    run_report = fresh()
    run_report["support"]["primary_event_support"]["train"][
        "maximum_same_side_run"
    ] = 0
    rehash(run_report)
    with pytest.raises(RuntimeError, match="same-side run invalid"):
        s.validate_report(run_report)

    boolean_report = fresh()
    contract = s.prereg.build_manifest()["novelty_contract"]["comparators"][0]
    selected_group = contract["selected_groups"][0]
    boolean_report["novelty"][contract["id"]][selected_group][
        "exact_entry_jaccard"
    ] = False
    rehash(boolean_report)
    with pytest.raises(RuntimeError, match="is not numeric"):
        s.validate_report(boolean_report)

    source_ratio_report = fresh()
    source_ratio_report["support"]["coverage"]["full"][
        "complete_operation_share"
    ] = True
    rehash(source_ratio_report)
    with pytest.raises(RuntimeError, match="is not numeric"):
        s.validate_report(source_ratio_report)

    comparator_ratio_report = fresh()
    comparator_ratio_report["novelty"][contract["id"]][selected_group][
        "candidate_24h_containment"
    ] = False
    rehash(comparator_ratio_report)
    with pytest.raises(RuntimeError, match="is not numeric"):
        s.validate_report(comparator_ratio_report)

    source_schema_report = fresh()
    source_schema_report["source"]["fabricated"] = True
    rehash(source_schema_report)
    with pytest.raises(RuntimeError, match="source evidence schema"):
        s.validate_report(source_schema_report)

    support_schema_report = fresh()
    support_schema_report["support"]["fabricated"] = True
    rehash(support_schema_report)
    with pytest.raises(RuntimeError, match="support evidence schema"):
        s.validate_report(support_schema_report)


def test_rehashed_first_failure_mismatch_fails_closed() -> None:
    operations, details = _synthetic_source(count=10)
    report, _ = s.build_support_from_rows(operations, details)
    report["first_failing_check"] = "fabricated_failure"
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    report["manifest_hash"] = s.canonical_hash(core)
    with pytest.raises(RuntimeError, match="first failure evidence"):
        s.validate_report(report)


def test_write_once_is_confined_idempotent_and_cleans_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    path = Path("results/output.json")
    payload = json.dumps({"ok": True}, sort_keys=True).encode()
    assert s._write_once(path, payload) == "created"
    assert s._write_once(path, payload) == "verified_existing"
    output = tmp_path / path
    output.chmod(0o644)
    output.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once(path, payload)
    with pytest.raises(RuntimeError, match="repository-relative"):
        s._write_once("../escape", payload)

    output.unlink()

    def fail_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("forced publication failure")

    monkeypatch.setattr(s, "_publish_temporary", fail_link)
    with pytest.raises(OSError, match="forced publication failure"):
        s._write_once(path, payload)
    assert list(results.iterdir()) == []


def test_write_once_rejects_symlink_parent_and_cleans_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "results").symlink_to(external, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent path is unsafe"):
        s._write_once("results/output.json", b"payload")
    assert list(external.iterdir()) == []

    (tmp_path / "results").unlink()
    results = tmp_path / "results"
    results.mkdir()
    real_fsync = s.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("forced fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(s.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="forced fsync failure"):
        s._write_once("results/output.json", b"payload")
    assert list(results.iterdir()) == []
