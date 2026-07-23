from __future__ import annotations

from decimal import Decimal
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from training import build_soma_collateral_allocation_fracture_support as s


def _source_frames(
    *,
    start: str = "2020-01-02T00:00:00Z",
    batches: int = 12,
    simultaneous_at: int | None = None,
    zero_available_at: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    operation_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    for batch in range(batches):
        timestamp = pd.Timestamp(start) + pd.Timedelta(days=batch)
        operation_count = 2 if simultaneous_at == batch else 1
        for operation_index in range(operation_count):
            operation_id = f"op-{batch:04d}-{operation_index}"
            submitted_values = [
                100 + ((batch + operation_index) % 5) * 7,
                80 + ((2 * batch + operation_index) % 7) * 5,
                60 + ((3 * batch + operation_index) % 4) * 9,
                40 + ((5 * batch + operation_index) % 6) * 4,
            ]
            accepted_values = [
                max(1, value - 20 - ((batch + atom) % 3) * 5)
                for atom, value in enumerate(submitted_values)
            ]
            available_values = [
                0
                if zero_available_at == batch
                else 70 + ((batch + 2 * atom + operation_index) % 8) * 6
                for atom in range(4)
            ]
            fees = [
                Decimal("0.01")
                + Decimal(batch % 5) * Decimal("0.001")
                + Decimal(atom) * Decimal("0.002")
                for atom in range(4)
            ]
            operation_rows.append(
                {
                    "operation_id": operation_id,
                    "operation_date": timestamp.strftime("%Y-%m-%d"),
                    "available_at_utc": timestamp.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "total_par_submitted": str(sum(submitted_values)),
                    "total_par_accepted": str(sum(accepted_values)),
                }
            )
            for atom in range(4):
                detail_rows.append(
                    {
                        "operation_id": operation_id,
                        "operation_date": timestamp.strftime("%Y-%m-%d"),
                        "available_at_utc": timestamp.strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "cusip": f"C{operation_index}{atom:02d}",
                        "par_submitted": str(submitted_values[atom]),
                        "par_accepted": str(accepted_values[atom]),
                        "weighted_average_rate": str(fees[atom]),
                        "actual_available_to_borrow": str(
                            available_values[atom]
                        ),
                    }
                )
    return (
        pd.DataFrame(operation_rows, columns=s.prereg.OPERATIONS_ALLOWLIST),
        pd.DataFrame(detail_rows, columns=s.prereg.DETAILS_ALLOWLIST),
    )


def _validated_frames(**kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    operations, details = _source_frames(**kwargs)
    validated_operations = s.validate_operations(operations)
    validated_details = s.validate_details(details)
    s.reconcile_source(validated_operations, validated_details)
    return validated_operations, validated_details


def test_source_validation_and_reconciliation_are_exact() -> None:
    operations, details = _validated_frames(batches=3)
    assert len(operations) == 3
    assert len(details) == 12
    broken = details.copy()
    broken.loc[0, "par_submitted"] = Decimal("999")
    with pytest.raises(RuntimeError, match="submitted total"):
        s.reconcile_source(operations, broken)


def test_detail_acceptance_and_fee_contract_fail_closed() -> None:
    operations, details = _source_frames(batches=1)
    details.loc[0, "par_accepted"] = "101"
    details.loc[0, "par_submitted"] = "100"
    with pytest.raises(RuntimeError, match="accepted exceeds"):
        s.validate_details(details)

    _, details = _source_frames(batches=1)
    details.loc[0, "weighted_average_rate"] = ""
    with pytest.raises(RuntimeError, match="positive award has no fee"):
        s.validate_details(details)


def test_noncanonical_decimal_and_fractional_timestamp_reject() -> None:
    operations, _ = _source_frames(batches=1)
    operations.loc[0, "total_par_submitted"] = "01"
    with pytest.raises(RuntimeError, match="noncanonical decimal"):
        s.validate_operations(operations)
    operations, _ = _source_frames(batches=1)
    operations.loc[0, "available_at_utc"] = "2020-01-02T00:00:00.1Z"
    with pytest.raises(RuntimeError, match="noncanonical UTC timestamp"):
        s.validate_operations(operations)


def test_jsd_and_unmet_mass_are_bounded_and_quantized() -> None:
    assert s.jsd([0.5, 0.5], [0.5, 0.5]) == Decimal("0E-12")
    separated = s.jsd([1.0, 0.0], [0.0, 1.0])
    assert separated == Decimal("1.000000000000")
    unmet = s._unmet_mass(
        [Decimal("1"), Decimal("2")],
        [Decimal("0"), Decimal("1")],
    )
    assert unmet == Decimal("0.333333333333")


def test_causal_batches_group_simultaneous_operations_without_merging() -> None:
    operations, details = _validated_frames(
        batches=4,
        simultaneous_at=2,
    )
    features = s.build_batch_features(operations, details)
    assert len(features) == 4
    assert features[2].operation_count == 2
    assert features[2].atom_count == 8
    assert all(feature.valid for feature in features)


def test_invalid_batch_resets_transition_continuity() -> None:
    operations, details = _validated_frames(
        batches=7,
        zero_available_at=3,
    )
    features = s.build_batch_features(operations, details)
    assert features[3].valid is False
    raw, transitions = s.build_raw_candidates(features)
    assert len(transitions) == 4
    assert all(
        row["signal_available_time"] != features[4].signal_time
        for rows in raw.values()
        for row in rows
    )


def test_transition_relation_and_fixed_side() -> None:
    assert s._relation((1, 1, 1, -1)) == ("FRACTURE", -1)
    assert s._relation((-1, -1, 0, -1)) == ("RELIEF", 1)
    assert s._relation((1, -1, 0, 1)) == ("NEUTRAL", 0)


def test_permutation_and_signal_ids_are_deterministic() -> None:
    atoms = [("op-a", "c1"), ("op-b", "c2"), ("op-c", "c3")]
    timestamp = pd.Timestamp("2020-01-02T00:00:00Z")
    assert s._permutation_destinations(atoms, timestamp) == (
        s._permutation_destinations(atoms, timestamp)
    )
    signal = s._primary_signal_id(timestamp, "FRACTURE")
    assert signal == s._primary_signal_id(timestamp, "FRACTURE")
    assert len(signal) == 64
    assert s._random_side(signal) in (-1, 1)


def test_global_reservation_precedes_split_containment() -> None:
    signal = pd.Timestamp("2019-12-31T00:00:00Z")
    first = s._candidate(
        control="primary",
        signal_time=signal,
        relation="FRACTURE",
        side_sign=-1,
        directions=(1, 1, 1, -1),
        prior_relation="BASELINE",
    )
    second_time = pd.Timestamp("2020-01-01T00:10:00Z")
    second = s._candidate(
        control="primary",
        signal_time=second_time,
        relation="RELIEF",
        side_sign=1,
        directions=(-1, -1, -1, 1),
        prior_relation="FRACTURE",
    )
    scheduled = s._schedule("primary", [first, second])
    assert scheduled == []


def test_build_clocks_preserves_control_order_and_primary_side_controls() -> None:
    operations, details = _validated_frames(batches=20)
    features = s.build_batch_features(operations, details)
    raw, _ = s.build_raw_candidates(features)
    clocks = s.build_clocks(raw)
    assert tuple(clocks) == s.prereg.CONTROL_ORDER
    for control in s.PRIMARY_CLOCK_SIDE_CONTROLS:
        assert len(clocks[control]) == len(clocks["primary"])


def test_synthetic_report_never_opens_comparator_or_outcomes() -> None:
    operations, details = _source_frames(batches=30)
    report, _ = s.build_support_from_frames(operations, details)
    assert report["artifact_eligible"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["outcomes_opened"] is False
    assert report["funding_loaded"] is False
    assert report["outcome_boundary"]["btc_market_rows_loaded"] == 0
    assert report["outcome_boundary"]["protocol_git_subprocess_calls"] == 0
    s.validate_report(report)


def test_deterministic_clock_is_canonical_and_symbolic_only() -> None:
    operations, details = _validated_frames(batches=30)
    features = s.build_batch_features(operations, details)
    raw, _ = s.build_raw_candidates(features)
    clocks = s.build_clocks(raw)
    first = s.deterministic_clock_bytes(clocks)
    second = s.deterministic_clock_bytes(clocks)
    assert first == second
    assert first[4:8] == b"\x00\x00\x00\x00"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        header = handle.readline().decode("utf-8").strip().split(",")
    assert header == list(s.CLOCK_COLUMNS)


def test_source_failure_short_circuits_comparator_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, details = _validated_frames(batches=10)
    features = s.build_batch_features(operations, details)
    raw, transitions = s.build_raw_candidates(features)
    clocks = s.build_clocks(raw)
    monkeypatch.setattr(
        s,
        "support_and_composition",
        lambda *args, **kwargs: (
            {},
            {"forced_source_failure": False},
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        s,
        "evaluate_novelty",
        lambda *args, **kwargs: pytest.fail("comparator accessed"),
    )
    clock = s.deterministic_clock_bytes(clocks)
    report = s._core_payload(
        operations=operations,
        details=details,
        features=features,
        raw=raw,
        transitions=transitions,
        clocks=clocks,
        preregistration=s.prereg.build_manifest(),
        source_audit={"bindings": {}},
        clock_bytes=clock,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )
    assert report["first_failing_stage"] == "source_support"
    assert report["comparator_rows_decoded"] == 0


def test_comparator_failure_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations, details = _validated_frames(batches=10)
    features = s.build_batch_features(operations, details)
    raw, transitions = s.build_raw_candidates(features)
    clocks = s.build_clocks(raw)
    monkeypatch.setattr(
        s,
        "support_and_composition",
        lambda *args, **kwargs: (
            {},
            {"source": True},
            {},
            {"composition": True},
        ),
    )

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise s.ComparatorContractFailure(
            "comparator_artifact_contract",
            17,
            "bad comparator",
        )

    monkeypatch.setattr(s, "evaluate_novelty", fail)
    clock = s.deterministic_clock_bytes(clocks)
    report = s._core_payload(
        operations=operations,
        details=details,
        features=features,
        raw=raw,
        transitions=transitions,
        clocks=clocks,
        preregistration=s.prereg.build_manifest(),
        source_audit={"bindings": {}},
        clock_bytes=clock,
        clock_path=s.DEFAULT_CLOCK_OUTPUT,
        artifact_eligible=True,
        protocol_git_subprocess_calls=2,
    )
    assert report["comparator_rows_decoded"] == 17
    assert report["first_failing_stage"] == "comparator_novelty"
    assert report["decision"] == "retire_SCAF_48_unchanged_before_outcomes"
    s.validate_report(report)


def test_comparator_validates_unselected_rows_before_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clock.csv.gz"
    frame = pd.DataFrame(
        {
            "control": ["primary", "unselected"],
            "entry_time": [
                "2021-01-01T00:00:00Z",
                "2021-02-01T00:00:00Z",
            ],
            "exit_time": [
                "2021-01-03T00:00:00Z",
                "2021-02-03T00:00:00Z",
            ],
            "side": ["LONG", "INVALID"],
        }
    )
    frame.to_csv(path, index=False, compression="gzip", lineterminator="\n")
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s.prereg, "REPOSITORY_ROOT", tmp_path)
    contract = {
        "path": path.name,
        "sha256": s.sha256_file(path.name),
        "header_sha256": s.prereg.sha256_csv_header(path.name),
        "read_csv": {
            "usecols": list(s.prereg.SLCS_USECOLS),
        },
        "groups": ["primary"],
        "minimum_contained_rows_each": 1,
    }
    with pytest.raises(
        s.ComparatorContractFailure,
        match="before filtering",
    ) as caught:
        s._read_comparator_groups(
            {"novelty_contract": {"comparator": contract}}
        )
    assert caught.value.rows_decoded == 2


def test_one_day_matching_and_occupancy_are_exact() -> None:
    candidate = pd.DataFrame(
        {
            "entry_time": [
                pd.Timestamp("2021-01-01T00:00:00Z"),
                pd.Timestamp("2021-01-05T00:00:00Z"),
            ],
            "exit_time": [
                pd.Timestamp("2021-01-03T00:00:00Z"),
                pd.Timestamp("2021-01-07T00:00:00Z"),
            ],
            "side": ["LONG", "SHORT"],
            "signal_id": ["a", "b"],
            "original_row_number": [0, 1],
        }
    )
    comparator = candidate.copy()
    assert s._maximum_bipartite_matches(candidate, comparator) == 2
    assert s._one_day_jaccard(candidate, comparator) == 1.0
    assert s._occupancy_correlation(candidate, comparator) == pytest.approx(1.0)


def test_run_checks_commit_before_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        s,
        "_assert_protocol_committed",
        lambda: (_ for _ in ()).throw(RuntimeError("not committed")),
    )
    monkeypatch.setattr(
        s,
        "load_sources",
        lambda: pytest.fail("source opened before commit proof"),
    )
    with pytest.raises(RuntimeError, match="not committed"):
        s.run()


def test_dirty_preregistration_builder_blocks_protocol_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> SimpleNamespace:
        calls.append(args)
        if args[0] == "ls-files":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(s, "_git_check", fake_git)
    with pytest.raises(RuntimeError, match="differ from HEAD"):
        s._assert_protocol_committed()
    assert any(
        str(s.PREREGISTRATION_BUILDER) in call for call in calls
    )


def test_write_once_is_confined_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    assert s._write_once("artifact.bin", b"alpha") == "created"
    assert s._write_once("artifact.bin", b"alpha") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once("artifact.bin", b"beta")
    with pytest.raises(RuntimeError, match="repository-relative"):
        s._write_once("../escape.bin", b"alpha")


def test_contract_hash_and_preregistration_are_bound() -> None:
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert s.sha256_file(s.PREREGISTRATION_BUILDER) == (
        s.PREREGISTRATION_BUILDER_SHA256
    )
