from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd
import pytest

from training import (
    build_collateral_liquidity_ordering_relation_source_support as s,
)


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def source_time(value: datetime) -> str:
    return value.isoformat()


def treasury_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=s.prereg.TREASURY_ALLOWLIST)


def treasury_row(
    *,
    auction_date: str = "2020-09-01",
    available: str = "2020-09-02T00:00:00+00:00",
    term: str = "2-Year",
    competitive: str = "10",
    primary: str = "5",
    direct: str = "3",
    indirect: str = "2",
    complete: str = "true",
) -> dict[str, str]:
    return {
        "auction_date": auction_date,
        "result_available_at_utc": available,
        "original_security_term": term,
        "competitive_accepted_usd": competitive,
        "primary_dealer_accepted_usd": primary,
        "direct_bidder_accepted_usd": direct,
        "indirect_bidder_accepted_usd": indirect,
        "source_complete": complete,
    }


def soma_frames(
    records: Iterable[
        tuple[str, str, str, str, str]
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    operations: list[dict[str, str]] = []
    details: list[dict[str, str]] = []
    for operation_id, day, available, submitted, accepted in records:
        operations.append(
            {
                "operation_id": operation_id,
                "operation_date": day,
                "available_at_utc": available,
                "total_par_submitted": submitted,
                "total_par_accepted": accepted,
            }
        )
        details.append(
            {
                "operation_id": operation_id,
                "operation_date": day,
                "available_at_utc": available,
                "par_submitted": submitted,
                "par_accepted": accepted,
            }
        )
    return (
        pd.DataFrame(operations, columns=s.prereg.SOMA_OPERATION_ALLOWLIST),
        pd.DataFrame(details, columns=s.prereg.SOMA_DETAIL_ALLOWLIST),
    )


def ofr_rows(
    day: str,
    *,
    rate_values: tuple[str, str, str] = ("5", "4", "3"),
    volume_values: tuple[str, str, str] = ("1", "2", "3"),
    disclosure_edit: str = "0",
    omit: str | None = None,
) -> list[dict[str, str]]:
    observation = datetime.fromisoformat(day).date()
    available = source_time(s._expected_ofr_availability(observation))
    values = (*rate_values, *volume_values)
    rows: list[dict[str, str]] = []
    for mnemonic, value in zip(s.prereg.OFR_MNEMONICS, values, strict=True):
        if mnemonic == omit:
            continue
        rows.append(
            {
                "mnemonic": mnemonic,
                "observation_date": day,
                "available_at_utc": available,
                "value": value,
                "disclosure_edit": disclosure_edit,
            }
        )
    return rows


def ofr_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=s.prereg.OFR_ALLOWLIST)


def decision_row(
    split: str,
    when: datetime,
    *,
    updated: tuple[str, ...] = s.SOURCE_ORDER,
    sequence: str | None = None,
    line: str = "line",
    primitives: tuple[str, str, str, str, str, str] = (
        "P>D>I",
        "UP",
        "DOWN",
        "EQUAL",
        "DVP>GCF>TRIV1",
        "TRIV1>GCF>DVP",
    ),
) -> s.JointRow:
    return s.JointRow(
        split=split,
        execution_time=when,
        valid=True,
        invalid_reason="",
        model_decision=True,
        updated=updated,
        treasury=primitives[0],
        soma_submitted_step=primitives[1],
        soma_accepted_step=primitives[2],
        soma_coverage_step=primitives[3],
        ofr_rate_order=primitives[4],
        ofr_volume_order=primitives[5],
        line_text=line,
        line_sha256=hashlib.sha256(line.encode("ascii")).hexdigest(),
        sequence_sha256=sequence or hashlib.sha256(
            f"{split}|{when.isoformat()}".encode("ascii")
        ).hexdigest(),
        decision_expiry_time=when + timedelta(hours=72),
    )


def nondecision_row(
    split: str,
    when: datetime,
    *,
    line: str,
    variant: int,
) -> s.JointRow:
    primitives = (
        "P>D>I" if variant % 2 else "D>P>I",
        "UP" if variant % 2 else "DOWN",
        "DOWN" if variant % 2 else "UP",
        "EQUAL" if variant % 2 else "UP",
        (
            "DVP>GCF>TRIV1"
            if variant % 2
            else "GCF>DVP>TRIV1"
        ),
        (
            "TRIV1>GCF>DVP"
            if variant % 2
            else "DVP>TRIV1>GCF"
        ),
    )
    return replace(
        decision_row(
            split,
            when,
            line=line,
            primitives=primitives,
        ),
        model_decision=False,
        sequence_sha256="",
        decision_expiry_time=None,
    )


def test_import_does_not_decode_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("module import decoded a CSV")

    monkeypatch.setattr(pd, "read_csv", fail)
    importlib.reload(s)


def test_forbidden_access_schema_is_exact_and_zero() -> None:
    assert tuple(s.forbidden_access()) == s.FORBIDDEN_COUNTER_NAMES
    assert all(value == 0 for value in s.forbidden_access().values())


def test_exact_scalar_parsers_and_timestamp_format() -> None:
    assert s.parse_date("2023-12-31").isoformat() == "2023-12-31"
    with pytest.raises(RuntimeError):
        s.parse_date("2023-1-01")
    timestamp = s.parse_source_time("2023-12-31T23:59:59+00:00")
    assert s.format_time(timestamp) == "2023-12-31T23:59:59Z"
    for value in (
        "",
        "2023-12-31T23:59:59",
        "2023-12-31T23:59:59.1+00:00",
        "2023-12-31T23:59:59+09:00",
        "2024-01-01T00:00:00+00:00",
        "2023-12-31T23:59:59Z",
    ):
        with pytest.raises(RuntimeError):
            s.parse_source_time(value)
    assert s.parse_fraction("1.25", nonnegative=True) == s.Fraction(5, 4)
    for value in ("1e3", "+1", "01", "nan", "-0", "-0.0"):
        with pytest.raises(RuntimeError):
            s.parse_fraction(value, nonnegative=True)


def test_weak_order_and_execution_clock_are_frozen() -> None:
    assert (
        s.weak_order(
            {"P": s.Fraction(2), "D": s.Fraction(2), "I": s.Fraction(1)},
            ("P", "D", "I"),
        )
        == "P=D>I"
    )
    assert len(s.weak_order_vocabulary(("P", "D", "I"))) == 13
    assert s.execution_time(utc("2020-09-10T00:00:00Z")) == utc(
        "2020-09-10T00:05:00Z"
    )
    assert s.execution_time(utc("2020-09-10T00:00:01Z")) == utc(
        "2020-09-10T00:10:00Z"
    )


def test_treasury_complete_and_incomplete_batches() -> None:
    complete = treasury_row()
    incomplete = treasury_row(
        auction_date="2020-09-02",
        available="2020-09-03T00:00:00+00:00",
        competitive="",
        primary="",
        direct="",
        indirect="",
        complete="false",
    )
    batches, audit = s.build_treasury_batches(
        treasury_frame([complete, incomplete])
    )
    assert batches[0].token == ("2-Year:P>D>I",)
    assert batches[1] == s.SourceBatch(
        "TREASURY",
        utc("2020-09-03T00:00:00Z"),
        False,
    )
    assert audit["complete_rows"] == 1
    assert audit["incomplete_rows"] == 1


def test_treasury_rejects_reconciliation_identity_and_batch_errors() -> None:
    with pytest.raises(RuntimeError):
        s.build_treasury_batches(
            treasury_frame([treasury_row(competitive="11")])
        )
    duplicate = treasury_row()
    with pytest.raises(RuntimeError):
        s.build_treasury_batches(treasury_frame([duplicate, duplicate]))
    duplicate_term = treasury_row(auction_date="2020-09-02")
    with pytest.raises(RuntimeError):
        s.build_treasury_batches(
            treasury_frame([treasury_row(), duplicate_term])
        )
    incomplete_with_amount = treasury_row(
        complete="false",
        competitive="",
        primary="1",
        direct="",
        indirect="",
    )
    with pytest.raises(RuntimeError):
        s.build_treasury_batches(
            treasury_frame([incomplete_with_amount])
        )


def test_soma_baseline_transition_and_reconciliation() -> None:
    operations, details = soma_frames(
        [
            (
                "A",
                "2020-09-01",
                "2020-09-02T00:00:00+00:00",
                "10",
                "5",
            ),
            (
                "B",
                "2020-09-02",
                "2020-09-03T00:00:00+00:00",
                "20",
                "4",
            ),
        ]
    )
    batches, audit = s.build_soma_batches(operations, details)
    assert len(batches) == 1
    assert batches[0].token == ("UP", "DOWN", "DOWN")
    assert audit["complete_batches"] == 2
    broken = details.copy()
    broken.loc[1, "par_accepted"] = "3"
    with pytest.raises(RuntimeError):
        s.build_soma_batches(operations, broken)


def test_soma_rejects_unknown_or_mismatched_details() -> None:
    operations, details = soma_frames(
        [
            (
                "A",
                "2020-09-01",
                "2020-09-02T00:00:00+00:00",
                "10",
                "5",
            )
        ]
    )
    unknown = details.copy()
    unknown.loc[0, "operation_id"] = "UNKNOWN"
    with pytest.raises(RuntimeError):
        s.build_soma_batches(operations, unknown)
    mismatched = details.copy()
    mismatched.loc[0, "operation_date"] = "2020-09-02"
    with pytest.raises(RuntimeError):
        s.build_soma_batches(operations, mismatched)


def test_ofr_retains_only_authorized_mnemonics() -> None:
    ignored = {
        "mnemonic": "UNAUTHORIZED",
        "observation_date": "2020-08-31",
        "available_at_utc": "2020-09-10T00:00:00+00:00",
        "value": "1",
        "disclosure_edit": "0",
    }
    rows = [ignored, *ofr_rows("2020-09-01")]
    batches, audit = s.build_ofr_batches(ofr_frame(rows))
    assert len(batches) == 1
    assert batches[0].valid
    assert audit["physical_dates"] == 1
    assert audit["required_rows"] == 6


@pytest.mark.parametrize(
    ("mutation", "expected_valid"),
    [
        ("missing", False),
        ("empty", False),
        ("edited", False),
    ],
)
def test_ofr_incomplete_dates_emit_invalidation(
    mutation: str,
    expected_valid: bool,
) -> None:
    rows = ofr_rows(
        "2020-09-01",
        omit=(
            s.prereg.OFR_MNEMONICS[-1]
            if mutation == "missing"
            else None
        ),
        disclosure_edit="1" if mutation == "edited" else "0",
    )
    if mutation == "empty":
        rows[-1]["value"] = ""
    batches, _ = s.build_ofr_batches(ofr_frame(rows))
    assert len(batches) == 1
    assert batches[0].valid is expected_valid
    assert batches[0].token == ()


def test_ofr_selects_greatest_complete_date_in_shared_batch() -> None:
    rows = [
        *ofr_rows(
            "2020-09-01",
            rate_values=("1", "2", "3"),
            volume_values=("3", "2", "1"),
        ),
        *ofr_rows(
            "2020-09-02",
            rate_values=("9", "2", "1"),
            volume_values=("1", "2", "9"),
        ),
    ]
    batches, _ = s.build_ofr_batches(ofr_frame(rows))
    assert len(batches) == 1
    assert batches[0].token == (
        "DVP>GCF>TRIV1",
        "TRIV1>GCF>DVP",
    )


def test_ofr_rejects_duplicate_identity_and_invalid_volume() -> None:
    rows = ofr_rows("2020-09-01")
    with pytest.raises(RuntimeError):
        s.build_ofr_batches(ofr_frame([*rows, rows[0]]))
    with pytest.raises(RuntimeError):
        s.build_ofr_batches(
            ofr_frame(
                ofr_rows(
                    "2020-09-01",
                    volume_values=("0", "0", "0"),
                )
            )
        )


def test_optional_physical_count_enforcement_rejects_synthetic_frames() -> None:
    with pytest.raises(RuntimeError):
        s.build_treasury_batches(
            treasury_frame([treasury_row()]),
            enforce_physical_counts=True,
        )
    operations, details = soma_frames(
        [
            (
                "A",
                "2020-09-01",
                "2020-09-02T00:00:00+00:00",
                "10",
                "5",
            )
        ]
    )
    with pytest.raises(RuntimeError):
        s.build_soma_batches(
            operations,
            details,
            enforce_physical_counts=True,
        )
    with pytest.raises(RuntimeError):
        s.build_ofr_batches(
            ofr_frame(ofr_rows("2020-09-01")),
            enforce_physical_counts=True,
        )


def test_joint_schedule_decides_on_twelfth_valid_line() -> None:
    rows = s.build_joint_rows(s._synthetic_batches())
    assert len(rows) == 18
    assert rows[0].execution_time == utc("2020-09-10T00:05:00Z")
    assert not any(row.model_decision for row in rows[:11])
    assert all(row.model_decision for row in rows[11:])
    sequence = "\n".join(row.line_text for row in rows[:12]).encode("ascii")
    assert rows[11].sequence_sha256 == hashlib.sha256(sequence).hexdigest()
    assert rows[11].decision_expiry_time == rows[11].execution_time + timedelta(
        hours=72
    )
    assert s._primary_schedule_checks(s._synthetic_batches(), rows)[
        "causal_state_freshness_history"
    ]


def test_invalid_source_resets_history_and_emits_empty_safety_row() -> None:
    batches = s._synthetic_batches()
    invalid_at = utc("2020-09-10T02:00:00Z")
    batches.append(s.SourceBatch("OFR", invalid_at, False))
    for index in range(12):
        available = invalid_at + timedelta(minutes=5 * (index + 1))
        batches.extend(
            (
                s.SourceBatch(
                    "TREASURY", available, True, ("2-Year:P>D>I",)
                ),
                s.SourceBatch(
                    "SOMA", available, True, ("UP", "DOWN", "EQUAL")
                ),
                s.SourceBatch(
                    "OFR",
                    available,
                    True,
                    ("DVP>GCF>TRIV1", "TRIV1>GCF>DVP"),
                ),
            )
        )
    rows = s.build_joint_rows(batches)
    invalid = next(row for row in rows if not row.valid)
    assert invalid.invalid_reason == "INVALID_OFR"
    assert invalid.csv_row()["treasury"] == ""
    after = [row for row in rows if row.execution_time > invalid.execution_time]
    assert not any(row.model_decision for row in after[:11])
    assert after[11].model_decision


def test_freshness_endpoint_is_inclusive_then_stale() -> None:
    initial = utc("2020-09-10T00:00:00Z")
    batches = [
        s.SourceBatch("TREASURY", initial, True, ("2-Year:P>D>I",)),
        s.SourceBatch("SOMA", initial, True, ("UP", "UP", "UP")),
        s.SourceBatch(
            "OFR",
            initial,
            True,
            ("DVP>GCF>TRIV1", "TRIV1>GCF>DVP"),
        ),
        s.SourceBatch(
            "TREASURY",
            utc("2020-09-13T23:55:00Z"),
            True,
            ("2-Year:D>P>I",),
        ),
        s.SourceBatch(
            "TREASURY",
            utc("2020-09-14T00:00:00Z"),
            True,
            ("2-Year:I>P>D",),
        ),
    ]
    rows = s.build_joint_rows(batches)
    endpoint = next(
        row for row in rows if row.execution_time == utc("2020-09-14T00:00:00Z")
    )
    stale = next(
        row for row in rows if row.execution_time == utc("2020-09-14T00:05:00Z")
    )
    assert endpoint.valid
    assert stale.invalid_reason == "STALE_SOMA|STALE_OFR"


def test_controls_preserve_schedule_and_future_append_is_invariant() -> None:
    batches = s._synthetic_batches()
    primary = s.build_joint_rows(batches)
    controls = {
        control: s.build_control_rows(batches, primary, control)
        for control in s.RELATION_CONTROLS
    }
    duplicate = {
        control: s.build_control_rows(batches, primary, control)
        for control in s.RELATION_CONTROLS
    }
    gate = s.relation_controls_gate(primary, controls, duplicate)
    assert gate["passed"], gate
    appended = s.build_joint_rows([*batches, *s.future_append_batches()])
    assert [row.csv_row() for row in appended] == [
        row.csv_row() for row in primary
    ]


def test_artifact_schedule_semantics_rejects_line_tamper() -> None:
    primary = s.build_joint_rows(s._synthetic_batches())
    assert s._artifact_schedule_semantics(primary)
    tampered = list(primary)
    tampered[0] = replace(
        tampered[0],
        line_text=tampered[0].line_text + " ",
        line_sha256=hashlib.sha256(
            (tampered[0].line_text + " ").encode("ascii")
        ).hexdigest(),
    )
    assert not s._artifact_schedule_semantics(tampered)


def test_decision_count_and_update_support_thresholds() -> None:
    rows: list[s.JointRow] = []
    for split, minimum in {"TRAIN": 450, "TEST": 180, "EVAL": 180}.items():
        start = s.SPLITS[split][0]
        rows.extend(
            decision_row(split, start + timedelta(minutes=5 * index))
            for index in range(minimum)
        )
    assert s.model_decision_count_gate(rows)["passed"]
    assert s.model_decision_count_gate(
        [
            *rows,
            decision_row(
                "TRAIN",
                s.SPLITS["TRAIN"][0] + timedelta(days=400),
            ),
        ]
    )["passed"]
    update = s.source_update_support_gate(rows)
    assert update["passed"]
    missing_one = [
        row
        for row in rows
        if not (
            row.split == "TEST"
            and row.execution_time
            < s.SPLITS["TEST"][0] + timedelta(minutes=5 * 161)
        )
    ]
    assert not s.model_decision_count_gate(missing_one)["passed"]


def test_maximum_gap_includes_split_endpoints() -> None:
    rows: list[s.JointRow] = []
    for split, (start, end) in s.SPLITS.items():
        current = start + timedelta(days=9)
        while current < end:
            rows.append(decision_row(split, current))
            current += timedelta(days=9)
        if end - rows[-1].execution_time > timedelta(days=10):
            rows.append(decision_row(split, end - timedelta(days=1)))
    assert s.maximum_decision_gap_gate(rows)["passed"]
    broken = [
        row
        for row in rows
        if not (row.split == "TEST" and row.execution_time < s.SPLITS["TEST"][0] + timedelta(days=20))
    ]
    assert not s.maximum_decision_gap_gate(broken)["passed"]


def test_calendar_support_uses_exact_half_open_quarters() -> None:
    rows: list[s.JointRow] = []
    windows = [
        (s.SPLITS["TRAIN"][0], utc("2021-01-01T00:00:00Z"), 30),
    ]
    for year, minimum in ((2021, 50), (2022, 40), (2023, 40)):
        for quarter in range(1, 5):
            start = s._quarter_start(year, quarter)
            end = (
                s._quarter_start(year, quarter + 1)
                if quarter < 4
                else utc(f"{year + 1}-01-01T00:00:00Z")
            )
            windows.append((start, end, minimum))
    for start, end, minimum in windows:
        split = s.split_for(start) or "TRAIN"
        span = int((end - start).total_seconds())
        rows.extend(
            decision_row(
                split,
                start + timedelta(seconds=((index + 1) * span // (minimum + 1))),
            )
            for index in range(minimum)
        )
    assert s.calendar_support_gate(rows)["passed"]
    assert not s.calendar_support_gate(rows[:-1])["passed"]


def test_diversity_signature_and_sequence_gates() -> None:
    valid_rows: list[s.JointRow] = []
    sequence_rows: list[s.JointRow] = []
    minimums = {"TRAIN": 150, "TEST": 70, "EVAL": 70}
    for split, minimum in minimums.items():
        start = s.SPLITS[split][0]
        valid_rows.extend(
            nondecision_row(
                split,
                start + timedelta(minutes=5 * index),
                line=f"{split}-line-{index}",
                variant=index,
            )
            for index in range(4)
        )
        sequence_rows.extend(
            decision_row(
                split,
                start + timedelta(minutes=5 * index),
                sequence=hashlib.sha256(
                    f"{split}-{index}".encode("ascii")
                ).hexdigest(),
            )
            for index in range(minimum)
        )
    assert s.primitive_diversity_gate(valid_rows)["passed"]
    assert s.state_signature_concentration_gate(valid_rows)["passed"]
    assert s.sequence_uniqueness_gate(sequence_rows)["passed"]
    concentrated = [
        replace(row, line_text="same") for row in valid_rows
    ]
    assert not s.state_signature_concentration_gate(concentrated)["passed"]


def test_forbidden_gate_requires_exact_schema_and_integer_zero() -> None:
    assert s.forbidden_access_gate(s.forbidden_access())["passed"]
    nonzero = s.forbidden_access()
    nonzero["network_calls"] = 1
    assert not s.forbidden_access_gate(nonzero)["passed"]
    wrong_type = s.forbidden_access()
    wrong_type["network_calls"] = False
    assert not s.forbidden_access_gate(wrong_type)["passed"]


def test_deterministic_gzip_has_zero_mtime_and_exact_schema() -> None:
    primary = s.build_joint_rows(s._synthetic_batches())
    controls = {
        control: s.build_control_rows(
            s._synthetic_batches(),
            primary,
            control,
        )
        for control in s.RELATION_CONTROLS
    }
    source = s.deterministic_source_gzip(primary)
    control = s.deterministic_control_gzip(controls)
    assert source == s.deterministic_source_gzip(primary)
    assert control == s.deterministic_control_gzip(controls)
    assert int.from_bytes(source[4:8], "little") == 0
    assert int.from_bytes(control[4:8], "little") == 0
    assert tuple(
        gzip.decompress(source).decode("ascii").splitlines()[0].split(",")
    ) == s.SOURCE_COLUMNS
    assert tuple(
        gzip.decompress(control).decode("ascii").splitlines()[0].split(",")
    ) == s.CONTROL_COLUMNS


def test_self_check_is_canonical_deterministic_and_source_blind() -> None:
    first = s.self_check_bytes()
    second = s.self_check_bytes()
    assert first == second
    payload = json.loads(first)
    assert first == s.canonical_bytes(payload) + b"\n"
    assert payload["source_value_rows_opened"] == 0
    assert payload["predecessor_value_rows_opened"] == 0
    assert payload["forbidden_access"] == s.forbidden_access()
    completed = subprocess.run(
        [sys.executable, s.RUNNER_PATH, "self-check"],
        cwd=s.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    assert completed.stderr == b""
    assert completed.stdout == first


def test_pytest_summary_parser_rejects_missing_passes() -> None:
    assert s._pytest_summary("27 passed in 1.00s\n", "")["passed"] == 27
    with pytest.raises(RuntimeError):
        s._pytest_summary("1 failed in 1.00s\n", "")


def use_temp_terminal_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s.prereg, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s, "SOURCE_OUTPUT", "data/source.csv.gz")
    monkeypatch.setattr(s, "CONTROL_OUTPUT", "data/controls.csv.gz")
    monkeypatch.setattr(s, "PASS_REPORT", "results/pass.json")
    monkeypatch.setattr(s, "REJECTION_REPORT", "results/reject.json")
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()


def test_rejection_terminal_is_write_once_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    use_temp_terminal_root(monkeypatch, tmp_path)
    report = s.build_result_report(
        decision="reject",
        authority={"synthetic": True},
        ledger=s.AccessLedger(),
        audits={},
        primary=[],
        controls=None,
        gates=[s._gate_record(1, {"failed": False}, {})],
        counters=s.forbidden_access(),
        artifacts=None,
    )
    s._publish_rejection(report)
    assert s.terminal_state() == report
    s._publish_rejection(report)
    drifted = dict(report)
    drifted["error"] = {"type": "X", "message": "drift"}
    with pytest.raises(RuntimeError):
        s._publish_rejection(drifted)
    path = tmp_path / s.REJECTION_REPORT
    payload = json.loads(path.read_text())
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError):
        s.terminal_state()


def test_pass_terminal_group_is_complete_and_hash_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    use_temp_terminal_root(monkeypatch, tmp_path)
    batches = s._synthetic_batches()
    primary = s.build_joint_rows(batches)
    controls = {
        control: s.build_control_rows(batches, primary, control)
        for control in s.RELATION_CONTROLS
    }
    source_bytes = s.deterministic_source_gzip(primary)
    control_bytes = s.deterministic_control_gzip(controls)
    source_records = [row.csv_row() for row in primary]
    control_records = s.control_records(controls)
    artifacts = {
        "source": {
            "path": s.SOURCE_OUTPUT,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(source_records),
            "row_hash": s.canonical_hash(source_records),
        },
        "controls": {
            "path": s.CONTROL_OUTPUT,
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "rows": len(control_records),
            "row_hash": s.canonical_hash(control_records),
        },
    }
    gates = [
        s._gate_record(1, {"ok": True}, {}),
        s._gate_record(
            2,
            {"ok": True},
            {"primary_row_hash": s.canonical_hash(source_records)},
        ),
        *[
            s._gate_record(index, {"ok": True}, {})
            for index in range(3, 12)
        ],
    ]
    gate_functions = (
        "model_decision_count_gate",
        "source_update_support_gate",
        "maximum_decision_gap_gate",
        "calendar_support_gate",
        "primitive_diversity_gate",
        "state_signature_concentration_gate",
        "sequence_uniqueness_gate",
        "relation_controls_gate",
        "forbidden_access_gate",
    )
    for gate, name in zip(gates[2:], gate_functions, strict=True):
        monkeypatch.setattr(s, name, lambda *args, gate=gate: gate)
    ledger = s.AccessLedger(
        treasury_rows=445,
        soma_operation_rows=1_259,
        soma_detail_rows=182_616,
        ofr_rows=77_369,
    )
    audits = {
        "treasury": {
            "physical_rows": 445,
            "complete_rows": 440,
            "incomplete_rows": 5,
        },
        "soma": {
            "operation_rows": 1_259,
            "detail_rows": 182_616,
            "operations": 1_259,
        },
        "ofr": {"physical_rows": 77_369},
    }
    report = s.build_result_report(
        decision="pass",
        authority={"synthetic": True},
        ledger=ledger,
        audits=audits,
        primary=primary,
        controls=controls,
        gates=gates,
        counters=s.forbidden_access(),
        artifacts=artifacts,
    )
    s._publish_pass_group(source_bytes, control_bytes, report)
    assert s.terminal_state() == report
    (tmp_path / s.SOURCE_OUTPUT).write_bytes(source_bytes + b"drift")
    with pytest.raises(RuntimeError):
        s.terminal_state()


def test_partial_or_conflicting_terminal_state_hard_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    use_temp_terminal_root(monkeypatch, tmp_path)
    (tmp_path / s.SOURCE_OUTPUT).write_bytes(b"partial")
    with pytest.raises(RuntimeError):
        s.terminal_state()
