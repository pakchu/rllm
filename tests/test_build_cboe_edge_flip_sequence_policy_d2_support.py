from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from training import build_cboe_edge_flip_sequence_policy_d2_support as s
from training import build_cboe_edge_flip_sequence_policy_support as engine


def common_row(day: date, index: int) -> engine.CommonRow:
    toggle = Decimal(index % 5)
    return engine.CommonRow(
        observation_date=day,
        vix9d=Decimal("12") + toggle,
        vix=Decimal("14") + Decimal(index % 2),
        vix3m=Decimal("16") - Decimal(index % 3),
        skew=Decimal("110") + Decimal((index * 3) % 11),
        vvix=Decimal("80") + Decimal((index * 7) % 13),
        total_pcr=Decimal("0.8") + Decimal(index % 4) / Decimal("10"),
        index_pcr=Decimal("1.0") + Decimal(index % 3) / Decimal("10"),
        equity_pcr=Decimal("0.6")
        + Decimal((index + 1) % 4) / Decimal("10"),
        vix_pcr=Decimal("0.7") + Decimal((index + 2) % 5) / Decimal("10"),
        spx_pcr=Decimal("0.9") + Decimal((index + 3) % 4) / Decimal("10"),
        index_volume=1000 + index * 13,
        vix_volume=100 + (index % 7) * 11,
    )


def common_rows(start: date, count: int) -> list[engine.CommonRow]:
    return [
        common_row(start + timedelta(days=index), index)
        for index in range(count)
    ]


def fake_execution_seal() -> dict:
    commit = "a" * 40
    core = {
        "protocol_version": s.SEAL_PROTOCOL,
        "policy_id": s.POLICY_ID,
        "runtime": {
            "path": s.prereg.GIT_EXECUTABLE,
            "path_component": "/usr/bin",
            "sha256": s.prereg.GIT_EXECUTABLE_SHA256,
            "version": "git version 2.43.0",
        },
        "contract": {
            "path": s.CONTRACT_PATH,
            "commit": s.CONTRACT_COMMIT,
            "sha256": s.CONTRACT_SHA256,
        },
        "preregistration": {
            "path": s.PREREGISTRATION_PATH,
            "commit": s.PREREGISTRATION_COMMIT,
            "sha256": s.PREREGISTRATION_SHA256,
            "manifest_hash": s.PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": {
            "path": s.prereg.PRODUCER_SCRIPT,
            "commit": s.PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": s.PREREGISTRATION_PRODUCER_SHA256,
        },
        "d1_preregistration": {
            "path": s.prereg.D1_PREREGISTRATION,
            "commit": s.prereg.D1_PREREGISTRATION_COMMIT,
            "sha256": s.prereg.D1_PREREGISTRATION_SHA256,
            "manifest_hash": s.prereg.D1_PREREGISTRATION_MANIFEST_HASH,
        },
        "d1_preregistration_producer": {
            "path": s.prereg.D1_PRODUCER,
            "commit": s.prereg.D1_PRODUCER_COMMIT,
            "sha256": s.prereg.D1_PRODUCER_SHA256,
        },
        "d1_engine": {
            "path": s.prereg.D1_ENGINE,
            "commit": s.prereg.D1_ENGINE_COMMIT,
            "sha256": s.prereg.D1_ENGINE_SHA256,
        },
        "runner": {
            "path": s.RUNNER_PATH,
            "commit": commit,
            "sha256": "b" * 64,
        },
        "tests": {
            "path": s.TEST_PATH,
            "commit": commit,
            "sha256": "c" * 64,
        },
        "source_values_opened": False,
        "outcomes_opened": False,
    }
    return {**core, "manifest_hash": s.canonical_hash(core)}


def write_terminal_fixture(root: Path, decision: str) -> dict:
    authority: dict = {}
    details: dict = {}
    source_output = None
    control_output = None
    source_hash = None
    control_hash = None
    gate_one_checks = (
        {
            "runtime_authority_valid": True,
            "absolute_git_valid": True,
            "authority_valid": True,
            "seal_valid": True,
            "worktree_clean": True,
            **{name: True for name in engine.FORBIDDEN_COUNTER_NAMES},
        }
        if decision == "pass"
        else {"runtime_or_authority_valid": False}
    )
    gates = [s._gate_record(1, s.GATE_NAMES[0], gate_one_checks)]
    if decision == "pass":
        seal = fake_execution_seal()
        authority = s._expected_terminal_authority(seal)
        for index, name in enumerate(s.GATE_NAMES[1:], start=2):
            gates.append(s._gate_record(index, name, {"fixture": True}))
        schedules = engine.build_schedules(
            common_rows(date(2020, 1, 1), 6)
        )
        controls = engine.build_controls(schedules)
        schedule_detail = engine.schedule_metrics(schedules)
        schedule_detail["schedule_replay"] = {}
        details = {
            "parser": {},
            "schedule": schedule_detail,
            "edge_support": engine.edge_metrics(schedules),
            "diversity_stability": engine.diversity_metrics(schedules),
            "controls": engine.control_metrics(schedules, controls),
            "determinism_append_replay": {},
        }
        source_records = [engine.schedule_record(row) for row in schedules]
        control_records = [engine.control_record(row) for row in controls]
        source_bytes = engine.deterministic_csv_gzip(
            source_records,
            engine.SOURCE_OUTPUT_COLUMNS,
        )
        control_bytes = engine.deterministic_csv_gzip(
            control_records,
            engine.CONTROL_OUTPUT_COLUMNS,
        )
        source_path = root / s.SOURCE_OUTPUT
        control_path = root / s.CONTROL_OUTPUT
        source_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source_bytes)
        control_path.write_bytes(control_bytes)
        source_hash = hashlib.sha256(
            engine._canonical_records(source_records)
        ).hexdigest()
        control_hash = hashlib.sha256(
            engine._canonical_records(control_records)
        ).hexdigest()
        source_output = {
            "path": s.SOURCE_OUTPUT,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(source_records),
        }
        control_output = {
            "path": s.CONTROL_OUTPUT,
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "rows": len(control_records),
        }
    report = s._result_report(
        decision=decision,
        failure_action=s.FAILURE_ACTION if decision == "fail" else None,
        pass_action=s.PASS_ACTION if decision == "pass" else None,
        authority=authority,
        gates=gates,
        details=details,
        counters=engine.forbidden_counters(),
        source_hash=source_hash,
        control_hash=control_hash,
        source_output=source_output,
        control_output=control_output,
        error=(
            RuntimeError("fixture runtime failure")
            if decision == "fail"
            else None
        ),
    )
    report_path = root / (
        s.PASS_REPORT if decision == "pass" else s.REJECTION_REPORT
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(s._json_bytes(report))
    return report


def configure_fake_terminal(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(engine, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        s,
        "_validate_terminal_execution_seal",
        fake_execution_seal,
    )
    monkeypatch.setattr(
        engine,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        engine,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )


def test_d2_identity_and_paths_are_disjoint_from_terminal_d1() -> None:
    assert s.POLICY_ID == "CEFS-D2"
    assert s.REJECTION_REPORT != engine.REJECTION_REPORT
    assert s.PASS_REPORT != engine.PASS_REPORT
    assert s.SOURCE_OUTPUT != engine.SOURCE_OUTPUT
    assert s.CONTROL_OUTPUT != engine.CONTROL_OUTPUT
    assert s.GATE_NAMES[1:] == engine.GATE_NAMES[1:]


def test_absolute_git_survives_lookup_failure_but_runtime_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/bin")
    assert s._git_output("--version").startswith("git version ")
    with pytest.raises(RuntimeError, match="lacks exact /usr/bin"):
        s.prereg.validate_runtime_authority()


def test_frozen_authority_binds_d1_engine_without_source_decode() -> None:
    authority = s.validate_frozen_authority()
    assert authority["d1_engine"] == {
        "path": s.prereg.D1_ENGINE,
        "commit": s.prereg.D1_ENGINE_COMMIT,
        "sha256": s.prereg.D1_ENGINE_SHA256,
    }
    assert authority["d1_terminal"]["source_values_opened"] is False
    assert authority["d1_terminal"]["outcomes_opened"] is False
    assert authority["d1_preregistration_producer"] == {
        "path": s.prereg.D1_PRODUCER,
        "commit": s.prereg.D1_PRODUCER_COMMIT,
        "sha256": s.prereg.D1_PRODUCER_SHA256,
    }
    assert authority["runtime"]["path"] == "/usr/bin/git"


def test_result_hash_detects_identity_mutation() -> None:
    report = s._result_report(
        decision="fail",
        failure_action=s.FAILURE_ACTION,
        pass_action=None,
        authority={},
        gates=[
            s._gate_record(
                1,
                s.GATE_NAMES[0],
                {"runtime_or_authority_valid": False},
            )
        ],
        details={},
        counters=engine.forbidden_counters(),
        source_hash=None,
        control_hash=None,
        source_output=None,
        control_output=None,
        error=RuntimeError("probe"),
    )
    report["policy_id"] = "CEFS-D1"
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    assert report["result_hash"] != s.canonical_hash(core)


def test_partial_terminal_state_aborts_without_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    source = tmp_path / s.SOURCE_OUTPUT
    source.parent.mkdir(parents=True)
    source.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="partial terminal"):
        s.pre_run_terminal_state()
    assert not (tmp_path / s.REJECTION_REPORT).exists()


def test_valid_pass_returns_idempotently_before_new_gate_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fake_terminal(monkeypatch, tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")

    def forbidden_seal_call() -> None:
        raise AssertionError("terminal return must precede new Gate 1")

    monkeypatch.setattr(s, "validate_execution_seal", forbidden_seal_call)
    assert s.run_official() == report


def test_valid_gate_one_rejection_returns_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "fail")
    assert s.run_official() == report


def test_gate_one_rejection_cannot_claim_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "fail")
    report["source_row_hash"] = "a" * 64
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.REJECTION_REPORT).write_bytes(s._json_bytes(report))
    with pytest.raises(RuntimeError, match="row hash stage mismatch"):
        s.pre_run_terminal_state()


def test_terminal_pass_rejects_d1_identity_even_with_recomputed_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fake_terminal(monkeypatch, tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    report["policy_id"] = "CEFS-D1"
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.PASS_REPORT).write_bytes(s._json_bytes(report))
    with pytest.raises(RuntimeError, match="terminal identity mismatch"):
        s.pre_run_terminal_state()


def test_invalid_terminal_seal_fails_before_output_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fake_terminal(monkeypatch, tmp_path)
    write_terminal_fixture(tmp_path, "pass")

    def invalid_seal() -> dict:
        raise RuntimeError("invalid terminal seal")

    def forbidden_output_decode(*args: object, **kwargs: object) -> list:
        raise AssertionError("output decoded before terminal seal validation")

    monkeypatch.setattr(s, "_validate_terminal_execution_seal", invalid_seal)
    monkeypatch.setattr(
        engine,
        "_terminal_output_records",
        forbidden_output_decode,
    )
    with pytest.raises(RuntimeError, match="invalid terminal seal"):
        s.pre_run_terminal_state()


def test_runner_contains_no_bare_git_subprocess_literal() -> None:
    source = (s.REPOSITORY_ROOT / s.RUNNER_PATH).read_text()
    assert '("git",' not in source
    assert "['git'," not in source
    assert "prereg.GIT_EXECUTABLE" in source
    assert "preregister_cboe_edge_flip_sequence_policy as" not in source
    assert s._sealed_source_contracts() == s.prereg.d1_scientific_contract()[
        "sources"
    ]
