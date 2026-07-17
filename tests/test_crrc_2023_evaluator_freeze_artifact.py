from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from training import evaluate_crrc_2023 as evaluate
from training import freeze_crrc_2023_evaluator as freeze


PATH = Path("results/crrc_2023_evaluator_freeze_2026-07-17.json")


def test_evaluator_freeze_artifact_validates_without_opening_outcomes() -> None:
    payload = json.loads(PATH.read_text())
    freeze.validate_manifest(payload)
    assert evaluate.verify_evaluation_freeze() == payload
    assert payload["outcomes_opened"] is False
    assert payload["opened_windows"] == []
    assert payload["sealed_windows"] == ["2023", "2024", "2025", "2026"]
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False


def test_freeze_binds_committed_evaluator_and_tests() -> None:
    payload = json.loads(PATH.read_text())
    commit = payload["evaluation_source_commit"]
    assert commit == "1ba0e8ca50e870958969d5e3f7d129e0b28ce0fb"
    for path_key, digest_key in (
        ("evaluation_source", "evaluation_source_sha256"),
        ("test_path", "test_sha256"),
        ("freeze_source", "freeze_source_sha256"),
        ("freeze_test_path", "freeze_test_sha256"),
    ):
        path = Path(payload[path_key])
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        assert hashlib.sha256(committed).hexdigest() == current
        assert current == payload[digest_key]


def test_freeze_binds_all_preoutcome_artifacts_and_no_mutability() -> None:
    payload = json.loads(PATH.read_text())
    assert payload["preregistration_sha256"] == evaluate.PREREGISTRATION_SHA256
    assert payload["support_sha256"] == evaluate.SUPPORT_SHA256
    assert payload["primary_clock_sha256"] == evaluate.PRIMARY_CLOCK_SHA256
    assert payload["primary_event_clock_hash"] == evaluate.PRIMARY_EVENT_CLOCK_HASH
    assert payload["control_clocks_sha256"] == evaluate.CONTROL_CLOCKS_SHA256
    assert payload["execution_source_manifest_sha256"] == (
        evaluate.EXECUTION_SOURCE_MANIFEST_SHA256
    )
    assert payload["execution_source_manifest_hash"] == (
        evaluate.EXECUTION_SOURCE_MANIFEST_HASH
    )
    assert payload["mutable_parameters"] == []
    assert payload["primary_clock_rows"] == 156
