from __future__ import annotations

import hashlib
import json
import subprocess

from training import evaluate_venue_ticket_migration_shock as evaluate


EXPECTED_FREEZE_SHA256 = (
    "9af4523bc031a4b4017aa6430ddf7e69469cc47d4ffcbd4f3d2dc9a32f2e6ac9"
)
EXPECTED_MANIFEST_HASH = (
    "1eeb879d38d4757a3c01b2bccdb25d33798630b65763c6d96dc5af3e0fda1239"
)


def test_frozen_evaluator_artifact_is_reproducible_and_outcome_blind() -> None:
    assert evaluate._sha256(evaluate.EVALUATION_FREEZE) == EXPECTED_FREEZE_SHA256
    payload = json.loads(evaluate.EVALUATION_FREEZE.read_text())
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert evaluate.verify_evaluation_freeze() == payload

    commit = payload["evaluation_source_commit"]
    source = payload["evaluation_source"]
    committed = subprocess.check_output(["git", "show", f"{commit}:{source}"])
    assert hashlib.sha256(committed).hexdigest() == payload["evaluation_source_sha256"]
    assert payload["candidate_id"] == "VTMS-288"
    assert payload["opened_windows"] == []
    assert payload["mutable_parameters"] == []
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_settlement_marks_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["sealed_windows"] == evaluate.SEALED_WINDOWS
    assert set(payload["control_schedules"]) == set(evaluate.support.POLICY_NAMES)
