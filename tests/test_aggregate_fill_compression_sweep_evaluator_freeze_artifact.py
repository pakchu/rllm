from __future__ import annotations

import hashlib
import json
import subprocess

from training import evaluate_aggregate_fill_compression_sweep as evaluate


EXPECTED_FREEZE_SHA256 = (
    "78ef1f8f72fa3cfee81f45a317d8044dcd217753f070bde08f6173ef98ad4012"
)
EXPECTED_MANIFEST_HASH = (
    "48b9e5653f2926456e71ce0be1adffa9de435d5caf54ba1f56d9ade3dab3af8a"
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
    assert payload["opened_windows"] == []
    assert payload["mutable_parameters"] == []
    assert payload["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert payload["funding_settlement_marks_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["sealed_windows"] == [
        "stage1_2020_2022",
        "stage2_2023",
        "2024",
        "2025",
        "2026_ytd",
    ]
