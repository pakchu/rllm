from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_miner_cadence_recovery_pre2024 as evaluate
from training import freeze_miner_cadence_recovery_evaluator as freeze


ARTIFACT = Path("results/miner_cadence_recovery_evaluator_freeze_2026-07-17.json")
EXPECTED_SHA256 = "3bcd142a581d4e40415e48bde25a8d2c96270797f4284be1d3e999633413403a"
EXPECTED_MANIFEST_HASH = (
    "23ffac153163afb376488625ddb64afd512fe7e7dad09e704c8a17cb7ead0c31"
)


def test_evaluator_freeze_is_bound_to_committed_source_before_outcomes() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_SHA256
    payload = json.loads(ARTIFACT.read_text())
    freeze.validate_manifest(payload)
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["evaluation_source_commit"] == (
        "5a90790f31d05e08c4c94eb5c0855dcf8f0ed872"
    )
    assert payload["evaluation_source_sha256"] == hashlib.sha256(
        evaluate.EVALUATION_SOURCE.read_bytes()
    ).hexdigest()
    assert payload["market_rows_parsed_during_freeze"] == 0
    assert payload["funding_rows_loaded_during_freeze"] == 0
    assert payload["execution_simulation_run_during_freeze"] is False
    assert payload["opened_windows"] == []
    assert payload["mutable_parameters"] == []


def test_control_clocks_reproduce_the_frozen_hashes_and_counts() -> None:
    payload = evaluate.verify_evaluation_freeze()
    clocks, _, _ = evaluate.verify_support_and_control_clocks()
    assert {name: evaluate._clock_hash(clock) for name, clock in clocks.items()} == payload[
        "control_clock_hashes"
    ]
    assert {name: len(clock) for name, clock in clocks.items()} == payload[
        "control_clock_counts"
    ]
