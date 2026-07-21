from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, cast

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator_v3 as freeze,
)


FREEZE = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_v3_2026-07-21.json"
)
FREEZE_SHA256 = "ad8d5072c616be0f5ad311d67853fd02adcd6d5276b1ee28d075e7cde04ff814"
MANIFEST_HASH = "9558600bd1b6e6cfefe97c478b0335a22c93a482bf5c8d2763205fda924ed796"


def _payload() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))


def test_v3_freeze_artifact_is_hash_locked_and_non_pristine() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == FREEZE_SHA256
    payload = _payload()
    freeze.validate_manifest(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["evaluator_protocol_version"].endswith("_v3")
    assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
    assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
    failures = payload["prior_failed_run_boundaries"]
    assert failures["v1_date_representation_failure"] == (freeze.V1_FAILED_RUN_BOUNDARY)
    assert failures["v2_duplicate_network_availability_failure"] == (
        freeze.V2_FAILED_RUN_BOUNDARY
    )
    assert all(value == 0 for value in payload["v3_freeze_boundary"].values())


def test_v3_evaluator_commit_reproduces_frozen_source() -> None:
    payload = _payload()
    committed = subprocess.check_output(
        [
            "git",
            "show",
            f"{payload['evaluator_source_commit']}:{payload['evaluator_source']}",
        ]
    )
    assert hashlib.sha256(committed).hexdigest() == freeze.EXPECTED_EVALUATOR_SHA256
    assert payload["predecessor_freeze"] == {
        "path": str(freeze.PREDECESSOR_FREEZE),
        "sha256": freeze.PREDECESSOR_FREEZE_SHA256,
    }
    assert payload["v2_failure_and_v3_correction"] == {
        "path": str(freeze.CORRECTION_DOCUMENT),
        "sha256": freeze.CORRECTION_DOCUMENT_SHA256,
        "correction_scope": "simultaneous network publication batching only",
    }


def test_v3_freeze_contract_did_not_change_research_policy() -> None:
    payload = _payload()
    contract = payload["frozen_contract"]
    assert contract["controls"] == list(freeze.evaluate.CONTROL_NAMES)
    assert contract["support_limits"] == freeze.evaluate.SUPPORT_LIMITS
    assert contract["novelty_limits"] == freeze.evaluate.NOVELTY_LIMITS
    assert contract["macro_ttl_hours"] == 36
    assert contract["relay_deadline_hours"] == 36
    assert contract["hold_hours"] == 72
    assert contract["accepted_date_representations"] == [
        "YYYY-MM-DD",
        "YYYY-MM-DD 00:00:00",
    ]
    assert contract["network_publication_batch_rule"] == (
        "raw available_at monotonic non-decreasing; exact ties collapse to latest "
        "observation_date; final timestamps unique increasing"
    )
    assert payload["mutable_parameters"] == []
