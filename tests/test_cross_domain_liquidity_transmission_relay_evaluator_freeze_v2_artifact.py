from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, cast

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator_v2 as freeze,
)


FREEZE = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_v2_2026-07-21.json"
)
FREEZE_SHA256 = "f11b9c52538b370b80f5d1ee3e8d62f8ec132083a6060c2d983e9b9ed8c7965d"
MANIFEST_HASH = "862741da1e3f762aa881e990fede928d930d636e5ad09f298441b7dfe58babb0"


def test_v2_freeze_artifact_is_hash_locked_and_non_pristine() -> None:
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == FREEZE_SHA256
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
    freeze.validate_manifest(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["evaluator_protocol_version"].endswith("_v2")
    assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
    assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
    assert payload["prior_failed_run_boundary"]["source_value_rows_loaded"] == 4_468
    assert payload["prior_failed_run_boundary"]["candidate_event_rows_derived"] == 0
    assert payload["prior_failed_run_boundary"]["btc_market_rows_loaded"] == 0
    assert all(value == 0 for value in payload["v2_freeze_boundary"].values())


def test_v2_evaluator_commit_reproduces_frozen_source() -> None:
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
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
    assert payload["v1_failure_and_v2_correction"] == {
        "path": str(freeze.CORRECTION_DOCUMENT),
        "sha256": freeze.CORRECTION_DOCUMENT_SHA256,
        "correction_scope": "date representation normalization only",
    }


def test_v2_freeze_contract_did_not_change_research_policy() -> None:
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
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
    assert payload["mutable_parameters"] == []
