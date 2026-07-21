from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, cast

from training import (
    freeze_cross_domain_liquidity_transmission_relay_support_evaluator as freeze,
)


FREEZE = Path(
    "results/cross_domain_liquidity_transmission_relay_evaluator_freeze_2026-07-21.json"
)
FREEZE_SHA256 = "b02091be3dad585c8cea865eaeb2535500ad47d3528af4b418f96ac2310d1bf5"
MANIFEST_HASH = "fd900ccf93f01a9622874a83efcde11d31ce8bea74e778e6f4d50cfe1d4cbbc0"
EVALUATOR_COMMIT = "6900b42ecc7d64c708218fcf048290e52ceb7a46"
EVALUATOR_SHA256 = "649a4d4da64df32c3acb66ccedc6ad607bc8abef6b247235ff42e837ab3992e1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evaluator_freeze_is_hash_locked_before_source_incidence() -> None:
    assert _sha256(FREEZE) == FREEZE_SHA256
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
    freeze.validate_manifest(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["evaluator_source_commit"] == EVALUATOR_COMMIT
    assert payload["evaluator_source_sha256"] == EVALUATOR_SHA256
    assert payload["preregistration"] == {
        "manifest_hash": freeze.evaluate.PREREGISTRATION_MANIFEST_HASH,
        "path": str(freeze.evaluate.PREREGISTRATION_ARTIFACT),
        "policy_hash": "3b59238a1b6c76dd4b9cb7c3bdde0571a6e852eb0ec9c36b469388e3aac95197",
        "sha256": freeze.evaluate.PREREGISTRATION_ARTIFACT_SHA256,
    }

    committed = subprocess.check_output(
        ["git", "show", f"{EVALUATOR_COMMIT}:{payload['evaluator_source']}"]
    )
    assert hashlib.sha256(committed).hexdigest() == EVALUATOR_SHA256


def test_evaluator_freeze_keeps_every_value_and_outcome_boundary_closed() -> None:
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
    for field in (
        "opened_source_value_rows",
        "derived_real_event_rows",
        "opened_comparator_event_rows",
        "opened_comparator_manifest_values",
        "opened_btc_market_rows",
        "opened_funding_rows",
        "opened_return_rows",
        "opened_pnl_or_equity_fields",
        "opened_post_2023_observation_rows",
        "network_calls",
    ):
        assert payload[field] == 0
    assert payload["economic_simulation_run"] is False
    assert payload["mutable_parameters"] == []
    assert payload["source_support_artifact_exists_before_freeze"] is False
    assert payload["source_support_clock_exists_before_freeze"] is False


def test_frozen_contract_matches_evaluator_constants() -> None:
    payload = cast(dict[str, Any], json.loads(FREEZE.read_text(encoding="utf-8")))
    contract = payload["frozen_contract"]
    assert contract["controls"] == list(freeze.evaluate.CONTROL_NAMES)
    assert contract["support_limits"] == freeze.evaluate.SUPPORT_LIMITS
    assert contract["novelty_limits"] == freeze.evaluate.NOVELTY_LIMITS
    assert contract["macro_ttl_hours"] == 36
    assert contract["relay_deadline_hours"] == 36
    assert contract["hold_hours"] == 72
    assert contract["exposure_grid"] == ("full [2021-01-01, 2024-01-01) UTC at 5m")
