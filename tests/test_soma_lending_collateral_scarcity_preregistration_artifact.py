from __future__ import annotations

import hashlib
import json

from training import preregister_soma_lending_collateral_scarcity as prereg


def test_frozen_preregistration_replays_exactly() -> None:
    stored = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    assert stored == prereg.build_preregistration()
    prereg.validate_preregistration(stored)
    assert stored["candidate"] == prereg.POLICY_ID
    assert stored["policy_hash"] == prereg.canonical_hash(stored["policy"])
    assert stored["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in stored.items() if key != "manifest_hash"}
    )


def test_frozen_artifact_opens_no_incidence_or_outcome() -> None:
    stored = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    assert stored["exact_source_incidence_opened"] is False
    assert stored["outcomes_opened"] is False
    assert stored["performance_values_opened"] is False
    assert stored["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert stored["source_binding"][
        "operation_value_rows_read_during_preregistration"
    ] == 0
    assert stored["source_binding"][
        "detail_value_rows_read_during_preregistration"
    ] == 0


def test_frozen_artifact_binds_source_and_mechanism_hashes() -> None:
    stored = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    assert stored["mechanism_decision"]["sha256"] == hashlib.sha256(
        prereg.MECHANISM_DECISION.read_bytes()
    ).hexdigest()
    assert stored["source_binding"]["operations_sha256"] == hashlib.sha256(
        prereg.OPERATIONS.read_bytes()
    ).hexdigest()
    assert stored["source_binding"]["details_sha256"] == hashlib.sha256(
        prereg.DETAILS.read_bytes()
    ).hexdigest()
    assert stored["source_binding"]["manifest_sha256"] == hashlib.sha256(
        prereg.SOURCE_MANIFEST.read_bytes()
    ).hexdigest()
