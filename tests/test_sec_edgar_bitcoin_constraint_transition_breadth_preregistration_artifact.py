from __future__ import annotations

import json
from pathlib import Path

from training.preregister_sec_edgar_bitcoin_constraint_transition_breadth import (
    MODEL_FILES,
    MODEL_REVISION,
    SOURCE_ARTIFACT_SHA256,
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path(
    "results/sec_edgar_bitcoin_constraint_transition_breadth_"
    "preregistration_2026-07-21.json"
)
ARTIFACT_SHA256 = "a9c55b98202b341ffb51bede731e5d2a2281d3851fbee604670868ea47470405"


def test_committed_preregistration_artifact_is_frozen_and_outcome_blind() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    manifest = payload.pop("manifest_hash")
    assert canonical_hash(payload) == manifest
    contract = payload["contract"]
    assert canonical_hash(contract) == payload["contract_hash"]
    assert contract["source"]["artifact_sha256"] == SOURCE_ARTIFACT_SHA256
    assert contract["model"]["revision"] == MODEL_REVISION
    assert contract["model"]["files"] == dict(MODEL_FILES)
    assert payload["anchors"]["local_model"]["validated"] is True
    source_anchor = payload["anchors"]["preregistration_source"]
    assert sha256_file(source_anchor["path"]) == source_anchor["sha256"]
    assert payload["outcome_boundary"] == {
        "filing_bodies_opened": 0,
        "semantic_model_calls": 0,
        "semantic_labels_created": 0,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "2024_or_later_source_rows_read": 0,
        "clean_room_claimed": False,
    }
    assert payload["decision"]["candidate_frozen"] is True
    assert payload["decision"]["synthetic_model_gate_authorized"] is True
    assert payload["decision"]["historical_semantic_execution_authorized"] is False
    assert payload["decision"]["economic_evaluation_authorized"] is False
    assert payload["decision"]["2024_or_later_authorized"] is False
