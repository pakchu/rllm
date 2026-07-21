from __future__ import annotations

import json
from pathlib import Path

from training.preregister_sec_edgar_bitcoin_product_access import (
    MODEL_REVISION,
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path(
    "results/sec_edgar_bitcoin_product_access_preregistration_2026-07-22.json"
)
ARTIFACT_SHA256 = "ab975eea454fbe1a784adaee979c5ad6162be9b18363c7fe3aa47959e075b883"


def test_committed_preregistration_artifact_is_exact_and_self_consistent() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == "sec_edgar_bitcoin_product_access_prereg_v1"
    assert payload["contract_hash"] == canonical_hash(payload["contract"])
    manifest = dict(payload)
    recorded_manifest = manifest.pop("manifest_hash")
    assert recorded_manifest == canonical_hash(manifest)
    source = payload["anchors"]["preregistration_source"]
    assert sha256_file(source["path"]) == source["sha256"]


def test_artifact_freezes_e2b_synthetic_memory_and_outcome_boundaries() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    contract = payload["contract"]
    assert contract["model"]["revision"] == MODEL_REVISION
    assert payload["anchors"]["local_model"]["validated"] is True
    assert len(contract["synthetic_controls"]["cases"]) == 24
    assert len(contract["novelty"]["clock_comparators"]) == 13
    assert contract["synthetic_gate"]["maximum_peak_allocated_bytes"] == 7 * 1024**3
    assert contract["synthetic_gate"]["maximum_peak_reserved_bytes"] == int(
        7.25 * 1024**3
    )
    assert payload["outcome_boundary"] == {
        "2024_or_later_source_rows_read": 0,
        "btc_market_rows_read": 0,
        "clean_room_claimed": False,
        "comparator_anchor_files_hashed": 16,
        "comparator_clock_fields_read": 0,
        "comparator_rows_parsed": 0,
        "filing_bodies_opened": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "semantic_labels_created": 0,
        "semantic_model_calls": 0,
    }
    assert payload["decision"]["synthetic_model_gate_authorized"] is True
    assert payload["decision"]["filing_body_transport_authorized"] is False
    assert payload["decision"]["economic_evaluation_authorized"] is False
