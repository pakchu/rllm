from __future__ import annotations

import json
from pathlib import Path

from training.preregister_sec_edgar_bitcoin_operational_capacity import (
    MODEL_REVISION,
    canonical_hash,
    sha256_file,
)


ARTIFACT = Path(
    "results/sec_edgar_bitcoin_operational_capacity_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = "5ce80db5875e282a1d173489bb910026b53f073f9c783c5140ee99fe3d72605d"
CONTRACT_HASH = "da227af07d68626974dd69c8934fa5258d3ea14f697b5a4e7960a5a460dda391"
MANIFEST_HASH = "76fe7387078dfd0c8783c889cdf81685ac2b3de8aa16c8854fb96b8c7a82901e"
DATASET_HASHES = {
    "train": "207c98d9a453e0913f5072732d81136a335a621120725d84f8734c66c7939630",
    "calibration": (
        "718ea047bcddd4268afc267df54b81a26d5c01ece30386ace1fb58e5d9a28a86"
    ),
    "adversarial": (
        "2e2c4fbcbf4894a034852f164db95721db840f416fa6e7f6f9fb81c8f491a14c"
    ),
    "swaps": "2f25aa4c0ce34d8581bff84bf34c3df58110c08918de3234dd30ecc874033cc2",
}


def _payload() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_preregistration_is_exact_and_self_consistent() -> None:
    assert sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = _payload()
    assert payload["contract_hash"] == CONTRACT_HASH
    assert payload["contract_hash"] == canonical_hash(payload["contract"])
    assert payload["manifest_hash"] == MANIFEST_HASH
    manifest = dict(payload)
    observed_manifest = manifest.pop("manifest_hash")
    assert observed_manifest == canonical_hash(manifest)
    source = payload["anchors"]["preregistration_source"]
    assert sha256_file(source["path"]) == source["sha256"]


def test_committed_synthetic_files_match_the_frozen_manifests() -> None:
    payload = _payload()
    datasets = payload["contract"]["synthetic"]["datasets"]
    for split, expected_hash in DATASET_HASHES.items():
        manifest = datasets[split]
        assert manifest["sha256"] == expected_hash
        assert sha256_file(manifest["path"]) == expected_hash
    assert datasets["train"]["rows"] == 512
    assert datasets["calibration"]["rows"] == 128
    assert datasets["adversarial"]["rows"] == 192
    assert datasets["swaps"]["rows"] == 128


def test_artifact_freezes_model_training_gate_and_zero_outcome_access() -> None:
    payload = _payload()
    contract = payload["contract"]
    assert contract["model"]["revision"] == MODEL_REVISION
    assert contract["model"]["lora"]["trainable_parameters"] == 2_678_784
    assert contract["training"]["optimizer_steps"] == 64
    assert contract["training"]["checkpoint_steps"] == [16, 32, 48, 64]
    assert contract["final_synthetic_gate"]["overall_exact_share_min"] == 0.95
    assert contract["final_synthetic_gate"]["mixed_exact_share"] == 1.0
    assert payload["anchors"]["local_model"]["validated"] is True
    assert payload["outcome_boundary"] == {
        "2024_or_later_source_rows_read": 0,
        "btc_market_rows_read": 0,
        "clean_room_claimed": False,
        "comparator_anchor_files_hashed": 11,
        "comparator_clock_fields_read": 0,
        "comparator_rows_parsed": 0,
        "filing_bodies_opened": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "historical_semantic_labels_created": 0,
        "historical_semantic_model_calls": 0,
        "historical_windows_created": 0,
        "return_or_pnl_fields_read": 0,
        "synthetic_model_calls": 0,
        "synthetic_rows_created": 960,
    }
    assert payload["decision"]["synthetic_training_authorized"] is True
    assert payload["decision"]["filing_body_transport_authorized"] is False
    assert payload["decision"]["economic_evaluation_authorized"] is False
