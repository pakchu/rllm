from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_usdc_recipient_concentration_dislocation as urcd


ARTIFACT = Path(
    "results/usdc_recipient_concentration_dislocation_preregistration_2026-07-23.json"
)
ARTIFACT_SHA256 = "078a40ee5e8604ec1dae1087541237617a287c6e2e953609bb71a392808e705e"
MANIFEST_HASH = "5be1af6621e59086ae5bf9a487e425e97c3c3403eebd227ae055b789c85ccb2e"
POLICY_HASH = "c9e7a04d5d41ae572f77bcb9f02b30ac2bc9b500c5dedcc7697bb9e4b5814bbd"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_urcd_preregistration_artifact_is_exact_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert payload["candidate"] == "URCD-72"
    assert payload["verification_mode"] == (
        "verified_hashes_headers_and_commit_guard"
    )
    assert payload["artifact_eligible"] is True
    assert payload["artifact_output"] == str(ARTIFACT)
    assert payload["manifest_hash"] == MANIFEST_HASH == urcd.canonical_hash(core)
    assert payload["policy_hash"] == POLICY_HASH
    assert len(payload["comparator_bindings"]) == 9
    assert payload["source_values_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False

    boundary = payload["outcome_boundary"]
    assert boundary == urcd.EXPECTED_BOUNDARY
    assert boundary["git_protocol_subprocess_calls"] == 2
    for field in (
        "source_value_rows_decoded",
        "comparator_value_rows_decoded",
        "urcd_features_or_incidence_computed",
        "btc_market_rows_decoded",
        "funding_rows_decoded",
        "future_return_rows_decoded",
        "pnl_cagr_mdd_values_decoded",
        "post_2023_source_value_rows_decoded",
        "network_calls",
    ):
        assert boundary[field] == 0

    assert _sha256(Path(payload["preregistration_source"])) == payload[
        "preregistration_source_sha256"
    ]
    assert _sha256(Path(payload["preregistration_test"])) == payload[
        "preregistration_test_sha256"
    ]
    assert _sha256(urcd.BOUNDARY_DOCUMENT) == urcd.BOUNDARY_DOCUMENT_SHA256
    assert _sha256(urcd.MECHANISM_DOCUMENT) == urcd.MECHANISM_DOCUMENT_SHA256
