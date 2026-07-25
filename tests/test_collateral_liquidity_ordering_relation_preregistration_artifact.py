from __future__ import annotations

import json
from pathlib import Path

from training import preregister_collateral_liquidity_ordering_relation as p


ARTIFACT = Path(p.DEFAULT_OUTPUT)
ARTIFACT_SHA256 = (
    "7aee03d42daade588a0e785133632ff6f9f9e2a8d23117b49ffd405a41341e89"
)
MANIFEST_HASH = (
    "881f4c631f924e26e827e71359ce8df1f9add309d0b828478a545c7262f00b2b"
)
SCIENTIFIC_CONTRACT_HASH = (
    "59b52c826eef8315dc81a15a0467a6a494c9fcad8c54f7bf2c73c8ecf344a22a"
)
PRODUCER = {
    "path": p.PRODUCER_SCRIPT,
    "commit": "52efeed17445157ac585b246fde16830e23236b3",
    "sha256": (
        "322b8d4e19649bcda4d99a5e2a7888f3e38908f443037e6c4569c7ab31075942"
    ),
}


def artifact_payload() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_clor_d1_preregistration_artifact_is_exact_and_valid() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = artifact_payload()
    p.validate_manifest(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["scientific_contract_hash"] == SCIENTIFIC_CONTRACT_HASH
    assert payload["authority"]["producer"] == PRODUCER
    assert payload["authority"]["boundary"] == {
        "path": p.BOUNDARY_DOCUMENT,
        "commit": p.BOUNDARY_COMMIT,
        "sha256": p.BOUNDARY_SHA256,
    }
    assert payload["authority"]["common_window_policy"] == {
        "path": p.COMMON_WINDOW_DOCUMENT,
        "commit": p.COMMON_WINDOW_COMMIT,
        "sha256": p.COMMON_WINDOW_SHA256,
    }


def test_clor_d1_preregistration_remains_source_and_outcome_blind() -> None:
    payload = artifact_payload()
    assert payload["source_rows_parsed"] == 0
    assert payload["source_values_opened"] is False
    assert payload["joint_state_rows_built"] == 0
    assert payload["outcomes_opened"] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())
    decision = payload["scientific_contract"]["decision"]
    assert decision["predecessor_gzip_headers_decoded"] is True
    assert decision["predecessor_value_or_action_rows_decoded"] == 0
    assert decision["joint_state_rows_decoded"] == 0


def test_clor_d1_preregistration_creation_is_idempotent() -> None:
    assert p.create() == artifact_payload()
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
