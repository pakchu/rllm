from __future__ import annotations

import hashlib
import json

from training import preregister_ethereum_settlement_demand_impulse as p


ARTIFACT_SHA256 = (
    "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
)
MANIFEST_HASH = (
    "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
)


def test_committed_preregistration_artifact_is_exact_and_still_outcome_blind() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    assert artifact.is_file()
    assert not artifact.is_symlink()
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256

    payload = json.loads(raw)
    identity = p.frozen_repository_identity()
    assert payload == p.build_manifest(identity)
    assert raw == p.canonical_manifest_bytes(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert all(payload[name] is False for name in p.EVIDENCE_BOUNDARIES)
    assert payload["producer_effects"]["data_rows_opened"] == 0
    assert payload["producer_effects"][
        "comparator_or_gross9_artifact_bytes_opened"
    ] == 0
    assert payload["gross9"]["authority"]["runtime_code_closure"][
        "exact_runtime_environment"
    ] == p.current_runtime_environment()


def test_write_once_reverifies_the_singleton_without_mutation() -> None:
    status, payload = p.write_once()
    assert status == "verified_existing"
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert hashlib.sha256(
        (p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT).read_bytes()
    ).hexdigest() == ARTIFACT_SHA256
