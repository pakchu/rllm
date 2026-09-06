from __future__ import annotations

import hashlib
import json

from training import preregister_tron_usdt_supply_impulse as p


ARTIFACT_SHA256 = (
    "54817044b8df76dc347ed64b6fe5f6f2dfdddcdb211bded4ba2b1af133d49067"
)
MANIFEST_HASH = (
    "d67cd1b67632ae92e9458395e729627a6f4c3b4b75ce97187653eac3a09e40c1"
)
PRODUCER_COMMIT = "d7bfc024ec1c23092e562d1ec7b2031ce8c30b35"


def test_committed_preregistration_artifact_is_exact_and_outcome_blind() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    assert artifact.is_file()
    assert not artifact.is_symlink()

    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARTIFACT_SHA256
    payload = json.loads(raw)
    assert raw == p.canonical_manifest_bytes(payload)
    assert payload["manifest_hash"] == MANIFEST_HASH

    identity = payload["frozen_preregistration"]["repository_identity"]
    assert identity["head_commit"] == PRODUCER_COMMIT
    assert identity["upstream_commit"] == PRODUCER_COMMIT
    assert identity["whole_worktree_clean_required"] is True
    assert all(payload[name] is False for name in p.EVIDENCE_BOUNDARIES)
    assert payload["producer_effects"]["git_read_only_subprocess_calls"] == 6
    assert payload["producer_effects"]["source_csv_or_source_rows_opened"] == 0
    assert payload["producer_effects"]["comparator_data_rows_opened"] == 0
    assert payload["producer_effects"]["gross9_data_rows_opened"] == 0
    assert payload["producer_effects"]["market_rows_opened"] == 0
    assert payload["producer_effects"]["funding_rows_opened"] == 0


def test_write_once_reverifies_committed_artifact_without_mutation() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    before = artifact.read_bytes()
    status, payload = p.write_once()

    assert status == "verified_existing"
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert artifact.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == ARTIFACT_SHA256
