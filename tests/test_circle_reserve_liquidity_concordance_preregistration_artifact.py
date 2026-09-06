from __future__ import annotations

import hashlib
import json

from training import preregister_circle_reserve_liquidity_concordance as p


ARTIFACT_SHA256 = (
    "a3da6ca20d42aa8253d0b126eb362774051e20a3e14540e81622efcb24483e70"
)
MANIFEST_HASH = (
    "d9bd957107bce86b446c640e7bc6b03e655489d4a30799616386b136d1eaffca"
)
PRODUCER_COMMIT = "9822f0e7b169c3e3e13db666c37a154ff1f151d6"


def test_committed_crlc_preregistration_is_exact_and_source_unseen() -> None:
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
    assert identity["head_equals_upstream_required"] is True
    p.validate_recorded_repository(identity)

    assert all(payload[name] is False for name in p.EVIDENCE_BOUNDARIES)
    effects = payload["producer_effects"]
    assert effects["production_source_urls_requested"] == 0
    assert effects["production_source_rows_opened"] == 0
    assert effects["future_protocol_files_opened_or_hashed"] == 0
    assert effects["source_unseen"] is True
    assert effects["outcome_blind"] is True


def test_write_once_reverifies_committed_crlc_artifact_without_mutation() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    before = artifact.read_bytes()
    status, payload = p.write_once()

    assert status == "verified_existing"
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert artifact.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == ARTIFACT_SHA256
