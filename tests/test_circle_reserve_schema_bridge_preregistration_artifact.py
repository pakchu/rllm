from __future__ import annotations

import hashlib
import json

from training import preregister_circle_reserve_schema_bridge as p


ARTIFACT_SHA256 = (
    "1f0eb234d4f8f12ab3f28568636fa4e9550857a17ebd69533c77521e7106aa23"
)
MANIFEST_HASH = (
    "cb7f255e00697796ce48bd4f16f686855fdc30bc83f6f813b845731acbab8d2a"
)
PRODUCER_COMMIT = "8a1b42c0503bc963f4025c70b82e5a2a01af1afb"


def test_committed_crsb_preregistration_is_exact_and_source_unseen() -> None:
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


def test_write_once_reverifies_committed_crsb_artifact_without_mutation() -> None:
    artifact = p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT
    before = artifact.read_bytes()
    status, payload = p.write_once()

    assert status == "verified_existing"
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert artifact.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == ARTIFACT_SHA256
