from __future__ import annotations

import hashlib
import json
import subprocess

from training import preregister_cboe_edge_flip_sequence_policy as p


ARTIFACT_SHA256 = (
    "5e515663e99ef4aa322cae25cfb2c07f69b3e24f289bc2f0f79463aca64a8878"
)
MANIFEST_HASH = (
    "9aa7c891ec241d4733db215068bed3507f41c03cbae7198c906a079ddb6467bf"
)
PRODUCER_COMMIT = "ec8eae23226d39f0c62b6c5711d6080f2bf990a4"
PRODUCER_SHA256 = (
    "1c4668e72846eadf66011c582c62e8574af3679204500c1ff9631101ecbb7ac1"
)


def artifact() -> dict:
    return json.loads((p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT).read_text())


def test_persisted_preregistration_is_exact_valid_and_outcome_blind() -> None:
    payload = artifact()
    p.validate_manifest(payload)
    assert p.sha256_file(p.DEFAULT_OUTPUT) == ARTIFACT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert all(value == 0 for value in payload["forbidden_access"].values())
    assert payload["contingent_economic_chronology"]["authorized_now"] is False


def test_artifact_is_bound_to_sealed_producer_bytes() -> None:
    payload = artifact()
    assert payload["authority"]["producer"] == {
        "commit": PRODUCER_COMMIT,
        "path": p.PRODUCER_SCRIPT,
        "sha256": PRODUCER_SHA256,
    }
    sealed = subprocess.run(
        ["git", "show", f"{PRODUCER_COMMIT}:{p.PRODUCER_SCRIPT}"],
        cwd=p.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(sealed).hexdigest() == PRODUCER_SHA256
    assert p.sha256_file(p.PRODUCER_SCRIPT) == PRODUCER_SHA256


def test_persisted_artifact_is_write_once_idempotent() -> None:
    payload = artifact()
    assert p.write_once(p.DEFAULT_OUTPUT, payload) == ARTIFACT_SHA256
