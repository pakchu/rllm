from __future__ import annotations

import hashlib
import json
import subprocess

from training import preregister_london_cash_derivative_path as p


ARTIFACT_SHA256 = (
    "da0dd2f24236c3b64e31604268b0ad9d9b342723629790d1ecf061d0a02f4ad4"
)
MANIFEST_HASH = (
    "0cbeeaad957c67187381405681e8e7935c7039c7c9f9e2d0a19cbe5e912d5dac"
)
PRODUCER_COMMIT = "a1e5435ea43fb17337ce76b80ccf53d5c26e9b0b"
PRODUCER_SHA256 = (
    "a20118ab1b7cfe1a12c050bfc4a612689286383d9fa8fd2043dfcce62fd7368f"
)


def artifact() -> dict:
    return json.loads((p.REPOSITORY_ROOT / p.DEFAULT_OUTPUT).read_text())


def test_persisted_preregistration_is_exact_and_valid() -> None:
    payload = artifact()
    p.validate_manifest(payload)
    assert p.sha256_file(p.DEFAULT_OUTPUT) == ARTIFACT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert all(value == 0 for value in payload["forbidden_access"].values())


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
