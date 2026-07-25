from __future__ import annotations

import hashlib
import json
import subprocess

from training import build_london_cash_derivative_path_source_support as s


SEAL_SHA256 = (
    "3a1aac307a06eaa22c08651957d07bff62ff542e4aadb4feb263a67d52cd7047"
)
SEAL_MANIFEST_HASH = (
    "55aa5c0081b23c2e0789e57085db8bf095729ff627581c5775d1fc5a028904d1"
)
RUNNER_COMMIT = "92f9fe11cd1047340c042c2b1ec3796add6523bf"
RUNNER_SHA256 = (
    "d1fa16c8b57154e8102902f17bf7032e65a8f4cfc5cc5098b561d390cb285bda"
)
TEST_SHA256 = (
    "51c015292c683ef55e090e9e7d5bf32f21fb4828102b3d8b69fe6c9f0445dbcf"
)


def seal() -> dict:
    return json.loads((s.REPOSITORY_ROOT / s.EXECUTION_SEAL_PATH).read_text())


def test_execution_seal_is_exact_and_outcome_blind() -> None:
    payload = seal()
    assert s.sha256_file(s.EXECUTION_SEAL_PATH) == SEAL_SHA256
    assert payload["manifest_hash"] == SEAL_MANIFEST_HASH
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["source_values_opened"] is False
    assert payload["outcomes_opened"] is False


def test_execution_seal_binds_committed_runner_and_tests() -> None:
    payload = seal()
    assert payload["runner"] == {
        "commit": RUNNER_COMMIT,
        "path": s.RUNNER_PATH,
        "sha256": RUNNER_SHA256,
    }
    assert payload["tests"] == {
        "commit": RUNNER_COMMIT,
        "path": s.TEST_PATH,
        "sha256": TEST_SHA256,
    }
    for path, expected in (
        (s.RUNNER_PATH, RUNNER_SHA256),
        (s.TEST_PATH, TEST_SHA256),
    ):
        sealed = subprocess.run(
            ["git", "show", f"{RUNNER_COMMIT}:{path}"],
            cwd=s.REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(sealed).hexdigest() == expected
        assert s.sha256_file(path) == expected


def test_runner_validates_committed_execution_seal() -> None:
    payload = s.validate_execution_seal()
    assert payload["manifest_hash"] == SEAL_MANIFEST_HASH
