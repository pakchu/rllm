from __future__ import annotations

import hashlib
import json
import subprocess

from training import build_cboe_edge_flip_sequence_policy_support as s


SEAL_SHA256 = (
    "c1d6aa251108afa520d1279c3fd0f2795a1c92ee229a593c1effa28f2445f331"
)
SEAL_MANIFEST_HASH = (
    "05c36438d6f82499c9fce6fff55e993bbdb05d09fd1c7ce92aa9ff7ca4a00f96"
)
RUNNER_COMMIT = "d7213f647128fc6160672bc61f080b3dcf7d1f42"
RUNNER_SHA256 = (
    "2069084d65146540488672115ee09f292cd31e6611bf92a569d534ab8a74c688"
)
TEST_SHA256 = (
    "01de1671cdf3c7fb4acf1b6e9ec8cb06d94f1c971381ab4252a4140c01132937"
)


def seal() -> dict:
    return json.loads((s.REPOSITORY_ROOT / s.EXECUTION_SEAL_PATH).read_text())


def test_execution_seal_is_exact_and_forbidden_data_blind() -> None:
    payload = seal()
    assert s.sha256_file(s.EXECUTION_SEAL_PATH) == SEAL_SHA256
    assert payload["manifest_hash"] == SEAL_MANIFEST_HASH
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["source_values_opened"] is False
    assert payload["outcomes_opened"] is False


def test_execution_seal_binds_runner_tests_and_preregistration_producer() -> None:
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
    assert payload["preregistration_producer"] == {
        "commit": s.PREREGISTRATION_PRODUCER_COMMIT,
        "path": s.prereg.PRODUCER_SCRIPT,
        "sha256": s.PREREGISTRATION_PRODUCER_SHA256,
    }
    preregistration = json.loads(
        (s.REPOSITORY_ROOT / s.PREREGISTRATION_PATH).read_text()
    )
    assert (
        preregistration["authority"]["producer"]
        == payload["preregistration_producer"]
    )
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
