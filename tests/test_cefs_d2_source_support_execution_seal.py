from __future__ import annotations

import hashlib
import json
import subprocess

from training import build_cboe_edge_flip_sequence_policy_d2_support as s


SEAL_SHA256 = (
    "01d1a9940c0a0bd7d377909658d9ecec23844e9a1da3ef12c5dc8d21b1ab12c8"
)
SEAL_MANIFEST_HASH = (
    "9a5fafb050fbdd2d71ff7161058c326ac30e127b8913e6851d298ca84af3ee40"
)
RUNNER_COMMIT = "f439136f73d03443e13bf3230a18bc96d8936691"
RUNNER_SHA256 = (
    "b057198f14cc8da056602f5b3ff4bf1953c012b12b82466ce999d82b2346d5be"
)
TEST_SHA256 = (
    "d6e9d74dbec6c0d52df8fb24f8d4b15b06cd5258d3891b6e35dcde7249ab56b4"
)


def seal() -> dict:
    return json.loads((s.REPOSITORY_ROOT / s.EXECUTION_SEAL_PATH).read_text())


def test_execution_seal_is_exact_and_forbidden_data_blind() -> None:
    payload = seal()
    assert s.sha256_file(s.EXECUTION_SEAL_PATH) == SEAL_SHA256
    assert payload["manifest_hash"] == SEAL_MANIFEST_HASH
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["policy_id"] == "CEFS-D2"
    assert payload["source_values_opened"] is False
    assert payload["outcomes_opened"] is False


def test_execution_seal_binds_runner_tests_and_all_d1_code_dependencies() -> None:
    payload = seal()
    assert payload["runner"] == {
        "path": s.RUNNER_PATH,
        "commit": RUNNER_COMMIT,
        "sha256": RUNNER_SHA256,
    }
    assert payload["tests"] == {
        "path": s.TEST_PATH,
        "commit": RUNNER_COMMIT,
        "sha256": TEST_SHA256,
    }
    assert payload["d1_engine"] == {
        "path": s.prereg.D1_ENGINE,
        "commit": s.prereg.D1_ENGINE_COMMIT,
        "sha256": s.prereg.D1_ENGINE_SHA256,
    }
    assert payload["d1_preregistration"] == {
        "path": s.prereg.D1_PREREGISTRATION,
        "commit": s.prereg.D1_PREREGISTRATION_COMMIT,
        "sha256": s.prereg.D1_PREREGISTRATION_SHA256,
        "manifest_hash": s.prereg.D1_PREREGISTRATION_MANIFEST_HASH,
    }
    assert payload["d1_preregistration_producer"] == {
        "path": s.prereg.D1_PRODUCER,
        "commit": s.prereg.D1_PRODUCER_COMMIT,
        "sha256": s.prereg.D1_PRODUCER_SHA256,
    }
    for path, expected in (
        (s.RUNNER_PATH, RUNNER_SHA256),
        (s.TEST_PATH, TEST_SHA256),
    ):
        sealed = subprocess.run(
            [
                s.prereg.GIT_EXECUTABLE,
                "show",
                f"{RUNNER_COMMIT}:{path}",
            ],
            cwd=s.REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(sealed).hexdigest() == expected
        assert s.sha256_file(path) == expected


def test_execution_seal_binds_absolute_git_runtime() -> None:
    assert seal()["runtime"] == {
        "path": "/usr/bin/git",
        "path_component": "/usr/bin",
        "sha256": s.prereg.GIT_EXECUTABLE_SHA256,
        "version": "git version 2.43.0",
    }
    assert s.sha256_file("/usr/bin/git") == s.prereg.GIT_EXECUTABLE_SHA256
    assert s._git_output("--version") == "git version 2.43.0"


def test_runner_validates_committed_execution_seal() -> None:
    payload = s.validate_execution_seal()
    assert payload["manifest_hash"] == SEAL_MANIFEST_HASH
