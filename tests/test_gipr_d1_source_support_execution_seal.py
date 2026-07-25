from __future__ import annotations

import hashlib
import json
import subprocess

from training import (
    build_governance_intent_payload_relation_source_support as gipr,
)


SEAL_SHA256 = (
    "c8d19e98e0923a72d6481dd61f7a5c009dd6baf2e861f68c1b18b640d85dc021"
)
SEAL_HASH = (
    "ed9c01bac7bb8af5a8ab99f991ae28b420889541780098bf35011289a94dd13e"
)
SHARED_COMMIT = "2347d1a159a44bebbdcd259e94716472b76756ec"
RUNNER_SHA256 = (
    "a940fdefed2e03c92c6ab9648f8250ab2443b797377ac879f72685eb29f2a60e"
)
TEST_SHA256 = (
    "cf6a7e9979942e10ef1f553a49e8edb624bef855d5106cdc56c886a91b150570"
)
SELF_CHECK_MANIFEST_HASH = (
    "6039d3f01f304ef80e2536ada4f31ecbecc5be12c1d84f9213ca4d0aa9868681"
)
SELF_CHECK_STDOUT_SHA256 = (
    "ee50d0375514b7d00a24f54cda414938c8d0776879f42b8aa05db62af8468514"
)


def seal_bytes() -> bytes:
    return (gipr.REPO_ROOT / gipr.EXECUTION_SEAL_PATH).read_bytes()


def seal() -> dict:
    return json.loads(seal_bytes())


def test_execution_seal_is_canonical_and_source_blind() -> None:
    raw = seal_bytes()
    payload = seal()
    assert hashlib.sha256(raw).hexdigest() == SEAL_SHA256
    assert raw == gipr.canonical_json_bytes(payload)
    core = {
        key: value for key, value in payload.items() if key != "seal_hash"
    }
    assert payload["seal_hash"] == SEAL_HASH == gipr.canonical_hash(core)
    assert payload["protocol_version"] == gipr.SEAL_PROTOCOL
    assert payload["policy_id"] == gipr.POLICY_ID
    assert payload["forbidden_access"] == gipr.AccessLedger.zero().snapshot()
    self_check = payload["synthetic_verification"]["self_check"]
    assert self_check["network_calls"] == 0
    assert self_check["source_event_rows_opened"] == 0
    assert self_check["outcomes_opened"] is False
    assert self_check["manifest_hash"] == SELF_CHECK_MANIFEST_HASH
    assert self_check["stdout_sha256"] == SELF_CHECK_STDOUT_SHA256
    assert self_check["forbidden_access"] == (
        gipr.AccessLedger.zero().snapshot()
    )
    pytest_record = payload["synthetic_verification"]["pytest"]
    assert pytest_record == {
        "argv": [
            ".venv/bin/pytest",
            "-q",
            gipr.TEST_PATH.as_posix(),
        ],
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": 0,
        "passed": 48,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_committed_runner_and_tests() -> None:
    payload = seal()
    assert payload["shared_commit"] == SHARED_COMMIT
    expected = {
        gipr.RUNNER_PATH: RUNNER_SHA256,
        gipr.TEST_PATH: TEST_SHA256,
    }
    for key, path in (
        ("runner", gipr.RUNNER_PATH),
        ("tests", gipr.TEST_PATH),
    ):
        assert payload[key] == {
            "path": path.as_posix(),
            "commit": SHARED_COMMIT,
            "sha256": expected[path],
        }
        committed = subprocess.run(
            ["git", "show", f"{SHARED_COMMIT}:{path.as_posix()}"],
            cwd=gipr.REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(committed).hexdigest() == expected[path]
        assert gipr.sha256_file(path) == expected[path]


def test_execution_seal_binds_frozen_authority() -> None:
    authority = seal()["authority"]
    assert authority["contract"] == {
        "path": gipr.CONTRACT_PATH.as_posix(),
        "commit": gipr.CONTRACT_COMMIT,
        "sha256": gipr.CONTRACT_SHA256,
    }
    assert authority["preregistration"] == {
        "path": gipr.PREREGISTRATION_PATH.as_posix(),
        "commit": gipr.PREREGISTRATION_COMMIT,
        "sha256": gipr.PREREGISTRATION_SHA256,
    }
    assert authority["preregistration_manifest_hash"] == (
        gipr.PREREGISTRATION_MANIFEST_HASH
    )
    assert authority["ethereum_helper"]["sha256"] == (
        gipr.ETHEREUM_HELPER_SHA256
    )


def test_runner_validates_committed_execution_seal() -> None:
    payload = gipr.validate_execution_seal()
    assert payload["seal_hash"] == SEAL_HASH
