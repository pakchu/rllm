from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_source_support as psim,
)


SEAL_SHA256 = (
    "6f387a86e60cf2b11c9b49a043c71d35825490993b1741d806704fc2f2020914"
)
SEAL_HASH = (
    "c26397920fa1137845f5dea56eab72cb1a8d4ead401e7ee3e249c5c1e39aa506"
)
SHARED_COMMIT = "80b656994f17548a7a599a548e23e9f1cd01302d"
SEAL_COMMIT = "d537ef0e3254f157daf197f6effbac73945a4034"
RUNNER_SHA256 = (
    "414e83256b3ea489a9e1cd0995f6061e5fab550cd12c795ef7e88eff8998d9fb"
)
TEST_SHA256 = (
    "343aa1a72cfbca23d9756988ced042b5c61a6e8fc5a21a0b6d18e45870e906e9"
)
CONTRACT_SHA256 = (
    "33c26a6d36e09872b714729aeba63b34659a59becf4ccdadfc9a05040ace0fb1"
)
SELF_CHECK_MANIFEST_HASH = (
    "24ad04222852e97ffbd37067102cb52b2e38d5d992fd4641ab416b0670168a61"
)
SELF_CHECK_STDOUT_SHA256 = (
    "4acc071bee5de333c804da59273d5d0ad1fcfc4e735e6f0ac78b5c1539e65a88"
)


def seal_bytes() -> bytes:
    return (psim.REPO_ROOT / psim.EXECUTION_SEAL_PATH).read_bytes()


def seal() -> dict:
    return json.loads(seal_bytes())


def test_execution_seal_is_canonical_and_source_blind() -> None:
    raw = seal_bytes()
    payload = seal()
    core = {
        key: value for key, value in payload.items() if key != "seal_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == SEAL_SHA256
    assert raw == psim.canonical_json_bytes(payload)
    assert payload["seal_hash"] == SEAL_HASH == psim.canonical_hash(core)
    assert payload["protocol_version"] == psim.SEAL_PROTOCOL
    assert payload["policy_id"] == psim.POLICY_ID
    assert payload["forbidden_access"] == psim.AccessLedger.zero().snapshot()

    self_check = payload["synthetic_verification"]["self_check"]
    assert self_check["manifest_hash"] == SELF_CHECK_MANIFEST_HASH
    assert self_check["stdout_sha256"] == SELF_CHECK_STDOUT_SHA256
    assert self_check["network_calls"] == 0
    assert self_check["source_event_rows_opened"] == 0
    assert self_check["outcomes_opened"] is False
    assert self_check["forbidden_access"] == (
        psim.AccessLedger.zero().snapshot()
    )
    assert payload["synthetic_verification"]["pytest"] == {
        "argv": [
            ".venv/bin/pytest",
            "-q",
            psim.TEST_PATH.as_posix(),
        ],
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": 0,
        "passed": 35,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_implementation_files() -> None:
    payload = seal()
    assert payload["shared_commit"] == SHARED_COMMIT
    expected = {
        "runner": (psim.RUNNER_PATH, RUNNER_SHA256),
        "tests": (psim.TEST_PATH, TEST_SHA256),
    }
    for key, (path, digest) in expected.items():
        assert payload[key] == {
            "path": path.as_posix(),
            "commit": SHARED_COMMIT,
            "sha256": digest,
        }
        assert psim._git_blob_sha256(SHARED_COMMIT, path) == digest
        assert psim.sha256_file(path) == digest

    contract = payload["authority"]["implementation_contract"]
    assert contract == {
        "path": psim.IMPLEMENTATION_CONTRACT_PATH.as_posix(),
        "commit": SHARED_COMMIT,
        "sha256": CONTRACT_SHA256,
    }
    assert psim._git_blob_sha256(
        SHARED_COMMIT,
        psim.IMPLEMENTATION_CONTRACT_PATH,
    ) == CONTRACT_SHA256


def test_execution_seal_binds_frozen_preregistration_authority() -> None:
    authority = seal()["authority"]
    assert authority["decision"] == {
        "path": psim.DECISION_PATH.as_posix(),
        "commit": psim.DECISION_COMMIT,
        "sha256": psim.DECISION_SHA256,
    }
    assert authority["preregistration"] == {
        "path": psim.PREREGISTRATION_PATH.as_posix(),
        "commit": psim.PREREGISTRATION_COMMIT,
        "sha256": psim.PREREGISTRATION_SHA256,
    }
    assert authority["preregistration_producer"] == {
        "path": psim.PREREGISTRATION_SCRIPT_PATH.as_posix(),
        "commit": psim.PREREGISTRATION_COMMIT,
        "sha256": psim.PREREGISTRATION_SCRIPT_SHA256,
    }
    assert authority["preregistration_document"] == {
        "path": psim.PREREGISTRATION_DOC_PATH.as_posix(),
        "commit": psim.PREREGISTRATION_COMMIT,
        "sha256": psim.PREREGISTRATION_DOC_SHA256,
    }
    assert authority["preregistration_manifest_hash"] == (
        psim.PREREGISTRATION_MANIFEST_HASH
    )
    assert len(authority["source_authority_hash"]) == 64


def test_seal_commit_is_exact_direct_child_with_only_seal_paths() -> None:
    seal_commit = psim._assert_committed(psim.EXECUTION_SEAL_PATH)
    assert seal_commit == SEAL_COMMIT
    assert psim._git_output(
        "rev-list",
        "--parents",
        "-n",
        "1",
        seal_commit,
    ).split() == [seal_commit, SHARED_COMMIT]
    assert set(
        psim._git_output(
            "diff",
            "--name-only",
            SHARED_COMMIT,
            seal_commit,
        ).splitlines()
    ) == {
        psim.EXECUTION_SEAL_PATH.as_posix(),
        psim.SEAL_TEST_PATH.as_posix(),
    }
    assert psim._git_output(
        "merge-base",
        "--is-ancestor",
        seal_commit,
        "HEAD",
    ) == ""


def test_runner_validates_committed_execution_seal() -> None:
    if psim._git_output("rev-parse", "HEAD") == SEAL_COMMIT:
        payload = psim.validate_execution_seal()
        assert payload == seal()
        assert payload["seal_hash"] == SEAL_HASH
    else:
        terminal = psim.terminal_state()
        assert terminal is not None
        assert terminal["authority"]["execution_seal"]["seal_hash"] == SEAL_HASH
        assert terminal["authority"]["execution_seal"]["shared_commit"] == (
            SHARED_COMMIT
        )


def test_seal_did_not_create_source_or_terminal_artifacts() -> None:
    if (psim.REPO_ROOT / psim.DEFAULT_REJECTION_PATH).exists():
        terminal = psim.terminal_state()
        assert terminal is not None
        assert terminal["decision"] == "reject"
        assert not any(
            (psim.REPO_ROOT / path).exists()
            for path in (
                psim.DEFAULT_RESULT_PATH,
                psim.DEFAULT_EVENTS_PATH,
                psim.DEFAULT_CARDS_PATH,
                psim.DEFAULT_CONTROLS_PATH,
                psim.RUN_LOCK_PATH,
            )
        )
    else:
        assert not psim.DEFAULT_SOURCE_ROOT.exists()
        assert not any(
            (psim.REPO_ROOT / path).exists()
            for path in (
                psim.DEFAULT_RESULT_PATH,
                psim.DEFAULT_EVENTS_PATH,
                psim.DEFAULT_CARDS_PATH,
                psim.DEFAULT_CONTROLS_PATH,
                psim.RUN_LOCK_PATH,
            )
        )
