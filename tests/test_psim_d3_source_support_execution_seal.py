from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_d3_source_support as runner,
)


SEAL_SHA256 = (
    "e2d4a503ac90fa971c7229aed0885862e48529fe17866065c2257de00f3fda50"
)
SEAL_HASH = (
    "1663fcc711bcde6d8e48a24434225957dfef5728f01ccb385051f96ccba3841b"
)
SHARED_COMMIT = "cf85aedaad0a0e2b15a440362d03702aad10175f"
RUNNER_SHA256 = (
    "a32f6fa3354a9765469985bcc78dc35fc67ac4d07b5216dc212c81b8e20d72dd"
)
TEST_SHA256 = (
    "a0e5dad8cb78d462828a63ab5b1a20fae9101cef4588dce40b8e3dcc78e9dc17"
)
CONTRACT_SHA256 = (
    "109b2fd3574048a181abd52c00e0a751835500babb559ca064e1c81f16fecf08"
)
SELF_CHECK_MANIFEST_HASH = (
    "d641559c614f3d3b32fdb4f41c75c47816271dbd45b832eff855566382e258b5"
)
SELF_CHECK_STDOUT_SHA256 = (
    "b8d4d6fa1e39c4ec8d2fb42cc4a1f333bbe05467bb0f1d6b4643914ca1909088"
)


def seal_bytes() -> bytes:
    return (runner.REPO_ROOT / runner.EXECUTION_SEAL_PATH).read_bytes()


def seal() -> dict:
    return json.loads(seal_bytes())


def test_execution_seal_is_canonical_and_source_blind() -> None:
    raw = seal_bytes()
    payload = seal()
    core = {
        key: value for key, value in payload.items() if key != "seal_hash"
    }
    assert hashlib.sha256(raw).hexdigest() == SEAL_SHA256
    assert raw == runner.canonical_json_bytes(payload)
    assert payload["seal_hash"] == SEAL_HASH == runner.canonical_hash(core)
    assert payload["protocol_version"] == runner.SEAL_PROTOCOL
    assert payload["policy_id"] == runner.POLICY_ID
    assert payload["forbidden_access"] == (
        runner.AccessLedger.zero().snapshot()
    )

    self_check = payload["synthetic_verification"]["self_check"]
    assert self_check["manifest_hash"] == SELF_CHECK_MANIFEST_HASH
    assert self_check["stdout_sha256"] == SELF_CHECK_STDOUT_SHA256
    assert self_check["network_calls"] == 0
    assert self_check["git_commands"] == 0
    assert self_check["source_event_rows_opened"] == 0
    assert self_check["official_source_opened"] is False
    assert self_check["outcomes_opened"] is False
    assert self_check["forbidden_access"] == (
        runner.AccessLedger.zero().snapshot()
    )
    assert self_check["transport_probe"][
        "no_lazy_fetch_semantic_probe_passed"
    ] is True
    assert self_check["transport_probe"]["single_fetch_invocations"] == 1
    assert self_check["transport_probe"]["access_boundary"] == {
        "market_data_accessed": False,
        "model_accessed": False,
        "official_eip_bip_source_accessed": False,
        "outcomes_accessed": False,
    }

    assert payload["synthetic_verification"]["pytest"] == {
        "argv": [
            ".venv/bin/pytest",
            "-q",
            *(path.as_posix() for path in runner.VERIFICATION_TEST_PATHS),
        ],
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": 0,
        "passed": 192,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_d3_implementation() -> None:
    payload = seal()
    assert payload["shared_commit"] == SHARED_COMMIT
    expected = {
        "runner": (runner.RUNNER_PATH, RUNNER_SHA256),
        "tests": (runner.TEST_PATH, TEST_SHA256),
    }
    for key, (path, digest) in expected.items():
        assert payload[key] == {
            "path": path.as_posix(),
            "commit": SHARED_COMMIT,
            "sha256": digest,
        }
        assert runner._git_blob_sha256(SHARED_COMMIT, path) == digest
        assert runner.sha256_file(path) == digest

    authority = payload["authority"]
    assert authority["implementation_contract"] == {
        "path": runner.IMPLEMENTATION_CONTRACT_PATH.as_posix(),
        "commit": SHARED_COMMIT,
        "sha256": CONTRACT_SHA256,
    }
    assert authority["core_runner"] == {
        "path": runner.D1_CORE_RUNNER_PATH.as_posix(),
        "commit": runner.D1_CORE_COMMIT,
        "sha256": runner.D1_CORE_RUNNER_SHA256,
    }
    assert authority["core_tests"] == {
        "path": runner.D1_CORE_TEST_PATH.as_posix(),
        "commit": runner.D1_CORE_COMMIT,
        "sha256": runner.D1_CORE_TEST_SHA256,
    }


def test_execution_seal_binds_frozen_d3_authority() -> None:
    authority = seal()["authority"]
    assert authority["decision"] == {
        "path": runner.DECISION_PATH.as_posix(),
        "commit": runner.DECISION_COMMIT,
        "sha256": runner.DECISION_SHA256,
    }
    assert authority["preregistration"] == {
        "path": runner.PREREGISTRATION_PATH.as_posix(),
        "commit": runner.PREREGISTRATION_COMMIT,
        "sha256": runner.PREREGISTRATION_SHA256,
    }
    assert authority["preregistration_producer"] == {
        "path": runner.PREREGISTRATION_SCRIPT_PATH.as_posix(),
        "commit": runner.PREREGISTRATION_COMMIT,
        "sha256": runner.PREREGISTRATION_SCRIPT_SHA256,
    }
    assert authority["preregistration_document"] == {
        "path": runner.PREREGISTRATION_DOC_PATH.as_posix(),
        "commit": runner.PREREGISTRATION_COMMIT,
        "sha256": runner.PREREGISTRATION_DOC_SHA256,
    }
    assert authority["d2_terminal_rejection"] == {
        "path": runner.D2_TERMINAL_PATH.as_posix(),
        "commit": runner.prereg.D2_TERMINAL_COMMIT,
        "sha256": runner.prereg.D2_TERMINAL_SHA256,
    }
    assert authority["transport_probe"] == {
        "path": runner.TRANSPORT_PROBE_PATH.as_posix(),
        "commit": runner.TRANSPORT_PROBE_COMMIT,
        "sha256": runner.TRANSPORT_PROBE_SHA256,
    }
    assert authority["preregistration_manifest_hash"] == (
        runner.PREREGISTRATION_MANIFEST_HASH
    )
    assert authority["authorized_delta_hash"] == (
        runner.prereg.AUTHORIZED_DELTA_HASH
    )
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.GIT_BINARY_SHA256,
        "version": runner.prereg.GIT_VERSION,
    }
    assert len(authority["source_authority_hash"]) == 64


def test_seal_commit_is_exact_direct_child_with_only_seal_paths() -> None:
    seal_commit = runner._assert_committed(runner.EXECUTION_SEAL_PATH)
    assert runner._git_output(
        "rev-list",
        "--parents",
        "-n",
        "1",
        seal_commit,
    ).split() == [seal_commit, SHARED_COMMIT]
    assert set(
        runner._git_output(
            "diff",
            "--name-only",
            SHARED_COMMIT,
            seal_commit,
        ).splitlines()
    ) == {
        runner.EXECUTION_SEAL_PATH.as_posix(),
        runner.SEAL_TEST_PATH.as_posix(),
    }
    assert runner._git_output(
        "cat-file",
        "-e",
        f"{seal_commit}:{runner.SEAL_TEST_PATH.as_posix()}",
    ) == ""
    assert runner._git_output(
        "merge-base",
        "--is-ancestor",
        seal_commit,
        "HEAD",
    ) == ""


def test_runner_validates_committed_execution_seal() -> None:
    seal_commit = runner._assert_committed(runner.EXECUTION_SEAL_PATH)
    if runner._git_output("rev-parse", "HEAD") == seal_commit:
        payload = runner.validate_execution_seal()
        assert payload == seal()
        assert payload["seal_hash"] == SEAL_HASH
        assert payload["shared_commit"] == SHARED_COMMIT
    else:
        terminal = runner.terminal_state()
        assert terminal is not None
        assert terminal["authority"]["execution_seal"]["seal_hash"] == (
            SEAL_HASH
        )
        assert terminal["authority"]["execution_seal"][
            "shared_commit"
        ] == SHARED_COMMIT


def test_seal_did_not_open_source_or_create_terminal_artifacts() -> None:
    rejection = runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH
    if rejection.exists():
        terminal = runner.terminal_state()
        assert terminal is not None
        assert terminal["decision"] == "reject"
        assert terminal["terminal_action"] == runner.FAILURE_ACTION
        assert not any(
            (runner.REPO_ROOT / path).exists()
            for path in (
                runner.DEFAULT_RESULT_PATH,
                runner.DEFAULT_EVENTS_PATH,
                runner.DEFAULT_CARDS_PATH,
                runner.DEFAULT_CONTROLS_PATH,
                runner.RUN_LOCK_PATH,
            )
        )
    else:
        assert not runner.DEFAULT_SOURCE_ROOT.exists()
        assert not any(
            (runner.REPO_ROOT / path).exists()
            for path in (
                runner.DEFAULT_RESULT_PATH,
                runner.DEFAULT_EVENTS_PATH,
                runner.DEFAULT_CARDS_PATH,
                runner.DEFAULT_CONTROLS_PATH,
                runner.RUN_LOCK_PATH,
            )
        )
