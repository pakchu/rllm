from __future__ import annotations

import hashlib
import json
import os

from training import (
    build_protocol_specification_intent_maturity_d6_source_support as runner,
)


SEAL_SHA256 = (
    "cf9bdbea467a499c6075059ef9275f00699fb0431fa27643751539ffdea64e1d"
)
SEAL_HASH = (
    "5c9bb27b63375dd4e9bf7f7345115f8d8bf8910a84693a9c15b5c306c6bc2e54"
)
SHARED_COMMIT = "5c3f3f6d26046a8bc7b2f7ad09178d944d61e17b"
RUNNER_SHA256 = (
    "bc78fb2ff6ac0b4f0cebaedd01d03a75830f97be81cd9a736e47e6ead46a9f8f"
)
TEST_SHA256 = (
    "eab0dad5e99d3480825f4f007d7b51b45d3f94552b7e1e81c80ca065a1a85fa3"
)
CONTRACT_SHA256 = (
    "14399f13a74519209bdf1575555e72c0946280827ba6dc43c937120e7c8aaf32"
)
SELF_CHECK_MANIFEST_HASH = (
    "e727425630c36049201b05d27c7f1679e67917589233e0c1d0ebadfe648d236d"
)
SELF_CHECK_STDOUT_SHA256 = (
    "7b9952c1d632f69e5c4a46e9f186ec72475a6cc0936eef0a9f62ac1dc0212f23"
)
SOURCE_AUTHORITY_HASH = (
    "b63b42232d387af3ef9471ae6656857375ed4e94c548d510bdd4a874a9a9e963"
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
    mechanism = self_check["mechanism_probe"]
    assert mechanism["result_hash"] == runner.MECHANISM_PROBE_RESULT_HASH
    assert mechanism["mechanism_version"] == runner.prereg.MECHANISM_VERSION
    assert mechanism["synthetic_only"] is True
    assert mechanism["access_boundary"] == {
        "d5_census_artifact_read": True,
        "d5_forensic_root_accessed": False,
        "d5_run_invoked": False,
        "external_network_accessed_by_probe": False,
        "historical_proposal_text_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
        "outcomes_accessed": False,
        "raw_official_text_published": False,
    }
    transport = self_check["transport_probe"]
    assert transport["result_hash"] == runner.TRANSPORT_PROBE_RESULT_HASH
    assert transport["synthetic_only"] is True
    assert transport["access_boundary"] == {
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
        "passed": 523,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_d6_implementation() -> None:
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


def test_execution_seal_binds_frozen_d6_authority() -> None:
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
    assert authority["preregistration_tests"] == {
        "path": runner.PREREGISTRATION_TEST_PATH.as_posix(),
        "commit": runner.PREREGISTRATION_COMMIT,
        "sha256": runner.PREREGISTRATION_TEST_SHA256,
    }
    assert authority["preregistration_document"] == {
        "path": runner.PREREGISTRATION_DOC_PATH.as_posix(),
        "commit": runner.PREREGISTRATION_COMMIT,
        "sha256": runner.PREREGISTRATION_DOC_SHA256,
    }
    assert authority["d5_terminal_rejection"] == {
        "path": runner.D5_TERMINAL_PATH.as_posix(),
        "commit": runner.prereg.D5_TERMINAL_COMMIT,
        "sha256": runner.prereg.D5_TERMINAL_SHA256,
    }
    assert authority["mechanism_probe"] == {
        "path": runner.MECHANISM_PROBE_PATH.as_posix(),
        "commit": runner.MECHANISM_PROBE_COMMIT,
        "sha256": runner.MECHANISM_PROBE_SHA256,
    }
    assert authority["mechanism_probe_producer"] == {
        "path": runner.MECHANISM_PROBE_SCRIPT_PATH.as_posix(),
        "commit": runner.MECHANISM_PROBE_COMMIT,
        "sha256": runner.MECHANISM_PROBE_SCRIPT_SHA256,
    }
    assert authority["mechanism_probe_tests"] == {
        "path": runner.MECHANISM_PROBE_TEST_PATH.as_posix(),
        "commit": runner.MECHANISM_PROBE_COMMIT,
        "sha256": runner.MECHANISM_PROBE_TEST_SHA256,
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
    assert authority["d5_runner"] == {
        "path": runner.D5_RUNNER_PATH.as_posix(),
        "commit": runner.D5_IMPLEMENTATION_COMMIT,
        "sha256": runner.D5_RUNNER_SHA256,
    }
    assert authority["d5_tests"] == {
        "path": runner.D5_TEST_PATH.as_posix(),
        "commit": runner.D5_IMPLEMENTATION_COMMIT,
        "sha256": runner.D5_TEST_SHA256,
    }
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.d5.d4.d3.GIT_BINARY_SHA256,
        "version": runner.prereg.d5.d4.d3.GIT_VERSION,
    }
    assert authority["source_authority_hash"] == SOURCE_AUTHORITY_HASH


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
        payload = (
            seal()
            if os.environ.get(runner.SEAL_TEST_CHILD_ENV) == "1"
            else runner.validate_execution_seal()
        )
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
