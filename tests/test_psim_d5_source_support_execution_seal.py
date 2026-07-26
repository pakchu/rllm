from __future__ import annotations

import hashlib
import json
import os

from training import (
    build_protocol_specification_intent_maturity_d5_source_support as runner,
)


SEAL_SHA256 = (
    "daac77d152a77645f7255832140e233273fd08de3da35ce7c2dea990d8c74bd0"
)
SEAL_HASH = (
    "0fadc88c8802503a10447eee957358545b412af54086cfd020f804a025b397b7"
)
SHARED_COMMIT = "90e7740edcd68a3b4c3acf8e9fe9a14f9e4eb8e1"
RUNNER_SHA256 = (
    "744959177c1f18d62cb920f5bd9c1068eb5415c07d4f7d5719af5b37542e0dba"
)
TEST_SHA256 = (
    "45785da991420161348b276c1fd299d5dd865c1782b31d40ddb0aa1649038fa4"
)
CONTRACT_SHA256 = (
    "418022b6643db965b464f1b843f856c7d67db5220a80367b899077662ed4b7c8"
)
SELF_CHECK_MANIFEST_HASH = (
    "2e35e38696a22d768dd6ca3f3c7bd5cee6361f9a1bfa4f6f42ede828c4957109"
)
SELF_CHECK_STDOUT_SHA256 = (
    "0329d4418e14f4a6b8d4dbb6f2e6c4a5588fb29a31769c40fee8a4a957403f7c"
)
SOURCE_AUTHORITY_HASH = (
    "ae3629c549b8f100e6d699972bba10d50743b49767099f6444c7202fac93b6d8"
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
    assert self_check["transport_probe"]["result_hash"] == (
        runner.TRANSPORT_PROBE_RESULT_HASH
    )
    assert self_check["transport_probe"]["synthetic_only"] is True
    assert self_check["transport_probe"]["access_boundary"] == {
        "market_data_accessed": False,
        "model_accessed": False,
        "official_eip_bip_source_accessed": False,
        "outcomes_accessed": False,
    }
    assert self_check["semantics_probe"]["result_hash"] == (
        runner.SEMANTICS_PROBE_RESULT_HASH
    )
    assert self_check["semantics_probe"]["semantics_version"] == (
        runner.prereg.SEMANTICS_VERSION
    )
    assert self_check["semantics_probe"]["synthetic_only"] is True
    assert self_check["semantics_probe"]["access_boundary"] == {
        "d4_census_artifact_read": True,
        "d4_forensic_root_accessed": False,
        "d4_terminal_artifact_read": True,
        "external_network_accessed_by_probe": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
        "official_reference_research_preexisted_probe": True,
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
        "passed": 384,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_d5_implementation() -> None:
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


def test_execution_seal_binds_frozen_d5_authority() -> None:
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
    assert authority["d4_terminal_rejection"] == {
        "path": runner.D4_TERMINAL_PATH.as_posix(),
        "commit": runner.prereg.D4_TERMINAL_COMMIT,
        "sha256": runner.prereg.D4_TERMINAL_SHA256,
    }
    assert authority["semantics_probe"] == {
        "path": runner.SEMANTICS_PROBE_PATH.as_posix(),
        "commit": runner.SEMANTICS_PROBE_COMMIT,
        "sha256": runner.SEMANTICS_PROBE_SHA256,
    }
    assert authority["semantics_probe_producer"] == {
        "path": runner.SEMANTICS_PROBE_SCRIPT_PATH.as_posix(),
        "commit": runner.SEMANTICS_PROBE_COMMIT,
        "sha256": runner.SEMANTICS_PROBE_SCRIPT_SHA256,
    }
    assert authority["semantics_probe_tests"] == {
        "path": runner.SEMANTICS_PROBE_TEST_PATH.as_posix(),
        "commit": runner.SEMANTICS_PROBE_COMMIT,
        "sha256": runner.SEMANTICS_PROBE_TEST_SHA256,
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
    assert authority["d4_runner"] == {
        "path": runner.D4_RUNNER_PATH.as_posix(),
        "commit": runner.D4_IMPLEMENTATION_COMMIT,
        "sha256": runner.D4_RUNNER_SHA256,
    }
    assert authority["d4_tests"] == {
        "path": runner.D4_TEST_PATH.as_posix(),
        "commit": runner.D4_IMPLEMENTATION_COMMIT,
        "sha256": runner.D4_TEST_SHA256,
    }
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.d4.d3.GIT_BINARY_SHA256,
        "version": runner.prereg.d4.d3.GIT_VERSION,
    }
    assert authority["source_authority_hash"] == SOURCE_AUTHORITY_HASH


def test_seal_commit_is_exact_direct_child_with_only_seal_paths() -> None:
    seal_commit = runner._assert_committed(runner.EXECUTION_SEAL_PATH)
    assert runner._git_output(
        "rev-list", "--parents", "-n", "1", seal_commit
    ).split() == [seal_commit, SHARED_COMMIT]
    assert set(
        runner._git_output(
            "diff", "--name-only", SHARED_COMMIT, seal_commit
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
        "merge-base", "--is-ancestor", seal_commit, "HEAD"
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
