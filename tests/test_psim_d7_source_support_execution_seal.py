from __future__ import annotations

import hashlib
import json
import os

from training import (
    build_protocol_specification_intent_maturity_d7_source_support as runner,
)


SEAL_SHA256 = (
    "ea94ec6566b5925fb0be16bc30aae0e47f7215d42a202943e4d5213f144573d6"
)
SEAL_HASH = (
    "8088c0902479612bb7cc64f0c729c7375640fcb095bdd9c3d0fe62dcd35fa308"
)
SHARED_COMMIT = "0e8f22f2680a9edb2cf8497343444c16e4946df0"
RUNNER_SHA256 = (
    "75d4345a1d2e311a49bc7bec837f2345a6f630b5d5382485e2afb04cadb92a47"
)
TEST_SHA256 = (
    "a9e00d86bb48811f95cfb417daef70baa33e203150c6202753c01f8d3921e887"
)
CONTRACT_SHA256 = (
    "95b41b464f5c6af1cb3adc90fb8b1b50e7b4b1806b290931fe19d8cc1fb7d5e3"
)
SELF_CHECK_MANIFEST_HASH = (
    "c6489283b2a4bb843bbb386a78a0f16aa67837c3f2329d7ef9f6517a96a16caa"
)
SELF_CHECK_STDOUT_SHA256 = (
    "ac2feb6db2126071e0b2f126e74be79aeb4fc8e785612830d877178a6bc9e9cf"
)
SOURCE_AUTHORITY_HASH = (
    "98ebc81f94bb14b8dd4f8ae8b10ee9e2a514683f2aa418830fa968cd0e1e8745"
)


def seal_bytes() -> bytes:
    return (runner.REPO_ROOT / runner.EXECUTION_SEAL_PATH).read_bytes()


def seal() -> dict:
    return json.loads(seal_bytes())


def _pytest_epoch(
    paths: tuple,
    *,
    epoch: str,
    expected_passed: int,
) -> dict:
    return {
        "argv": [
            ".venv/bin/pytest",
            "-q",
            *(path.as_posix() for path in paths),
        ],
        "epoch": epoch,
        "environment_override": dict(
            sorted(runner.VERIFICATION_ENVIRONMENT_OVERRIDE.items())
        ),
        "expected_passed": expected_passed,
        "exit_code": 0,
        "passed": expected_passed,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


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
        "d6_census_artifact_read": True,
        "d6_forensic_root_accessed": False,
        "d6_run_invoked": False,
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

    inherited = _pytest_epoch(
        runner.INHERITED_VERIFICATION_TEST_PATHS,
        epoch="PRE_REBASE_INHERITED_AUTHORITY",
        expected_passed=runner.EXPECTED_INHERITED_VERIFICATION_PASSED,
    )
    inherited.update(
        {
            "authority_commit": runner.PRE_REBASE_AUTHORITY_COMMIT,
            "authority_ref": runner.PRE_REBASE_AUTHORITY_REF,
        }
    )
    current = _pytest_epoch(
        runner.CURRENT_D7_VERIFICATION_TEST_PATHS,
        epoch="CURRENT_D7_IMPLEMENTATION",
        expected_passed=runner.EXPECTED_CURRENT_D7_VERIFICATION_PASSED,
    )
    assert payload["synthetic_verification"]["pytest"] == {
        "protocol_version": (
            "psim_d7_dual_epoch_pytest_verification_v1"
        ),
        "inherited_pre_rebase": inherited,
        "current_d7": current,
        "totals": {
            "passed": 688,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
    }


def test_execution_seal_binds_exact_d7_implementation() -> None:
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


def test_execution_seal_binds_frozen_d7_authority() -> None:
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
    assert authority["d6_terminal_rejection"] == {
        "path": runner.D6_TERMINAL_PATH.as_posix(),
        "commit": runner.D6_TERMINAL_COMMIT,
        "sha256": runner.D6_TERMINAL_SHA256,
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
    assert authority["d6_runner"] == {
        "path": runner.D6_RUNNER_PATH.as_posix(),
        "commit": runner.D6_IMPLEMENTATION_COMMIT,
        "sha256": runner.D6_RUNNER_SHA256,
    }
    assert authority["d6_tests"] == {
        "path": runner.D6_TEST_PATH.as_posix(),
        "commit": runner.D6_IMPLEMENTATION_COMMIT,
        "sha256": runner.D6_TEST_SHA256,
    }
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.d6.d5.d4.d3.GIT_BINARY_SHA256,
        "version": runner.prereg.d6.d5.d4.d3.GIT_VERSION,
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


def test_seal_verification_environment_is_frozen() -> None:
    payload = seal()
    assert payload["synthetic_verification"]["pytest"]["totals"] == {
        "passed": 688,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    assert runner.VERIFICATION_ENVIRONMENT_OVERRIDE == {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": ".",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TZ": "UTC",
    }
