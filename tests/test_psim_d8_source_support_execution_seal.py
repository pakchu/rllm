from __future__ import annotations

import hashlib
import json
import os

from training import (
    build_protocol_specification_intent_maturity_d8_source_support as runner,
)


SEAL_SHA256 = (
    "c63951fddbae7aabf0eaa51edaacfdfc67203b004580d080189eb8635648f9df"
)
SEAL_HASH = (
    "f8f7ac92585227a3430008e4d68c170b48729798e99773866b63a2596059587b"
)
SHARED_COMMIT = "17e17fa96ddb7866ffda0d67727b8630737188f5"
RUNNER_SHA256 = (
    "b30128ce4856da9dc9546e306d0cc29f6975595b3f282fee55431f984677c93b"
)
TEST_SHA256 = (
    "fdf02ed5f4c38db7ef656fb6a030a84619e399d4fcdbb8ffe7721a19f13153c8"
)
CONTRACT_SHA256 = (
    "6ee911369be44441daaeb1ff1da1627efd43b43d9b82533e42a119bdd1c00058"
)
SELF_CHECK_MANIFEST_HASH = (
    "e695843d90b341839cf0b09dc292295e0536ffdf09b89ba3f10dbfe5c89de1bd"
)
SELF_CHECK_STDOUT_SHA256 = (
    "c6df181cf20f9601f7174fe77600a3cd680d23fb1be4f6190dbe2a9173d3d118"
)
SOURCE_AUTHORITY_HASH = (
    "78d6f1dbfe5d5f5cbecb749b9f600f38a25b018e6b81ee8f4914457dbf02e8f7"
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


def _binding(path, commit: str, digest: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "commit": commit,
        "sha256": digest,
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

    grammar = self_check["d7_grammar_mechanism_probe"]
    assert (
        grammar["result_hash"]
        == runner.D7_GRAMMAR_MECHANISM_PROBE_RESULT_HASH
    )
    assert grammar["mechanism_version"] == runner.prereg.d7.MECHANISM_VERSION
    assert grammar["synthetic_only"] is True
    assert grammar["access_boundary"] == {
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

    mechanism = self_check["mechanism_probe"]
    assert (
        mechanism["result_hash"]
        == runner.RELATION_SUBCARD_MECHANISM_PROBE_RESULT_HASH
    )
    assert mechanism["mechanism_version"] == runner.prereg.MECHANISM_VERSION
    assert mechanism["synthetic_only"] is True
    assert mechanism["access_boundary"] == {
        "d7_forensic_artifact_read": True,
        "d7_forensic_source_root_accessed": False,
        "d7_run_invoked": False,
        "d7_terminal_artifact_read": True,
        "external_network_accessed": False,
        "historical_proposal_text_accessed": False,
        "market_data_accessed": False,
        "model_accessed": False,
        "outcomes_accessed": False,
        "reward_trade_pnl_accessed": False,
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
        runner.CURRENT_D8_VERIFICATION_TEST_PATHS,
        epoch="CURRENT_D8_IMPLEMENTATION",
        expected_passed=runner.EXPECTED_CURRENT_D8_VERIFICATION_PASSED,
    )
    assert payload["synthetic_verification"]["pytest"] == {
        "protocol_version": "psim_d8_dual_epoch_pytest_verification_v1",
        "inherited_pre_rebase": inherited,
        "current_d8": current,
        "totals": {
            "passed": 719,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
    }


def test_execution_seal_binds_exact_d8_implementation() -> None:
    payload = seal()
    assert payload["shared_commit"] == SHARED_COMMIT
    expected = {
        "runner": (runner.RUNNER_PATH, RUNNER_SHA256),
        "tests": (runner.TEST_PATH, TEST_SHA256),
    }
    for key, (path, digest) in expected.items():
        assert payload[key] == _binding(path, SHARED_COMMIT, digest)
        assert runner._git_blob_sha256(SHARED_COMMIT, path) == digest
        assert runner.sha256_file(path) == digest

    authority = payload["authority"]
    assert authority["implementation_contract"] == _binding(
        runner.IMPLEMENTATION_CONTRACT_PATH,
        SHARED_COMMIT,
        CONTRACT_SHA256,
    )
    assert authority["core_runner"] == _binding(
        runner.D1_CORE_RUNNER_PATH,
        runner.D1_CORE_COMMIT,
        runner.D1_CORE_RUNNER_SHA256,
    )
    assert authority["core_tests"] == _binding(
        runner.D1_CORE_TEST_PATH,
        runner.D1_CORE_COMMIT,
        runner.D1_CORE_TEST_SHA256,
    )


def test_execution_seal_binds_frozen_d7_and_d8_authority() -> None:
    authority = seal()["authority"]
    expected = {
        "decision": (
            runner.DECISION_PATH,
            runner.DECISION_COMMIT,
            runner.DECISION_SHA256,
        ),
        "preregistration": (
            runner.PREREGISTRATION_PATH,
            runner.PREREGISTRATION_COMMIT,
            runner.PREREGISTRATION_SHA256,
        ),
        "preregistration_producer": (
            runner.PREREGISTRATION_SCRIPT_PATH,
            runner.PREREGISTRATION_COMMIT,
            runner.PREREGISTRATION_SCRIPT_SHA256,
        ),
        "preregistration_tests": (
            runner.PREREGISTRATION_TEST_PATH,
            runner.PREREGISTRATION_COMMIT,
            runner.PREREGISTRATION_TEST_SHA256,
        ),
        "preregistration_document": (
            runner.PREREGISTRATION_DOC_PATH,
            runner.PREREGISTRATION_COMMIT,
            runner.PREREGISTRATION_DOC_SHA256,
        ),
        "d6_terminal_rejection": (
            runner.D6_TERMINAL_PATH,
            runner.D6_TERMINAL_COMMIT,
            runner.D6_TERMINAL_SHA256,
        ),
        "d7_terminal_rejection": (
            runner.D7_TERMINAL_PATH,
            runner.D7_TERMINAL_COMMIT,
            runner.D7_TERMINAL_SHA256,
        ),
        "d7_grammar_mechanism_probe": (
            runner.D7_GRAMMAR_MECHANISM_PROBE_PATH,
            runner.D7_GRAMMAR_MECHANISM_PROBE_COMMIT,
            runner.D7_GRAMMAR_MECHANISM_PROBE_SHA256,
        ),
        "d7_grammar_mechanism_probe_producer": (
            runner.D7_GRAMMAR_MECHANISM_PROBE_SCRIPT_PATH,
            runner.D7_GRAMMAR_MECHANISM_PROBE_COMMIT,
            runner.D7_GRAMMAR_MECHANISM_PROBE_SCRIPT_SHA256,
        ),
        "d7_grammar_mechanism_probe_tests": (
            runner.D7_GRAMMAR_MECHANISM_PROBE_TEST_PATH,
            runner.D7_GRAMMAR_MECHANISM_PROBE_COMMIT,
            runner.D7_GRAMMAR_MECHANISM_PROBE_TEST_SHA256,
        ),
        "mechanism_probe": (
            runner.RELATION_SUBCARD_MECHANISM_PROBE_PATH,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_COMMIT,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_SHA256,
        ),
        "mechanism_probe_producer": (
            runner.RELATION_SUBCARD_MECHANISM_PROBE_SCRIPT_PATH,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_COMMIT,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_SCRIPT_SHA256,
        ),
        "mechanism_probe_tests": (
            runner.RELATION_SUBCARD_MECHANISM_PROBE_TEST_PATH,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_COMMIT,
            runner.RELATION_SUBCARD_MECHANISM_PROBE_TEST_SHA256,
        ),
        "transport_probe": (
            runner.TRANSPORT_PROBE_PATH,
            runner.TRANSPORT_PROBE_COMMIT,
            runner.TRANSPORT_PROBE_SHA256,
        ),
        "d6_runner": (
            runner.D6_RUNNER_PATH,
            runner.D6_IMPLEMENTATION_COMMIT,
            runner.D6_RUNNER_SHA256,
        ),
        "d6_tests": (
            runner.D6_TEST_PATH,
            runner.D6_IMPLEMENTATION_COMMIT,
            runner.D6_TEST_SHA256,
        ),
        "d7_runner": (
            runner.D7_RUNNER_PATH,
            runner.D7_IMPLEMENTATION_COMMIT,
            runner.D7_RUNNER_SHA256,
        ),
        "d7_tests": (
            runner.D7_TEST_PATH,
            runner.D7_IMPLEMENTATION_COMMIT,
            runner.D7_TEST_SHA256,
        ),
    }
    for key, (path, commit, digest) in expected.items():
        assert authority[key] == _binding(path, commit, digest)

    assert authority["preregistration_manifest_hash"] == (
        runner.PREREGISTRATION_MANIFEST_HASH
    )
    assert authority["authorized_delta_hash"] == (
        runner.prereg.AUTHORIZED_DELTA_HASH
    )
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.d7.d6.d5.d4.d3.GIT_BINARY_SHA256,
        "version": runner.prereg.d7.d6.d5.d4.d3.GIT_VERSION,
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
    seal_commit = runner._assert_committed(runner.EXECUTION_SEAL_PATH)
    head = runner._git_output("rev-parse", "HEAD")
    if head == seal_commit:
        assert not runner.DEFAULT_SOURCE_ROOT.exists()
        assert not any(
            (runner.REPO_ROOT / path).exists()
            for path in (
                runner.DEFAULT_RESULT_PATH,
                runner.DEFAULT_REJECTION_PATH,
                runner.DEFAULT_EVENTS_PATH,
                runner.DEFAULT_CARDS_PATH,
                runner.DEFAULT_CONTROLS_PATH,
                runner.RUN_LOCK_PATH,
            )
        )
    else:
        terminal = runner.terminal_state()
        assert terminal is not None
        assert terminal["decision"] in {"pass", "reject"}
        assert terminal["authority"]["execution_seal"]["seal_hash"] == (
            SEAL_HASH
        )
        assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()


def test_seal_verification_environment_is_frozen() -> None:
    payload = seal()
    assert payload["synthetic_verification"]["pytest"]["totals"] == {
        "passed": 719,
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
