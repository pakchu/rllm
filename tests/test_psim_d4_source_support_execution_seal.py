from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_d4_source_support as runner,
)


SEAL_SHA256 = (
    "66a63c7c06fc1f19d85106ccaee04c1f2e384bf69f9a2cc5a9907d78c565b88a"
)
SEAL_HASH = (
    "097ad1112607f1f4e5b47ada4abfc11700d4532dff56a50afc243f4c597867da"
)
SHARED_COMMIT = "2d3216d5a144ba8eb694270301231850f0e015ca"
RUNNER_SHA256 = (
    "bcdb207e5627fba2640298aaac6897e650423ffa53171e6c192fd08821dc3ba2"
)
TEST_SHA256 = (
    "c1ab17623ceb8ba8a7c6749b3bea62a12911e4da265570c8a7116ad49624d26f"
)
CONTRACT_SHA256 = (
    "ad2aaedd23b6600b197549e77cc0d3f080dfb5cfa9fa026e14731e1c868b6470"
)
SELF_CHECK_MANIFEST_HASH = (
    "fa216eab3422ad86fdb3a9dc676199ed031927ef59722f5480f46b9ee0c1b93e"
)
SELF_CHECK_STDOUT_SHA256 = (
    "ee03f9747602ac7910ee35b391560ccb0553c60a22f29e4dd80712885d4699ea"
)
SOURCE_AUTHORITY_HASH = (
    "4d0368c6ff4cfb8f7c3508f1a214367734a4fee86b362971a10ad903d290f797"
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
    assert self_check["parser_probe"]["result_hash"] == (
        runner.PARSER_PROBE_RESULT_HASH
    )
    assert self_check["parser_probe"]["parser_version"] == (
        runner.prereg.PARSER_VERSION
    )
    assert self_check["parser_probe"]["synthetic_only"] is True
    assert self_check["parser_probe"]["access_boundary"] == {
        "d3_forensic_root_accessed": False,
        "d3_terminal_artifact_read": True,
        "market_data_accessed": False,
        "model_accessed": False,
        "official_historical_proposal_source_accessed": False,
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
        "passed": 289,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }


def test_execution_seal_binds_exact_d4_implementation() -> None:
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


def test_execution_seal_binds_frozen_d4_authority() -> None:
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
    assert authority["d3_terminal_rejection"] == {
        "path": runner.D3_TERMINAL_PATH.as_posix(),
        "commit": runner.prereg.D3_TERMINAL_COMMIT,
        "sha256": runner.prereg.D3_TERMINAL_SHA256,
    }
    assert authority["parser_probe"] == {
        "path": runner.PARSER_PROBE_PATH.as_posix(),
        "commit": runner.PARSER_PROBE_COMMIT,
        "sha256": runner.PARSER_PROBE_SHA256,
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
    assert authority["d3_runner"] == {
        "path": runner.D3_RUNNER_PATH.as_posix(),
        "commit": runner.D3_IMPLEMENTATION_COMMIT,
        "sha256": runner.D3_RUNNER_SHA256,
    }
    assert authority["d3_tests"] == {
        "path": runner.D3_TEST_PATH.as_posix(),
        "commit": runner.D3_IMPLEMENTATION_COMMIT,
        "sha256": runner.D3_TEST_SHA256,
    }
    assert authority["runtime"]["git"] == {
        "path": "/usr/bin/git",
        "sha256": runner.prereg.d3.GIT_BINARY_SHA256,
        "version": runner.prereg.d3.GIT_VERSION,
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
