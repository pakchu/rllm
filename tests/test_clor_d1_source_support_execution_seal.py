from __future__ import annotations

import hashlib
import json
import subprocess

from training import (
    build_collateral_liquidity_ordering_relation_source_support as s,
)


SEAL_SHA256 = (
    "2bddde9f0d4a92178b166bd3179c7abb33da4e7fe2cc6f2413937a04e09f40fe"
)
SEAL_HASH = (
    "464a99bb708eaa01300b9c41359e3586a2033477a1632f7e0cad54bb151089da"
)
SHARED_COMMIT = "92ae322a584eb93d621234c3a349fd80a9a22be5"
RUNNER_SHA256 = (
    "156aafff3fa99ce415408395c75c2ca3591835e457905ead0053e2707edcdb4a"
)
TEST_SHA256 = (
    "0f2750746af5276a2a43613957bb7547614823525c5b900abdf2f72c31b3e525"
)
SELF_CHECK_MANIFEST_HASH = (
    "88e94ec76c51081503afc77709ee5cf1851971e64ebaa8fc51373327ea0b3fae"
)
SELF_CHECK_STDOUT_SHA256 = (
    "561e2284a26979c8d2a22b399278cd1f2a037b28c553082deab66a9e5f0bd6ca"
)


def seal_bytes() -> bytes:
    return (s.REPOSITORY_ROOT / s.EXECUTION_SEAL_PATH).read_bytes()


def seal() -> dict:
    return json.loads(seal_bytes())


def test_execution_seal_is_canonical_and_source_blind() -> None:
    raw = seal_bytes()
    payload = seal()
    assert hashlib.sha256(raw).hexdigest() == SEAL_SHA256
    assert raw == s.json_bytes(payload)
    core = {key: value for key, value in payload.items() if key != "seal_hash"}
    assert payload["seal_hash"] == SEAL_HASH == s.canonical_hash(core)
    assert payload["protocol_version"] == s.SEAL_PROTOCOL
    assert payload["policy_id"] == s.POLICY_ID
    assert payload["forbidden_access"] == s.forbidden_access()
    self_check = payload["synthetic_verification"]["self_check"]
    assert self_check["source_value_rows_opened"] == 0
    assert self_check["predecessor_value_rows_opened"] == 0
    assert self_check["forbidden_access"] == s.forbidden_access()
    assert self_check["manifest_hash"] == SELF_CHECK_MANIFEST_HASH
    assert self_check["stdout_sha256"] == SELF_CHECK_STDOUT_SHA256
    pytest_record = payload["synthetic_verification"]["pytest"]
    assert pytest_record == {
        "argv": [".venv/bin/pytest", "-q", s.TEST_PATH],
        "environment_override": {"PYTHONPATH": "."},
        "exit_code": 0,
        "passed": 31,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
    }


def test_execution_seal_binds_committed_runner_and_tests() -> None:
    payload = seal()
    assert payload["shared_commit"] == SHARED_COMMIT
    expected = {
        s.RUNNER_PATH: RUNNER_SHA256,
        s.TEST_PATH: TEST_SHA256,
    }
    for key, path in (("runner", s.RUNNER_PATH), ("tests", s.TEST_PATH)):
        assert payload[key] == {
            "path": path,
            "commit": SHARED_COMMIT,
            "sha256": expected[path],
        }
        committed = subprocess.run(
            [s.prereg.GIT_EXECUTABLE, "show", f"{SHARED_COMMIT}:{path}"],
            cwd=s.REPOSITORY_ROOT,
            env=s.prereg._git_environment(),
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(committed).hexdigest() == expected[path]
        assert s.sha256_file(path) == expected[path]


def test_runner_validates_committed_execution_seal() -> None:
    payload = s.validate_execution_seal()
    assert payload["seal_hash"] == SEAL_HASH
