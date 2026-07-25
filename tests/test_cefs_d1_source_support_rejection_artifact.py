from __future__ import annotations

import json

from training import build_cboe_edge_flip_sequence_policy_support as s


REJECTION_SHA256 = (
    "4c2839e5ac59738367d5116ff05ed50c900d16f06de8c4d4cc724fd25978c169"
)
RESULT_HASH = (
    "9963981f6d56fcff65f1367fc7c3c1fc006b60b821894b3e0ff59c6b9aa35d7b"
)


def rejection() -> dict:
    return json.loads(
        (s.REPOSITORY_ROOT / s.REJECTION_REPORT).read_text()
    )


def test_rejection_artifact_is_exact_and_self_consistent() -> None:
    payload = rejection()
    assert s.sha256_file(s.REJECTION_REPORT) == REJECTION_SHA256
    assert payload["result_hash"] == RESULT_HASH
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == s.canonical_hash(core)
    assert payload["decision"] == "fail"
    assert payload["failure_action"] == (
        "retire_cefs_d1_unchanged_before_outcomes"
    )
    assert payload["pass_action"] is None


def test_rejection_stopped_at_authority_before_source_decode() -> None:
    payload = rejection()
    assert payload["gates"] == [
        {
            "checks": {"authority_valid": False},
            "index": 1,
            "name": "authority_forbidden_access",
            "passed": False,
        }
    ]
    assert payload["details"] == {}
    assert payload["authority"] == {}
    assert payload["source_row_hash"] is None
    assert payload["control_row_hash"] is None
    assert payload["source_output"] is None
    assert payload["control_output"] is None
    assert payload["error"] == {
        "message": "[Errno 2] No such file or directory: 'git'",
        "type": "FileNotFoundError",
    }
    assert all(value == 0 for value in payload["forbidden_counters"].values())


def test_rejection_has_no_pass_artifact_and_is_idempotent() -> None:
    assert not (s.REPOSITORY_ROOT / s.SOURCE_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.CONTROL_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.PASS_REPORT).exists()
    payload = rejection()
    assert s.pre_run_terminal_state() == payload
    assert s.run_official() == payload
