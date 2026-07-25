from __future__ import annotations

import json

from training import build_cboe_edge_flip_sequence_policy_d2_support as s


REJECTION_SHA256 = (
    "7c4ee86ad540ad1eefb92d35859948818a0637dc1be7ddbb2d527cfb6f2924bb"
)
RESULT_HASH = (
    "ade0d792d7231693482ab713c9f848fcce0aa1f3abafb0d0c14aed959358ea2b"
)
SOURCE_ROW_HASH = (
    "4c0f7c4bf398a5f0fa7266a025d72e7bc6fae65a8d3bb6bf0df67c678c4b3c89"
)


def rejection() -> dict:
    return json.loads((s.REPOSITORY_ROOT / s.REJECTION_REPORT).read_text())


def test_rejection_artifact_is_exact_and_self_consistent() -> None:
    payload = rejection()
    assert s.sha256_file(s.REJECTION_REPORT) == REJECTION_SHA256
    assert payload["result_hash"] == RESULT_HASH
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == s.canonical_hash(core)
    assert payload["decision"] == "fail"
    assert payload["failure_action"] == (
        "retire_cefs_d2_unchanged_before_outcomes"
    )
    assert payload["pass_action"] is None
    assert payload["error"] is None


def test_rejection_stopped_at_gate_four_on_exact_eval_edge_collapse() -> None:
    payload = rejection()
    assert [
        (gate["index"], gate["name"], gate["passed"])
        for gate in payload["gates"]
    ] == [
        (1, "runtime_authority_forbidden_access", True),
        (2, "schema_chronology", True),
        (3, "schedule_support", True),
        (4, "primitive_edge_support", False),
    ]
    failed_checks = {
        name
        for name, passed in payload["gates"][-1]["checks"].items()
        if not passed
    }
    assert failed_checks == {
        "eval_term_back_level_max_share",
        "eval_term_back_level_two_levels",
    }
    assert set(payload["details"]) == {
        "parser",
        "schedule",
        "edge_support",
    }

    expected = {
        "TRAIN": {
            "rows": 498,
            "counts": {"EQUAL": 0, "HIGHER": 55, "LOWER": 443},
            "shares": {
                "EQUAL": 0.0,
                "HIGHER": 0.11044176706827309,
                "LOWER": 0.8895582329317269,
            },
        },
        "TEST": {
            "rows": 251,
            "counts": {"EQUAL": 0, "HIGHER": 15, "LOWER": 236},
            "shares": {
                "EQUAL": 0.0,
                "HIGHER": 0.05976095617529881,
                "LOWER": 0.9402390438247012,
            },
        },
        "EVAL": {
            "rows": 250,
            "counts": {"EQUAL": 0, "HIGHER": 0, "LOWER": 250},
            "shares": {"EQUAL": 0.0, "HIGHER": 0.0, "LOWER": 1.0},
        },
    }
    for split, split_expected in expected.items():
        split_detail = payload["details"]["edge_support"][split]
        assert split_detail["rows"] == split_expected["rows"]
        assert (
            split_detail["edges"]["TERM_BACK_LEVEL"]["counts"]
            == split_expected["counts"]
        )
        assert (
            split_detail["edges"]["TERM_BACK_LEVEL"]["shares"]
            == split_expected["shares"]
        )


def test_rejection_preserves_outcome_blind_stage_boundary() -> None:
    payload = rejection()
    assert payload["source_row_hash"] == SOURCE_ROW_HASH
    assert payload["control_row_hash"] is None
    assert payload["source_output"] is None
    assert payload["control_output"] is None
    assert all(value == 0 for value in payload["forbidden_counters"].values())
    assert not (s.REPOSITORY_ROOT / s.SOURCE_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.CONTROL_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.PASS_REPORT).exists()


def test_rejection_is_terminal_and_idempotent() -> None:
    payload = rejection()
    assert s.pre_run_terminal_state() == payload
    assert s.run_official() == payload
