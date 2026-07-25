from __future__ import annotations

import json

from training import build_london_cash_derivative_path_source_support as s


REJECTION_PATH = (
    "results/lcdp_d1_source_support_rejection_2026-07-25.json"
)
REJECTION_SHA256 = (
    "eb93e2a2dc1c7660a230dfc098a44ece96af3a2c94c18f391e8f664271841b9b"
)
RESULT_HASH = (
    "7986295a8b7aceb96a827bcb14ab4bffb2b46a3121d6d290a68f01c454544c8e"
)


def rejection() -> dict:
    return json.loads((s.REPOSITORY_ROOT / REJECTION_PATH).read_text())


def test_rejection_artifact_is_exact_and_self_consistent() -> None:
    payload = rejection()
    assert s.sha256_file(REJECTION_PATH) == REJECTION_SHA256
    assert payload["result_hash"] == RESULT_HASH
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == s.canonical_hash(core)
    assert payload["decision"] == "fail"
    assert payload["failure_action"] == (
        "retire_lcdp_d1_unchanged_before_outcomes"
    )
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False


def test_rejection_stopped_exactly_at_source_validity() -> None:
    payload = rejection()
    assert [
        (gate["name"], gate["passed"]) for gate in payload["gates"]
    ] == [
        ("protocol_source_integrity", True),
        ("calendar_dst_integrity", True),
        ("source_validity", False),
    ]
    assert payload["gates"][-1]["checks"] == {
        "annual": False,
        "quarterly": False,
    }
    assert set(payload["details"]) == {"parser_audit", "source_validity"}
    assert payload["source_token_row_hash"] is None
    assert payload["append_replay"] is None
    assert payload["token_output"] is None


def test_rejection_validity_counts_match_frozen_gate() -> None:
    validity = rejection()["details"]["source_validity"]
    assert validity["annual"]["2020"] == {
        "valid": 355,
        "total": 366,
        "share": 355 / 366,
    }
    assert validity["annual"]["2021"] == {
        "valid": 363,
        "total": 365,
        "share": 363 / 365,
    }
    assert validity["annual"]["2022"] == {
        "valid": 364,
        "total": 365,
        "share": 364 / 365,
    }
    assert validity["quarterly"]["2020Q3"] == {
        "valid": 87,
        "total": 92,
        "share": 87 / 92,
    }


def test_rejection_opened_no_forbidden_evidence_or_pass_output() -> None:
    payload = rejection()
    assert all(value == 0 for value in payload["forbidden_counters"].values())
    assert not (s.REPOSITORY_ROOT / s.TOKEN_OUTPUT).exists()
    assert not (s.REPOSITORY_ROOT / s.PASS_REPORT).exists()
    assert payload["error"] is None


def test_rejection_binds_sealed_runner_and_tests() -> None:
    authority = rejection()["authority"]
    assert authority["runner"] == {
        "commit": "92f9fe11cd1047340c042c2b1ec3796add6523bf",
        "path": s.RUNNER_PATH,
        "sha256": (
            "d1fa16c8b57154e8102902f17bf7032e65a8f4cfc5cc5098b561d390cb285bda"
        ),
    }
    assert authority["tests"] == {
        "commit": "92f9fe11cd1047340c042c2b1ec3796add6523bf",
        "path": s.TEST_PATH,
        "sha256": (
            "51c015292c683ef55e090e9e7d5bf32f21fb4828102b3d8b69fe6c9f0445dbcf"
        ),
    }
    assert authority["execution_seal"]["manifest_hash"] == (
        "55aa5c0081b23c2e0789e57085db8bf095729ff627581c5775d1fc5a028904d1"
    )
