from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_d2_source_support as runner,
)


REJECTION_SHA256 = (
    "461ea699ada0d6873422c537e63f5fcff3bca56a436caae9aeff4bb74761ca24"
)
RESULT_HASH = (
    "b8134ab47a1c69916593d1092b9125e0a8a78da11cf3080660064b12a2e6387c"
)
SEAL_HASH = (
    "b6a101b2d6f41b70ac789ed243b8315589c109c4247d81e14c08d42c5aae0f27"
)
SEAL_COMMIT = "6ea1b283c6f35b8c8eeeac838869968aa7560ae0"


def rejection_bytes() -> bytes:
    return (
        runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH
    ).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_terminal_rejection_is_canonical_hash_bound_and_source_only() -> None:
    raw = rejection_bytes()
    payload = rejection()
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    assert hashlib.sha256(raw).hexdigest() == REJECTION_SHA256
    assert raw == runner.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == runner.canonical_hash(core)
    assert payload["protocol_version"] == runner.RESULT_PROTOCOL
    assert payload["policy_id"] == runner.POLICY_ID
    assert payload["decision"] == "reject"
    assert payload["terminal_action"] == runner.FAILURE_ACTION
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is True
    assert payload["artifacts"] is None


def test_first_three_source_gates_passed_and_gate_four_failed() -> None:
    payload = rejection()
    assert payload["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert [row["name"] for row in payload["gates"]] == list(
        runner.GATE_NAMES[:4]
    )
    assert [row["passed"] for row in payload["gates"]] == [
        True,
        True,
        True,
        False,
    ]
    assert payload["gates"][3] == {
        "name": "historical_blob_preamble_dependency_integrity",
        "passed": False,
        "metrics": {
            "gate_evaluation_completed": False,
            "error_type": "RuntimeError",
        },
        "failure": (
            "historical_blob_preamble_dependency_integrity "
            "raised RuntimeError"
        ),
    }
    assert payload["error"] == {"type": "RuntimeError"}


def test_bare_gate_one_succeeded_for_all_four_independent_roots() -> None:
    gate = rejection()["gates"][0]
    assert gate["name"] == "sealed_git_identity_and_object_integrity"
    assert gate["failure"] == ""
    assert set(gate["metrics"]["checks"].values()) == {True}
    receipts = gate["metrics"]["receipts"]
    assert {
        (row["protocol"], row["replica"], row["root_name"])
        for row in receipts
    } == {
        ("ethereum", "a", "ethereum-a.git"),
        ("ethereum", "b", "ethereum-b.git"),
        ("bitcoin", "a", "bitcoin-a.git"),
        ("bitcoin", "b", "bitcoin-b.git"),
    }
    assert all(
        row["is_bare_repository"] is True
        and row["is_inside_work_tree"] is False
        and row["git_status_invoked"] is False
        and row["checkout_created"] is False
        and row["forbidden_paths_absent"] is True
        and row["shared_object_alternates"] is False
        and row["ref_roster"]
        == ["refs/heads/master", runner.SEALED_REF]
        and row["sealed_ref"] == runner.SEALED_REF
        and row["sealed_tip"]
        == runner._repository_spec(row["protocol"]).sealed_tip
        and row["disk_used_gib"] <= runner.DISK_LIMIT_GIB
        for row in receipts
    )


def test_commit_and_path_replays_passed_identically_across_replicas() -> None:
    gates = rejection()["gates"]
    chain = gates[1]["metrics"]
    assert chain["ethereum"] == {
        "records": 6958,
        "replica_a_sha256": (
            "c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9"
        ),
        "replica_b_sha256": (
            "c022f028dfe9df0a9d36aeec173f227604d51243c0671a8cf090f687182b88d9"
        ),
        "effective_day_monotonic": True,
        "first_parent_continuity": True,
        "passed": True,
    }
    assert chain["bitcoin"] == {
        "records": 1482,
        "replica_a_sha256": (
            "7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2"
        ),
        "replica_b_sha256": (
            "7e60f24b78aa863a2b317a7dc3a32b2af8e367c3d25f4a97012f4ddfd28d89d2"
        ),
        "effective_day_monotonic": True,
        "first_parent_continuity": True,
        "passed": True,
    }

    groups = gates[2]["metrics"]
    assert groups["ethereum"]["groups"] == 4985
    assert groups["bitcoin"]["groups"] == 371
    assert groups["ethereum"]["replica_a_sha256"] == (
        groups["ethereum"]["replica_b_sha256"]
    )
    assert groups["ethereum"]["replica_a_sha256"] == (
        "a3eea9350bc5d0e1b6131515200cb771338063b7f673c971d67fa1684cda821c"
    )
    assert groups["bitcoin"]["replica_a_sha256"] == (
        groups["bitcoin"]["replica_b_sha256"]
    )
    assert groups["bitcoin"]["replica_a_sha256"] == (
        "3f7a8e10bb5f9ba57bb0231b5cd54a613fb81e67830c1ec1d9781fe0d22b6a8b"
    )
    assert not groups["ethereum"]["replica_a_issues"]
    assert not groups["ethereum"]["replica_b_issues"]
    assert not groups["bitcoin"]["replica_a_issues"]
    assert not groups["bitcoin"]["replica_b_issues"]


def test_access_ledger_records_partial_blob_transport_and_zero_outcomes() -> None:
    access = rejection()["access_ledger"]
    assert access["git_commands"] == 21_211
    assert access["network_commands"] == 13
    assert access["source_path_rows_opened"] == 16_312
    assert access["proposal_blobs_opened"] == 259
    assert access["proposal_text_rows_opened"] == 0
    assert access["daily_cards_built"] == 0
    assert access["pre_2020_proposal_blobs_opened"] == 0
    assert access["post_2023_proposal_blobs_opened"] == 0
    assert not any(
        access[name] for name in runner.FORBIDDEN_ACCESS_FIELDS
    )
    assert rejection()["counts"] == {"events": 0, "daily_cards": 0}


def test_source_audit_binds_exact_progress_without_repair() -> None:
    audit = rejection()["source_audit"]
    assert audit["repository_representation"] == (
        "BARE_OBJECT_DATABASE_NO_WORKTREE_NO_INDEX"
    )
    assert audit["source_traversal_ref"] == runner.SEALED_REF
    assert audit["source_classes_opened"] == [
        "git_remote_identity",
        "git_commit_metadata",
        "proposal_path_incidence",
        "proposal_blob_content",
    ]
    assert audit["remote_identity_opened"] is True
    assert audit["commit_metadata_opened"] is True
    assert audit["proposal_path_incidence_opened"] is True
    assert audit["proposal_blobs_opened"] is True
    assert audit["git_status_invoked"] is False
    assert audit["checkout_created"] is False
    assert audit["repair_or_provider_swap_used"] is False
    assert audit["source_run_attempt"] == 1
    assert audit["clone_receipts_sha256"] == (
        "c105b6abc919d91b4fa6d6104c6e119d5a23b4a08d8647fa789ea3818b965dd6"
    )
    assert audit["commit_chains_sha256"] == (
        "81ad9d409b2d640f0ac91ebc8b12cb7faa88b4af5c4046c3b93944784fd5a9b2"
    )
    assert audit["proposal_groups_sha256"] == (
        "4a9b099afc7bf1ce84fe94b896e734459e8c62405f0eecc14a73d4fe11e8a24c"
    )
    assert audit["events_sha256"] is None
    assert audit["cards_sha256"] is None
    assert audit["controls_sha256"] is None


def test_terminal_authority_binds_exact_seal_and_no_pass_artifacts() -> None:
    payload = rejection()
    seal = payload["authority"]["execution_seal"]
    assert seal["seal_hash"] == SEAL_HASH
    assert runner._assert_committed(runner.EXECUTION_SEAL_PATH) == (
        SEAL_COMMIT
    )
    assert seal["sha256"] == (
        "60dfe828df03751754d056366f977d6626ec00f3b795b5570f854918f022d800"
    )
    assert runner.terminal_state() == payload
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
