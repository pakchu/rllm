from __future__ import annotations

import hashlib
import json

from training import (
    build_protocol_specification_intent_maturity_d3_source_support as runner,
)


REJECTION_SHA256 = (
    "a9be5b5990ad79b7da7d72a22968f4f62a2700877b198606565cc70206fe9802"
)
RESULT_HASH = (
    "b00b54b70720d42d213b315e82e7ff3ad0df03909b92aaa514299e750fa1ba2c"
)
SEAL_HASH = (
    "1663fcc711bcde6d8e48a24434225957dfef5728f01ccb385051f96ccba3841b"
)
SEAL_SHA256 = (
    "e2d4a503ac90fa971c7229aed0885862e48529fe17866065c2257de00f3fda50"
)
SEAL_COMMIT = "4e931cbda848e6914e912bdd10eeb35c250dd821"
TERMINAL_DOCUMENT = (
    runner.REPO_ROOT
    / "docs/psim-d3-source-support-terminal-rejection-2026-07-26.md"
)


def rejection_bytes() -> bytes:
    return (
        runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH
    ).read_bytes()


def rejection() -> dict:
    return json.loads(rejection_bytes())


def test_terminal_rejection_is_canonical_hash_bound_and_source_only() -> None:
    assert runner.DEFAULT_REJECTION_PATH.as_posix() == (
        "results/protocol_specification_intent_maturity_d3_source_rejection_"
        "2026-07-25.json"
    )
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
            "error_type": "ValueError",
        },
        "failure": (
            "historical_blob_preamble_dependency_integrity "
            "raised ValueError"
        ),
    }
    assert payload["error"] == {"type": "ValueError"}


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
        and row["fetch_head_absent"] is True
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


def test_batch_hydration_succeeded_before_frozen_parser_rejection() -> None:
    audit = rejection()["source_audit"]
    receipts = audit["batch_hydration_receipts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    receipt_core = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_hash"
    }
    assert receipt["receipt_hash"] == runner.canonical_hash(receipt_core)
    assert receipt["receipt_hash"] == (
        "4c273231fb5a4f4675c35107b68d65d13ee16e97443deae6f6a95c451d7b2e3e"
    )
    assert receipt["repository_root_name"] == "ethereum-a.git"
    assert receipt["passed"] is True
    assert receipt["stage"] == "complete"
    assert receipt["fetch_invocations"] == 1
    assert receipt["requested_blob_count"] == 5206
    assert receipt["new_total_object_count"] == 5206
    assert receipt["new_pack_count"] == 1
    assert receipt["new_promisor_count"] == 1
    assert receipt["new_loose_object_count"] == 0
    assert receipt["maintenance_child_processes"] == 0
    assert receipt["post_read_fetch_child_processes"] == 0
    assert receipt["refs_unchanged"] is True
    assert receipt["fetch_head_absent"] is True
    assert receipt["post_read_object_store_unchanged"] is True
    assert receipt["hydrated_snapshot_hash"] == (
        receipt["post_read_snapshot_hash"]
    )
    assert audit["batch_hydration_receipts_sha256"] == (
        "17a0bf4fd13dc45eca96eee1430acd3689587b0b65495473634851eb4ca47032"
    )
    assert audit["batch_hydration_receipts_sha256"] == (
        runner.canonical_hash(receipts)
    )


def test_access_ledger_records_one_hydration_and_zero_outcomes() -> None:
    access = rejection()["access_ledger"]
    assert access["git_commands"] == 21_225
    assert access["network_commands"] == 13
    assert access["source_path_rows_opened"] == 16_312
    assert access["proposal_blobs_opened"] == 5_206
    assert access["proposal_text_rows_opened"] == 17
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
        "d1891c178e68a71ffee4c45c83030d07a4f0b9e896fa90334a77aceed05ee565"
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
    assert seal["sha256"] == SEAL_SHA256
    assert runner._assert_committed(runner.EXECUTION_SEAL_PATH) == (
        SEAL_COMMIT
    )
    assert payload["authority"]["source_authority_hash"] == (
        "d443b354a990b8b58c152fc75712bec0af6a1fb6d44727d5013abb52422dfd93"
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


def test_terminal_document_records_transport_success_and_parser_cause() -> None:
    text = TERMINAL_DOCUMENT.read_text(encoding="utf-8")
    for value in (
        REJECTION_SHA256,
        RESULT_HASH,
        SEAL_HASH,
        SEAL_SHA256,
        SEAL_COMMIT,
        "requested blobs             5,206",
        "PSIM blank line inside header",
        "EIPS/eip-2378.md",
        "b788f38a216ca4cfea9d9de8ccfcf4cf658c8950",
        "ac34c07b91d6dffa14922951473f50dd587eb900",
        "a2fd3d87db7861f2b50739bf6c9015b968abc6fb6ffee7629492626034f41bb1",
        "without invoking\n`run` again",
        "PSIM-D3 is terminally rejected unchanged",
    ):
        assert value in text
