from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from training import (
    preregister_protocol_specification_intent_maturity_d5 as d5,
)
from training import (
    preregister_protocol_specification_intent_maturity_d6 as d6,
)
from training import (
    probe_protocol_specification_intent_maturity_d6_mechanism as mechanism,
)


RESULT_PATH = (
    d6.REPO_ROOT
    / "results/protocol_specification_intent_maturity_d6_"
    "preregistration_2026-07-26.json"
)
PREREGISTRATION_DOCUMENT_PATH = (
    d6.REPO_ROOT
    / "docs/psim-d6-source-support-preregistration-2026-07-26.md"
)
RESULT_SHA256 = (
    "9b6177ba02bf02783f7ddffe90cf4c5f1e385422ff658e17b28bf72d2f051d82"
)


def contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_d5_and_d6_authorities_are_exactly_bound() -> None:
    payload = d6.build_preregistration()
    inheritance = payload["inheritance_proof"]

    d5_registration = d6._read_canonical_json(
        d6.D5_PREREGISTRATION_PATH
    )
    assert d5_registration == d5.build_preregistration()
    assert d6.sha256_file(d6.D5_PREREGISTRATION_PATH) == (
        d6.D5_PREREGISTRATION_SHA256
    )
    assert inheritance["d5_preregistration"] == {
        "path": d6.D5_PREREGISTRATION_PATH.as_posix(),
        "commit": d6.D5_PREREGISTRATION_COMMIT,
        "sha256": d6.D5_PREREGISTRATION_SHA256,
        "manifest_hash": d6.D5_PREREGISTRATION_MANIFEST_HASH,
        "contract_core_hash": d6.canonical_hash(
            d6._contract_core(d5_registration)
        ),
        "producer": {
            "path": d6.D5_PREREGISTRATION_SCRIPT_PATH.as_posix(),
            "sha256": d6.D5_PREREGISTRATION_SCRIPT_SHA256,
        },
        "test": {
            "path": d6.D5_PREREGISTRATION_TEST_PATH.as_posix(),
            "sha256": d6.D5_PREREGISTRATION_TEST_SHA256,
        },
        "document": {
            "path": d6.D5_PREREGISTRATION_DOCUMENT_PATH.as_posix(),
            "sha256": d6.D5_PREREGISTRATION_DOCUMENT_SHA256,
        },
    }

    terminal = d6._read_canonical_json(d6.D5_TERMINAL_PATH)
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert terminal["error"] == {"type": "ValueError"}
    assert terminal["outcomes_opened"] is False
    assert terminal["profitability_result"] is False
    assert inheritance["d5_terminal_rejection"]["result_hash"] == (
        d6.D5_TERMINAL_RESULT_HASH
    )

    probe = d6._read_canonical_json(d6.MECHANISM_PROBE_PATH)
    assert probe == mechanism.build_probe()
    assert inheritance["d6_mechanism_probe"][
        "mechanism_contract_hash"
    ] == d6.canonical_hash(d6.MECHANISM_PROBE_CONTRACT)
    assert inheritance["d5_post_terminal_census"] == (
        probe["d5_census_binding"]
    )


def test_candidate_scope_does_not_authorize_official_execution() -> None:
    payload = d6.build_preregistration()

    assert payload["protocol_version"] == (
        "psim_d6_source_preregistration_v1"
    )
    assert payload["candidate"]["id"] == "PSIM-D6"
    assert payload["candidate"]["stage"] == "source_support_only"
    assert payload["candidate"]["selection_commit"] == d6.DECISION_COMMIT
    assert payload["execution_authorization_contract"] == (
        d6.EXECUTION_AUTHORIZATION_CONTRACT
    )
    assert payload["execution_authorization_contract"][
        "official_source_execution_authorized_by_this_preregistration"
    ] is False
    assert payload["execution_authorization_contract"][
        "synthetic_mechanism_probe_authorizes_official_execution"
    ] is False
    assert payload["next_authorized_step"] == (
        "implement, test, review, and seal a synthetic-only PSIM-D6 "
        "source-support evaluator; this preregistration does not authorize "
        "official source execution"
    )


def test_d5_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d5_payload = d5.build_preregistration()
    d6_payload = d6.build_preregistration()
    successor = contract_core(d6_payload)
    delta = d6._diff_values(contract_core(d5_payload), successor)
    inheritance = d6_payload["inheritance_proof"]

    assert len(delta) == 42
    assert tuple(sorted(delta)) == tuple(
        sorted(d6.AUTHORIZED_DELTA_PATHS)
    )
    assert delta == inheritance["authorized_delta"]
    assert inheritance["authorized_delta_paths"] == list(
        d6.AUTHORIZED_DELTA_PATHS
    )
    assert d6.canonical_hash(delta) == d6.AUTHORIZED_DELTA_HASH
    assert inheritance["authorized_delta_hash"] == (
        d6.AUTHORIZED_DELTA_HASH
    )
    assert inheritance["all_other_contract_paths_byte_equal"] is True


def test_mechanism_overlay_is_probe_equal_and_hash_bound() -> None:
    payload = d6.build_preregistration()
    overlay = payload["event_contract"]["d6_source_mechanisms"]
    probe = mechanism.build_probe()

    assert overlay == d6.SOURCE_MECHANISM_CONTRACT
    assert overlay["mechanism_probe_contract"] == (
        probe["mechanism_contract"]
    )
    assert overlay["mechanism_version"] == probe["mechanism_version"]
    assert overlay["base_semantics"] == (
        "PSIM_D5_UNCHANGED_EXCEPT_THIS_EXACT_FROZEN_D6_OVERLAY"
    )
    assert d6.canonical_hash(overlay) == (
        d6.SOURCE_MECHANISM_CONTRACT_HASH
    )
    assert payload["inheritance_proof"][
        "source_mechanism_contract_hash"
    ] == d6.SOURCE_MECHANISM_CONTRACT_HASH


def test_migration_restoration_is_exact_receipt_bound_and_hidden() -> None:
    migration = d6.build_preregistration()["event_contract"][
        "d6_source_mechanisms"
    ]["migration_restoration"]
    authority = migration["authority"]

    assert authority == {
        "episode_count": 365,
        "episode_receipt_manifest_hash": (
            d6.D5_EPISODE_RECEIPT_MANIFEST_HASH
        ),
        "episode_roster_hash": d6.D5_EPISODE_ROSTER_HASH,
        "proposal_roster_hash": d6.D5_MIGRATION_PROPOSAL_ROSTER_HASH,
        "source_census_commit": d6.D5_CENSUS_COMMIT,
        "source_census_path": d6.D5_CENSUS_PATH.as_posix(),
        "source_census_result_hash": d6.D5_CENSUS_RESULT_HASH,
        "source_census_sha256": d6.D5_CENSUS_SHA256,
    }
    assert migration[
        "generic_administrative_to_valid_transition_authorized"
    ] is False
    assert migration["exact_path_and_blob_continuity_required"] is True
    assert migration["authorized_sequence"]["commit_oids"] == list(
        mechanism.MIGRATION_COMMIT_SEQUENCE
    )
    assert migration["authorized_sequence"]["effective_days"] == list(
        mechanism.MIGRATION_DAY_SEQUENCE
    )
    assert migration["authorized_sequence"]["blob_classes"] == [
        list(row) for row in mechanism.MIGRATION_CLASS_SEQUENCE
    ]
    assert migration["authority_receipt_hashes_model_visible"] is False
    assert migration["administrative_events_model_visible"] is False
    assert migration["model_payload_for_restoration"] == {
        "administrative_quarantined": True,
        "model_visibility": "ADMINISTRATIVE_QUARANTINE",
        "normalized_text_delta_chunks": [],
    }
    assert "caller" not in json.dumps(migration, sort_keys=True).lower()


def test_chunk_transport_is_lossless_bounded_and_not_aggregated() -> None:
    payload = d6.build_preregistration()
    representation = payload["representation_contract"]
    daily = payload["daily_relation_contract"]
    transport = representation["model_text_transport_contract"]

    assert representation["model_text_field"] == (
        "normalized_text_delta_chunks"
    )
    assert transport == d6.MODEL_TEXT_TRANSPORT_CONTRACT
    assert transport["chunk_payload_fields"] == [
        "chunk_count",
        "chunk_index",
        "normalized_text_delta_chunk",
    ]
    assert transport["max_bytes_per_chunk"] == 8_192
    assert transport["max_chunks_per_event"] == 8
    assert transport["max_bytes_per_event"] == 65_536
    assert transport["full_text_reconstruction"] == (
        "BYTE_FOR_BYTE_REQUIRED"
    )
    assert transport["canonical_partition_validation_required"] is True
    assert transport["strict_utf8_required"] is True
    assert transport["audit_receipt_model_visible"] is False
    assert transport[
        "chunk_payloads_are_transport_fragments_not_events_or_labels"
    ] is True
    assert transport["model_aggregation_policy"] == (
        "UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION"
    )
    assert "NO_TRUNCATION_OR_SUMMARIZATION" in (
        transport["ninth_chunk_action"]
    )
    assert "LINE_IS_OPAQUE_AFTER_THE_SECOND_PIPE" in (
        transport["line_serialization_semantics"]
    )
    assert daily["maximum_model_text_bytes_per_chunk"] == 8_192
    assert daily["maximum_model_text_chunks_per_event"] == 8
    assert daily["maximum_model_text_bytes_per_event"] == 65_536
    assert daily["model_text_field"] == (
        "normalized_text_delta_chunks"
    )
    assert payload["memorization_contract"][
        "model_text_chunk_aggregation_policy"
    ] == transport["model_aggregation_policy"]


def test_gate_four_collects_complete_typed_roster_before_rejection() -> None:
    payload = d6.build_preregistration()
    support = payload["source_support_contract"]
    totality = support["gate_four_totality_contract"]

    assert totality == d6.GATE_FOUR_TOTALITY_CONTRACT
    assert totality["decision_after_complete_roster_only"] is True
    assert totality[
        "event_semantics_exception_may_abort_roster_collection"
    ] is False
    assert totality["event_semantics_outcome_per_event_required"] is True
    assert totality["replica_outcome_roster_identity_required"] is True
    assert totality[
        "canonical_rejection_required_before_return_or_raise"
    ] is True
    assert totality["error_report_raw_or_normalized_text_allowed"] is False
    assert totality["semantic_error_terminal_action"] == d6.FAILURE_ACTION
    assert totality["complete_roster_scope"] == (
        "ALL_RETAINED_2020_2023_PROPOSAL_GROUP_EVENTS_IN_ALL_FOUR_"
        "FRESH_REPLICAS_AFTER_SUCCESSFUL_HYDRATION"
    )
    assert "COMPLETE_TYPED_ERROR_ROSTER_AND_REJECT" in (
        support["gate_four_semantics"]
    )


def test_d5_semantics_remain_base_and_unknowns_never_reach_model() -> None:
    d5_event = d5.build_preregistration()["event_contract"]
    d6_event = copy.deepcopy(
        d6.build_preregistration()["event_contract"]
    )
    overlay = d6_event.pop("d6_source_mechanisms")

    assert d6_event == d5_event
    assert overlay["migration_restoration"][
        "unknown_or_mutated_episode_action"
    ].endswith("BEFORE_MODEL_OR_OUTCOMES")
    assert overlay["gate_four_totality"][
        "error_report_raw_or_normalized_text_allowed"
    ] is False
    assert overlay["migration_restoration"][
        "authority_receipt_hashes_model_visible"
    ] is False


def test_d5_hydration_mechanics_are_identical_after_namespace_rebase() -> None:
    payload = d6.build_preregistration()
    contract = payload["source_contract"]["batch_hydration_contract"]

    assert contract == d6.BATCH_HYDRATION_CONTRACT
    assert d6._transport_contract_rebased_to_d5(contract) == (
        d5.BATCH_HYDRATION_CONTRACT
    )
    assert d6.canonical_hash(contract) == (
        d6.BATCH_HYDRATION_CONTRACT_HASH
    )
    assert contract["forbidden_transports"][-1] == (
        "D1, D2, D3, D4, or D5 source-object reuse"
    )
    assert contract["one_fetch_invocation_per_replica"] is True
    assert contract["post_hydration_read"][
        "object_store_ref_and_fetch_head_snapshot_must_be_unchanged"
    ] is True


def test_fresh_d6_root_refs_and_artifacts_forbid_d5_reuse() -> None:
    source = d6.build_preregistration()["source_contract"]

    assert source["source_root"] == "/tmp/psim-d6-source"
    assert source["source_root"] != "/tmp/psim-d5-source"
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d6/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d6/sealed-tip",
    ]
    assert all(
        value.startswith(
            (
                "results/protocol_specification_intent_maturity_d6_",
                "data/protocol_specification_intent_maturity_d6_",
            )
        )
        for value in source["artifact_paths"].values()
    )
    assert d6.build_preregistration()["execution_authorization_contract"][
        "d5_forensic_or_source_root_reuse_allowed"
    ] is False


def test_gate_control_and_unrelated_contract_rosters_are_unchanged() -> None:
    d5_payload = d5.build_preregistration()
    d6_payload = d6.build_preregistration()

    assert d6_payload["source_support_contract"]["gates_in_order"] == (
        d5_payload["source_support_contract"]["gates_in_order"]
    )
    assert d6_payload["source_support_contract"]["relation_controls"] == (
        d5_payload["source_support_contract"]["relation_controls"]
    )
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "split_contract",
    ):
        assert d6_payload[key] == d5_payload[key]


def test_preregistration_has_zero_forbidden_access() -> None:
    payload = d6.build_preregistration()
    access = payload["inheritance_proof"]["preregistration_access"]

    assert access == {
        "git_commands": 0,
        "network_calls": 0,
        "d5_preregistration_artifact_read": True,
        "d5_terminal_artifact_read": True,
        "d5_census_artifact_read": True,
        "d6_mechanism_probe_artifact_read": True,
        "d5_forensic_root_opened": False,
        "d5_source_runner_invoked": False,
        "d6_official_source_execution_invoked": False,
        "official_historical_proposal_source_opened": False,
        "market_model_outcomes_opened": False,
        "raw_official_text_published": False,
    }


def test_manifest_hash_binds_exact_core_and_replays() -> None:
    payload = d6.build_preregistration()
    core = copy.deepcopy(payload)
    manifest_hash = core.pop("manifest_hash")

    assert manifest_hash == d6.canonical_hash(core)
    assert payload == d6.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = RESULT_PATH.read_bytes()

    assert d6.sha256_bytes(raw) == RESULT_SHA256
    assert raw == d6.canonical_json_bytes(d6.build_preregistration())
    assert json.loads(raw) == d6.build_preregistration()


def test_write_is_canonical_idempotent_and_rejects_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(d6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        d6,
        "build_preregistration",
        lambda: {"manifest_hash": "unit"},
    )
    destination = Path("results/unit.json")

    first = d6.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d6.write_preregistration(destination)

    assert first == second == tmp_path / destination
    assert first_bytes == d6.canonical_json_bytes(
        {"manifest_hash": "unit"}
    )

    first.write_text('{"changed":true}\\n', encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D6 preregistration differs",
    ):
        d6.write_preregistration(destination)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/absolute.json",
        "unit.json",
        "results/nested/unit.json",
        "results/unit.txt",
    ],
)
def test_output_boundary_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(
        RuntimeError,
        match="safe repo-local result",
    ):
        d6._safe_output_path(path)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        (
            "DECISION_SHA256",
            "0" * 64,
            "decision authority changed",
        ),
        (
            "D5_PREREGISTRATION_SHA256",
            "0" * 64,
            "preregistration authority changed",
        ),
        (
            "D5_TERMINAL_SHA256",
            "0" * 64,
            "terminal authority changed",
        ),
        (
            "MECHANISM_PROBE_SHA256",
            "0" * 64,
            "mechanism probe authority changed",
        ),
    ],
)
def test_authority_mutations_fail_closed(
    name: str,
    replacement: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d6, name, replacement)
    with pytest.raises(RuntimeError, match=message):
        d6.build_preregistration()


def test_delta_and_mechanism_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d6, "SOURCE_ROOT", "/tmp/psim-d6-mutated")
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d6.build_preregistration()

    monkeypatch.undo()
    mutated = copy.deepcopy(d6.SOURCE_MECHANISM_CONTRACT)
    mutated["migration_restoration"][
        "generic_administrative_to_valid_transition_authorized"
    ] = True
    monkeypatch.setattr(d6, "SOURCE_MECHANISM_CONTRACT", mutated)
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d6.build_preregistration()


def test_hydration_or_execution_scope_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated_hydration = copy.deepcopy(d6.BATCH_HYDRATION_CONTRACT)
    mutated_hydration["timeout_seconds"] += 1
    monkeypatch.setattr(
        d6,
        "BATCH_HYDRATION_CONTRACT",
        mutated_hydration,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed",
    ):
        d6.build_preregistration()

    monkeypatch.undo()
    mutated_execution = copy.deepcopy(
        d6.EXECUTION_AUTHORIZATION_CONTRACT
    )
    mutated_execution[
        "official_source_execution_authorized_by_this_preregistration"
    ] = True
    monkeypatch.setattr(
        d6,
        "EXECUTION_AUTHORIZATION_CONTRACT",
        mutated_execution,
    )
    with pytest.raises(
        RuntimeError,
        match="authorized source delta hash changed",
    ):
        d6.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d6.REPO_ROOT / d6.SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported.isdisjoint(
        {
            "aiohttp",
            "binance",
            "ccxt",
            "git",
            "httpx",
            "models",
            "numpy",
            "pandas",
            "requests",
            "sklearn",
            "subprocess",
            "torch",
            "transformers",
            "urllib",
            "yfinance",
        }
    )


def test_preregistration_document_binds_machine_contract() -> None:
    text = PREREGISTRATION_DOCUMENT_PATH.read_text(encoding="utf-8")
    payload = d6.build_preregistration()

    assert "PSIM-D6" in text
    assert d6.DECISION_COMMIT in text
    assert d6.D5_TERMINAL_RESULT_HASH in text
    assert d6.D5_CENSUS_RESULT_HASH in text
    assert d6.MECHANISM_PROBE_RESULT_HASH in text
    assert payload["manifest_hash"] in text
    assert d6.AUTHORIZED_DELTA_HASH in text
    assert d6.SOURCE_MECHANISM_CONTRACT_HASH in text
    assert d6.BATCH_HYDRATION_CONTRACT_HASH in text
    assert d6.EXECUTION_AUTHORIZATION_CONTRACT_HASH in text
    assert RESULT_SHA256 in text
    assert d6.sha256_file(d6.SCRIPT_PATH) in text
    assert d6.sha256_file(Path(__file__).relative_to(d6.REPO_ROOT)) in text
    assert "/tmp/psim-d6-source" in text
    assert "does not authorize official source execution" in text
    assert "UNDECIDED_NOT_AUTHORIZED_BY_D6_PREREGISTRATION" in text
