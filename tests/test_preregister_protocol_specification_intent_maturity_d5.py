from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_protocol_specification_intent_maturity as d1
from training import (
    preregister_protocol_specification_intent_maturity_d4 as d4,
)
from training import (
    preregister_protocol_specification_intent_maturity_d5 as d5,
)
from training import (
    probe_protocol_specification_intent_maturity_d5_event_semantics as probe,
)


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d5_preregistration_"
    "2026-07-26.json"
)
RESULT_SHA256 = (
    "11465540d59181bc48ea28c5164579847cbd936bf005c69d874ec2c873c949b9"
)
MANIFEST_HASH = (
    "f08eeb300fceb906cdcde485b4bce184c48d4cb14a1cd9028046e0c21a287309"
)
PREREGISTRATION_DOCUMENT_PATH = Path(
    "docs/psim-d5-source-support-preregistration-2026-07-26.md"
)


def contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_decision_d4_terminal_and_semantics_probe_are_exactly_bound() -> None:
    assert d5.DECISION_COMMIT == (
        "0e62ec05e6861b2619e6737dd594e7306ad7c93a"
    )
    assert d5.sha256_file(d5.DECISION_PATH) == d5.DECISION_SHA256
    assert d5.sha256_file(d5.D4_PREREGISTRATION_PATH) == (
        d5.D4_PREREGISTRATION_SHA256
    )
    assert d5.sha256_file(d5.D4_TERMINAL_PATH) == d5.D4_TERMINAL_SHA256
    assert d5.sha256_file(d5.SEMANTICS_PROBE_PATH) == (
        d5.SEMANTICS_PROBE_SHA256
    )
    assert d5.sha256_file(d5.SEMANTICS_PROBE_SCRIPT_PATH) == (
        d5.SEMANTICS_PROBE_SCRIPT_SHA256
    )
    assert d5.sha256_file(d5.SEMANTICS_PROBE_TEST_PATH) == (
        d5.SEMANTICS_PROBE_TEST_SHA256
    )

    terminal = d5._read_canonical_json(d5.D4_TERMINAL_PATH)
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert terminal["access_ledger"]["proposal_blobs_opened"] == 5_206
    assert terminal["access_ledger"]["proposal_text_rows_opened"] == 44
    assert terminal["outcomes_opened"] is False

    semantics = d5._read_canonical_json(d5.SEMANTICS_PROBE_PATH)
    assert semantics == probe.build_probe()
    assert semantics["result_hash"] == d5.SEMANTICS_PROBE_RESULT_HASH
    assert semantics["semantics_contract"] == d5.PROBE_SEMANTICS_CONTRACT
    assert semantics["synthetic_only"] is True


def test_candidate_and_next_step_are_source_only() -> None:
    payload = d5.build_preregistration()

    assert payload["protocol_version"] == (
        "psim_d5_source_preregistration_v1"
    )
    assert payload["candidate"] == {
        "id": "PSIM-D5",
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "exact path identity plus normalized text-delta semantics"
        ),
        "selection_commit": d5.DECISION_COMMIT,
        "source_axis": "official_eip_bip_specification_revision_relation",
        "stage": "source_support_only",
    }
    assert payload["next_authorized_step"] == (
        "implement and seal synthetic-only PSIM-D5 path/text-delta "
        "source-support evaluator"
    )


def test_d4_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d4_payload = d4.build_preregistration()
    d5_payload = d5.build_preregistration()
    successor = contract_core(d5_payload)
    delta = d5._diff_values(contract_core(d4_payload), successor)
    inheritance = d5_payload["inheritance_proof"]

    assert tuple(sorted(delta)) == tuple(
        sorted(d5.AUTHORIZED_DELTA_PATHS)
    )
    assert len(delta) == 52
    assert delta == inheritance["authorized_delta"]
    assert inheritance["authorized_delta_paths"] == list(
        d5.AUTHORIZED_DELTA_PATHS
    )
    assert d5.canonical_hash(delta) == d5.AUTHORIZED_DELTA_HASH
    assert inheritance["authorized_delta_hash"] == (
        d5.AUTHORIZED_DELTA_HASH
    )
    assert inheritance["all_other_contract_paths_byte_equal"] is True


def test_event_semantics_contract_is_probe_equal_and_hash_bound() -> None:
    payload = d5.build_preregistration()
    contract = payload["event_contract"]["d5_source_semantics"]

    assert contract == d5.EVENT_SEMANTICS_CONTRACT
    assert contract["semantics_version"] == probe.SEMANTICS_VERSION
    assert contract["probe_semantics"] == d5.PROBE_SEMANTICS_CONTRACT
    assert contract["synthetic_probe_binding"] == (
        d5.SEMANTICS_PROBE_BINDING
    )
    assert d5.canonical_hash(contract) == (
        d5.EVENT_SEMANTICS_CONTRACT_HASH
    )
    assert payload["inheritance_proof"][
        "event_semantics_contract_hash"
    ] == d5.EVENT_SEMANTICS_CONTRACT_HASH
    assert sum(
        contract["historical_ethereum_blob_class_counts"].values()
    ) == 5_206


def test_model_boundary_excludes_identifiers_metadata_and_admin_events() -> (
    None
):
    payload = d5.build_preregistration()
    parser = payload["parser_contract"]
    representation = payload["representation_contract"]
    daily = payload["daily_relation_contract"]
    event = payload["event_contract"]["d5_source_semantics"]

    assert parser["declared_status_is_model_visible"] is False
    assert representation[
        "raw_proposal_number_hash_path_timestamp_date_author_url_allowed"
    ] is False
    assert representation["model_text_field"] == "normalized_text_delta"
    assert representation["legacy_intent_text_field_allowed"] is False
    assert representation["model_text_sections"] == list(
        d5.MODEL_TEXT_SECTIONS
    )
    assert representation[
        "raw_header_other_or_copyright_text_model_visible"
    ] is False
    assert representation[
        "exact_paths_or_path_identity_hash_model_visible"
    ] is False
    assert representation["administrative_events_model_visible"] is False
    assert daily["administrative_events_retained_in_model_cards"] is False
    assert daily[
        "administrative_events_retained_in_source_event_artifact"
    ] is True
    assert event["full_normalized_delta_audit"]["model_visible"] is False
    assert event["model_card_integration"][
        "raw_header_other_or_copyright_lines_model_visible"
    ] is False


def test_known_invalid_metadata_is_explicit_unknown_dependency_no_repair() -> (
    None
):
    payload = d5.build_preregistration()
    parser = payload["parser_contract"]
    representation = payload["representation_contract"]
    semantics = payload["event_contract"]["d5_source_semantics"]

    assert parser["duplicate_header_key_allowed"] is False
    assert parser["duplicate_or_self_dependency_allowed"] is False
    assert parser["metadata_parse_failure_action"] == (
        "CLASSIFY_D5_KNOWN_INVALID_WITHOUT_REPAIR_OR_REJECT_UNKNOWN"
    )
    assert semantics["probe_semantics"]["metadata_resolution"] == (
        "NONE_NO_FIRST_LAST_MERGE_DEDUP_RENAME_OR_SELF_EDGE_DROP"
    )
    assert semantics["probe_semantics"][
        "dependency_when_metadata_invalid"
    ] == "UNKNOWN_WITH_NULL_COUNT_NO_REPAIR"
    assert "UNKNOWN_INVALID_METADATA" in representation[
        "dependency_delta_states"
    ]
    assert representation["known_invalid_metadata_state_model_visible"] is (
        True
    )


def test_d4_parser_is_otherwise_unchanged() -> None:
    d4_parser = d4.build_preregistration()["parser_contract"]
    d5_parser = copy.deepcopy(d5.build_preregistration()["parser_contract"])

    assert d5_parser["metadata_parse_failure_action"] != d4_parser[
        "metadata_parse_failure_action"
    ]
    d5_parser["metadata_parse_failure_action"] = d4_parser[
        "metadata_parse_failure_action"
    ]
    assert d5_parser == d4_parser


def test_gate_four_totality_replaces_strict_parser_fraction() -> None:
    support = d5.build_preregistration()["source_support_contract"]

    assert "parser_success_fraction_required" not in support
    assert support["blob_semantics_total_fraction_required"] == "1.0"
    assert support["ethereum_historical_blob_class_counts"] == (
        d5.EVENT_SEMANTICS_CONTRACT[
            "historical_ethereum_blob_class_counts"
        ]
    )
    assert support["gate_four_semantics"] == (
        "STRICT_D4_VALID_OR_EXACT_ADMINISTRATIVE_REDIRECT_OR_"
        "KNOWN_INVALID_METADATA_STATE_OTHERWISE_REJECT"
    )
    assert support["administrative_events_retained_in_model_cards"] is False
    assert support[
        "administrative_events_retained_in_source_artifact"
    ] is True
    assert support["gates_in_order"] == list(d1.SOURCE_ONLY_GATES)
    assert support["relation_controls"] == list(d1.RELATION_CONTROLS)


def test_d4_contracts_outside_authorized_source_semantics_are_identical() -> (
    None
):
    d4_payload = d4.build_preregistration()
    d5_payload = d5.build_preregistration()
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "split_contract",
    ):
        assert d5_payload[key] == d4_payload[key]


def test_hydration_mechanics_are_d4_identical_after_namespace_rebase() -> None:
    contract = d5.build_preregistration()["source_contract"][
        "batch_hydration_contract"
    ]

    assert contract == d5.BATCH_HYDRATION_CONTRACT
    assert d5._transport_contract_rebased_to_d4(contract) == (
        d4.BATCH_HYDRATION_CONTRACT
    )
    assert d5.canonical_hash(contract) == (
        d5.BATCH_HYDRATION_CONTRACT_HASH
    )
    assert contract["one_fetch_invocation_per_replica"] is True
    assert contract["maintenance_child_processes_allowed"] == 0
    assert contract["post_hydration_read"]["environment"] == (
        "GIT_NO_LAZY_FETCH=1"
    )
    assert "D1, D2, D3, or D4 source-object reuse" in contract[
        "forbidden_transports"
    ]


def test_fresh_roots_refs_and_artifacts_forbid_d4_reuse() -> None:
    source = d5.build_preregistration()["source_contract"]

    assert source["source_root"] == "/tmp/psim-d5-source"
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d5/sealed-tip"
    )
    assert source["bare_repository_contract"]["source_traversal_ref"] == (
        "refs/psim-d5/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d5/sealed-tip",
    ]
    assert source["bare_repository_contract"][
        "shared_objects_or_cache_allowed"
    ] is False
    assert all(
        row["sealed_ref"] == "refs/psim-d5/sealed-tip"
        for row in source["repositories"]
    )
    assert source["artifact_paths"] == d5.ARTIFACT_PATHS
    assert all("d5" in path for path in d5.ARTIFACT_PATHS.values())
    assert len(set(d5.ARTIFACT_PATHS.values())) == len(d5.ARTIFACT_PATHS)


def test_preregistration_has_zero_forbidden_access() -> None:
    payload = d5.build_preregistration()
    access = payload["forbidden_access_contract"]

    assert set(access["counters"]) == set(d1.FORBIDDEN_COUNTERS)
    assert set(access["counters"].values()) == {0}
    assert access["network_calls_during_preregistration"] == 0
    assert access["git_commands_during_preregistration"] == 0
    assert access["source_incidence_opened"] is False
    assert access["proposal_blobs_opened"] is False
    assert access["btc_or_funding_outcomes_opened"] is False
    assert access["models_loaded"] == 0
    assert payload["inheritance_proof"]["preregistration_access"] == {
        "git_commands": 0,
        "network_calls": 0,
        "d4_forensic_root_opened": False,
        "official_historical_proposal_source_opened": False,
        "market_model_outcomes_opened": False,
        "official_reference_notes_model_visible": False,
    }


def test_manifest_hash_binds_exact_core() -> None:
    payload = d5.build_preregistration()
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")

    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["manifest_hash"] == d5.canonical_hash(core)
    assert payload == d5.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (d5.REPO_ROOT / RESULT_PATH).read_bytes()

    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert raw == d5.canonical_json_bytes(d5.build_preregistration())
    assert json.loads(raw) == d5.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "psim-d5.json"
    first = d5.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d5.write_preregistration(destination)

    assert first == second == destination
    assert first_bytes == second.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == d5.build_preregistration()


def test_write_rejects_conflict_and_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "conflict.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D5 preregistration differs",
    ):
        d5.write_preregistration(destination)

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D5 preregistration differs",
    ):
        d5.write_preregistration(symlink)


def test_d4_terminal_and_probe_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d5, "D4_TERMINAL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="terminal authority changed"):
        d5.build_preregistration()

    monkeypatch.undo()
    monkeypatch.setattr(d5, "SEMANTICS_PROBE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="semantics probe authority changed"):
        d5.build_preregistration()


def test_delta_semantics_and_transport_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d5,
        "AUTHORIZED_DELTA_PATHS",
        d5.AUTHORIZED_DELTA_PATHS[:-1],
    )
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d5.build_preregistration()

    monkeypatch.undo()
    mutated_semantics = copy.deepcopy(d5.EVENT_SEMANTICS_CONTRACT)
    mutated_semantics["model_card_integration"][
        "administrative_events_retained_in_model_cards"
    ] = True
    monkeypatch.setattr(
        d5,
        "EVENT_SEMANTICS_CONTRACT",
        mutated_semantics,
    )
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed|semantics delta hash changed",
    ):
        d5.build_preregistration()

    monkeypatch.undo()
    mutated_transport = copy.deepcopy(d5.BATCH_HYDRATION_CONTRACT)
    mutated_transport["one_fetch_invocation_per_replica"] = False
    monkeypatch.setattr(
        d5,
        "BATCH_HYDRATION_CONTRACT",
        mutated_transport,
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "inherited-contract delta changed|"
            "semantics delta hash changed|"
            "changed D4 hydration mechanics"
        ),
    ):
        d5.build_preregistration()


def test_unapproved_split_or_model_visibility_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d5._successor_core

    def mutate_split(payload: dict[str, object]) -> dict[str, object]:
        successor = original(payload)
        successor["split_contract"][
            "later_test_eval_minimum_cagr_strict_mdd"
        ] = "2.9"
        return successor

    monkeypatch.setattr(d5, "_successor_core", mutate_split)
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d5.build_preregistration()

    monkeypatch.undo()

    def leak_paths(payload: dict[str, object]) -> dict[str, object]:
        successor = original(payload)
        successor["representation_contract"][
            "exact_paths_or_path_identity_hash_model_visible"
        ] = True
        return successor

    monkeypatch.setattr(d5, "_successor_core", leak_paths)
    with pytest.raises(
        RuntimeError,
        match="inherited-contract delta changed|semantics delta hash changed",
    ):
        d5.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d5.REPO_ROOT / d5.SCRIPT_PATH).read_text(encoding="utf-8")
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
            "requests",
            "urllib",
            "httpx",
            "aiohttp",
            "subprocess",
            "git",
            "pandas",
            "numpy",
            "torch",
            "transformers",
            "ccxt",
        }
    )


def test_preregistration_document_binds_machine_contract() -> None:
    text = PREREGISTRATION_DOCUMENT_PATH.read_text(encoding="utf-8")
    payload = d5.build_preregistration()

    assert "PSIM-D5" in text
    assert d5.DECISION_COMMIT in text
    assert d5.D4_TERMINAL_RESULT_HASH in text
    assert d5.SEMANTICS_PROBE_RESULT_HASH in text
    assert payload["manifest_hash"] in text
    assert d5.AUTHORIZED_DELTA_HASH in text
    assert d5.EVENT_SEMANTICS_CONTRACT_HASH in text
    assert d5.BATCH_HYDRATION_CONTRACT_HASH in text
    assert RESULT_SHA256 in text
    assert d5.sha256_file(d5.SCRIPT_PATH) in text
    assert d5.sha256_file(Path(__file__).relative_to(d5.REPO_ROOT)) in text
    assert "/tmp/psim-d5-source" in text
    assert "official source execution is not authorized" in text
    assert "normalized_text_delta" in text
