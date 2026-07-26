from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_protocol_specification_intent_maturity as d1
from training import (
    preregister_protocol_specification_intent_maturity_d2 as d2,
)
from training import (
    preregister_protocol_specification_intent_maturity_d3 as d3,
)


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d3_preregistration_"
    "2026-07-25.json"
)
RESULT_SHA256 = (
    "332743f25d5be45ce4d022c67758051c01297f4cc18ccdf2138be75b5ef159ab"
)


def _contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_decision_d2_terminal_and_probe_are_exactly_bound() -> None:
    assert d3.DECISION_COMMIT == (
        "126f7f1354eff90f30d5a6b3d60bd6641268b03b"
    )
    assert d3.sha256_file(d3.DECISION_PATH) == d3.DECISION_SHA256
    assert d3.sha256_file(d3.D2_PREREGISTRATION_PATH) == (
        d3.D2_PREREGISTRATION_SHA256
    )
    assert d3.sha256_file(d3.D2_TERMINAL_PATH) == d3.D2_TERMINAL_SHA256
    assert d3.sha256_file(d3.TRANSPORT_PROBE_PATH) == (
        d3.TRANSPORT_PROBE_SHA256
    )

    terminal = d3._read_canonical_json(d3.D2_TERMINAL_PATH)
    assert terminal["result_hash"] == d3.D2_TERMINAL_RESULT_HASH
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert terminal["source_incidence_opened"] is True
    assert terminal["outcomes_opened"] is False
    assert terminal["profitability_result"] is False

    probe = d3._read_canonical_json(d3.TRANSPORT_PROBE_PATH)
    assert probe["result_hash"] == d3.TRANSPORT_PROBE_RESULT_HASH
    assert probe["synthetic_only"] is True
    assert set(probe["access_boundary"].values()) == {False}


def test_candidate_and_next_step_are_source_only() -> None:
    payload = d3.build_preregistration()
    assert payload["protocol_version"] == (
        "psim_d3_source_preregistration_v1"
    )
    assert payload["candidate"] == {
        "id": "PSIM-D3",
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "targeted batch-hydration bare replay"
        ),
        "selection_commit": d3.DECISION_COMMIT,
        "source_axis": "official_eip_bip_specification_revision_relation",
        "stage": "source_support_only",
    }
    assert payload["next_authorized_step"] == (
        "implement and seal synthetic-only PSIM-D3 targeted batch-hydration "
        "source-support evaluator"
    )


def test_d2_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d2_payload = d2.build_preregistration()
    d3_payload = d3.build_preregistration()
    successor = _contract_core(d3_payload)
    delta = d3._diff_values(_contract_core(d2_payload), successor)
    inheritance = d3_payload["inheritance_proof"]
    assert tuple(sorted(delta)) == tuple(sorted(d3.AUTHORIZED_DELTA_PATHS))
    assert delta == inheritance["authorized_delta"]
    assert inheritance["authorized_delta_paths"] == list(
        d3.AUTHORIZED_DELTA_PATHS
    )
    assert d3.canonical_hash(delta) == d3.AUTHORIZED_DELTA_HASH
    assert inheritance["authorized_delta_hash"] == (
        d3.AUTHORIZED_DELTA_HASH
    )
    assert inheritance["all_other_contract_paths_byte_equal"] is True


def test_parser_split_event_and_support_thresholds_are_d2_identical() -> None:
    d2_payload = d2.build_preregistration()
    d3_payload = d3.build_preregistration()
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "daily_relation_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "parser_contract",
        "representation_contract",
        "split_contract",
    ):
        assert d3_payload[key] == d2_payload[key]

    d2_support = copy.deepcopy(d2_payload["source_support_contract"])
    d3_support = copy.deepcopy(d3_payload["source_support_contract"])
    d2_support["first_failure_action"] = d3_support[
        "first_failure_action"
    ]
    d2_support["control_sensitivity_metric"]["first_failure_action"] = (
        d3_support["control_sensitivity_metric"]["first_failure_action"]
    )
    assert d3_support == d2_support


def test_source_contract_changes_only_namespaces_refs_and_transport() -> None:
    d2_source = d2.build_preregistration()["source_contract"]
    d3_source = copy.deepcopy(
        d3.build_preregistration()["source_contract"]
    )
    assert d3_source.pop("batch_hydration_contract") == (
        d3.BATCH_HYDRATION_CONTRACT
    )
    assert d3_source.pop("git_binary_binding") == d3.GIT_BINARY_BINDING
    d3_source["source_root"] = d2_source["source_root"]
    d3_source["artifact_paths"] = d2_source["artifact_paths"]
    d3_source["bare_repository_contract"][
        "sealed_ref"
    ] = d2.SEALED_REF
    d3_source["bare_repository_contract"]["ref_roster"][1] = (
        d2.SEALED_REF
    )
    d3_source["bare_repository_contract"][
        "source_traversal_ref"
    ] = d2.SEALED_REF
    for repository in d3_source["repositories"]:
        repository["sealed_ref"] = d2.SEALED_REF
    assert d3_source == d2_source


def test_batch_hydration_command_and_object_boundary_are_frozen() -> None:
    contract = d3.build_preregistration()["source_contract"][
        "batch_hydration_contract"
    ]
    assert d3.canonical_hash(contract) == (
        d3.BATCH_HYDRATION_CONTRACT_HASH
    )
    assert contract["gate_id"] == 4
    assert contract["oid_derivation_after_gate_id"] == 3
    assert contract["one_fetch_invocation_per_replica"] is True
    assert contract["replica_count"] == 4
    assert contract["command"] == [
        "/usr/bin/git",
        "-C",
        "<fresh-bare-root>",
        "-c",
        "fetch.negotiationAlgorithm=noop",
        "fetch",
        "origin",
        "--no-tags",
        "--no-write-fetch-head",
        "--recurse-submodules=no",
        "--filter=blob:none",
        "--no-auto-maintenance",
        "--stdin",
    ]
    assert contract["physical_pack_count_fixed"] is False
    assert contract["multiple_new_packfiles_allowed"] is True
    assert contract["complete_new_object_set_must_equal_requested_oids"]
    assert contract["all_new_object_types_must_be_blob"] is True
    assert contract["new_loose_objects_allowed"] is False
    assert contract["maintenance_child_processes_allowed"] == 0
    assert contract["stdout_stderr_consumption"] == (
        "subprocess_run_communicate"
    )


def test_post_hydration_read_is_strictly_local_and_snapshot_invariant() -> None:
    contract = d3.BATCH_HYDRATION_CONTRACT
    post = contract["post_hydration_read"]
    assert post == {
        "environment": "GIT_NO_LAZY_FETCH=1",
        "cat_file_transport_role": "local_decode_only",
        "fetch_child_processes_allowed": 0,
        "object_store_ref_and_fetch_head_snapshot_must_be_unchanged": True,
        "missing_object_action": d3.FAILURE_ACTION,
    }
    assert "interactive or buffered cat-file lazy hydration" in contract[
        "forbidden_transports"
    ]
    assert "retry" in contract["forbidden_transports"]
    assert "fallback lazy fetch" in contract["forbidden_transports"]
    assert "full clone" in contract["forbidden_transports"]
    assert "git fetch --refetch" in contract["forbidden_transports"]
    assert "D1 or D2 source-object reuse" in contract[
        "forbidden_transports"
    ]


def test_exact_git_binary_is_bound_without_path_lookup() -> None:
    binding = d3.build_preregistration()["source_contract"][
        "git_binary_binding"
    ]
    assert binding == {
        "path": "/usr/bin/git",
        "sha256": (
            "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
        ),
        "version": "git version 2.43.0",
        "exact_binary_required": True,
        "path_lookup_allowed": False,
        "synthetic_no_lazy_fetch_semantic_probe_required": True,
    }
    assert d3.canonical_hash(binding) == d3.GIT_BINARY_BINDING_HASH


def test_fresh_roots_and_namespaces_forbid_d2_reuse() -> None:
    source = d3.build_preregistration()["source_contract"]
    assert source["source_root"] == "/tmp/psim-d3-source"
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d3/sealed-tip"
    )
    assert source["bare_repository_contract"]["source_traversal_ref"] == (
        "refs/psim-d3/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d3/sealed-tip",
    ]
    assert source["bare_repository_contract"][
        "shared_objects_or_cache_allowed"
    ] is False
    assert all(
        row["sealed_ref"] == "refs/psim-d3/sealed-tip"
        for row in source["repositories"]
    )


def test_artifact_paths_are_d3_namespaced_and_unique() -> None:
    paths = d3.build_preregistration()["source_contract"]["artifact_paths"]
    assert paths == d3.ARTIFACT_PATHS
    assert all("d3" in path for path in paths.values())
    assert len(set(paths.values())) == len(paths)


def test_gate_control_and_clone_rosters_remain_d2_identical() -> None:
    d2_source = d2.build_preregistration()["source_contract"]
    d3_source = d3.build_preregistration()["source_contract"]
    assert d3_source["clone_arguments"] == d2_source["clone_arguments"]
    assert d3_source["repository_representation"] == d2_source[
        "repository_representation"
    ]
    assert d3_source["git_status_allowed"] is False
    d2_support = d2.build_preregistration()["source_support_contract"]
    d3_support = d3.build_preregistration()["source_support_contract"]
    assert d3_support["gates_in_order"] == list(d1.SOURCE_ONLY_GATES)
    assert d3_support["gates_in_order"] == d2_support["gates_in_order"]
    assert len(d3_support["gates_in_order"]) == 13
    assert d3_support["relation_controls"] == list(d1.RELATION_CONTROLS)
    assert d3_support["relation_controls"] == d2_support[
        "relation_controls"
    ]


def test_transport_probe_is_evidence_not_pack_count_contract() -> None:
    binding = d3.BATCH_HYDRATION_CONTRACT[
        "synthetic_probe_binding"
    ]
    assert binding["result_hash"] == d3.TRANSPORT_PROBE_RESULT_HASH
    assert binding["official_source_opened"] is False
    assert binding["market_model_outcomes_opened"] is False
    assert binding["single_fetch_observed_pack_count"] == 1
    assert binding["buffered_cat_file_control_pack_count"] == 6
    assert binding["probe_may_change_only_transport_contract"] is True
    assert d3.BATCH_HYDRATION_CONTRACT[
        "physical_pack_count_fixed"
    ] is False


def test_preregistration_has_zero_forbidden_access() -> None:
    payload = d3.build_preregistration()
    access = payload["forbidden_access_contract"]
    assert set(access["counters"]) == set(d1.FORBIDDEN_COUNTERS)
    assert set(access["counters"].values()) == {0}
    assert access["network_calls_during_preregistration"] == 0
    assert access["git_commands_during_preregistration"] == 0
    assert access["source_incidence_opened"] is False
    assert access["proposal_blobs_opened"] is False
    assert access["btc_or_funding_outcomes_opened"] is False
    assert access["models_loaded"] == 0
    prereg_access = payload["inheritance_proof"][
        "preregistration_access"
    ]
    assert prereg_access == {
        "git_commands": 0,
        "network_calls": 0,
        "official_source_opened": False,
        "market_model_outcomes_opened": False,
    }


def test_manifest_hash_binds_exact_core() -> None:
    payload = d3.build_preregistration()
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    assert payload["manifest_hash"] == d3.canonical_hash(core)
    assert payload == d3.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (d3.REPO_ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert raw == d3.canonical_json_bytes(d3.build_preregistration())
    assert json.loads(raw) == d3.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "psim-d3.json"
    first = d3.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d3.write_preregistration(destination)
    assert first == second == destination
    assert first_bytes == second.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == d3.build_preregistration()


def test_write_rejects_conflicting_existing_artifact(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "psim-d3.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D3 preregistration differs",
    ):
        d3.write_preregistration(destination)


def test_write_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "psim-d3.json"
    destination.symlink_to(target)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D3 preregistration differs",
    ):
        d3.write_preregistration(destination)


def test_d2_terminal_authority_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d3, "D2_TERMINAL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="terminal authority changed"):
        d3.build_preregistration()


def test_transport_probe_authority_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d3, "TRANSPORT_PROBE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="transport probe authority changed"):
        d3.build_preregistration()


def test_authorized_delta_roster_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d3,
        "AUTHORIZED_DELTA_PATHS",
        d3.AUTHORIZED_DELTA_PATHS[:-1],
    )
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d3.build_preregistration()


def test_transport_contract_value_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = copy.deepcopy(d3.BATCH_HYDRATION_CONTRACT)
    mutated["one_fetch_invocation_per_replica"] = False
    monkeypatch.setattr(d3, "BATCH_HYDRATION_CONTRACT", mutated)
    with pytest.raises(
        RuntimeError,
        match="authorized transport delta hash changed",
    ):
        d3.build_preregistration()


def test_unapproved_split_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d3._successor_core

    def mutate(payload: dict[str, object]) -> dict[str, object]:
        successor = original(payload)
        successor["split_contract"][
            "later_test_eval_minimum_cagr_strict_mdd"
        ] = "2.9"
        return successor

    monkeypatch.setattr(d3, "_successor_core", mutate)
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d3.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d3.REPO_ROOT / d3.SCRIPT_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
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
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(
        {
            "urlopen",
            "get",
            "post",
            "request",
            "run",
            "Popen",
            "check_output",
        }
    )
