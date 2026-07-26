from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_protocol_specification_intent_maturity as d1
from training import (
    preregister_protocol_specification_intent_maturity_d3 as d3,
)
from training import (
    preregister_protocol_specification_intent_maturity_d4 as d4,
)
from training import (
    probe_protocol_specification_intent_maturity_d4_parser as parser_probe,
)


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d4_preregistration_"
    "2026-07-26.json"
)
RESULT_SHA256 = (
    "52d77eafef0e9e79f1d7a47b9c262aad148765a34ac1928b26992cfafce4d515"
)
PREREGISTRATION_DOCUMENT_PATH = Path(
    "docs/psim-d4-source-support-preregistration-2026-07-26.md"
)


def _contract_core(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    core.pop("inheritance_proof")
    return core


def test_decision_d3_terminal_and_parser_probe_are_exactly_bound() -> None:
    assert d4.DECISION_COMMIT == (
        "131009359c60bc5b28b76d22a63abf698011fbcb"
    )
    assert d4.sha256_file(d4.DECISION_PATH) == d4.DECISION_SHA256
    assert d4.sha256_file(d4.D3_PREREGISTRATION_PATH) == (
        d4.D3_PREREGISTRATION_SHA256
    )
    assert d4.sha256_file(d4.D3_TERMINAL_PATH) == d4.D3_TERMINAL_SHA256
    assert d4.sha256_file(d4.PARSER_PROBE_PATH) == (
        d4.PARSER_PROBE_SHA256
    )

    terminal = d4._read_canonical_json(d4.D3_TERMINAL_PATH)
    assert terminal["result_hash"] == d4.D3_TERMINAL_RESULT_HASH
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"] == {
        "gate_id": 4,
        "name": "historical_blob_preamble_dependency_integrity",
    }
    assert terminal["terminal_action"] == d4.D3_TERMINAL_ACTION
    assert terminal["access_ledger"]["proposal_text_rows_opened"] == 17
    assert terminal["outcomes_opened"] is False

    probe = d4._read_canonical_json(d4.PARSER_PROBE_PATH)
    assert probe == parser_probe.build_probe()
    assert probe["result_hash"] == d4.PARSER_PROBE_RESULT_HASH
    assert probe["parser_version"] == d4.PARSER_VERSION
    assert probe["synthetic_only"] is True
    assert probe["access_boundary"][
        "official_historical_proposal_source_accessed"
    ] is False


def test_candidate_and_next_step_are_source_only() -> None:
    payload = d4.build_preregistration()
    assert payload["protocol_version"] == (
        "psim_d4_source_preregistration_v1"
    )
    assert payload["candidate"] == {
        "id": "PSIM-D4",
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "historical EIP normalized-empty separator grammar"
        ),
        "selection_commit": d4.DECISION_COMMIT,
        "source_axis": "official_eip_bip_specification_revision_relation",
        "stage": "source_support_only",
    }
    assert payload["next_authorized_step"] == (
        "implement and seal synthetic-only PSIM-D4 historical EIP parser "
        "source-support evaluator"
    )


def test_d3_inheritance_delta_is_exact_value_and_path_bound() -> None:
    d3_payload = d3.build_preregistration()
    d4_payload = d4.build_preregistration()
    successor = _contract_core(d4_payload)
    delta = d4._diff_values(_contract_core(d3_payload), successor)
    inheritance = d4_payload["inheritance_proof"]
    assert tuple(sorted(delta)) == tuple(sorted(d4.AUTHORIZED_DELTA_PATHS))
    assert delta == inheritance["authorized_delta"]
    assert inheritance["authorized_delta_paths"] == list(
        d4.AUTHORIZED_DELTA_PATHS
    )
    assert d4.canonical_hash(delta) == d4.AUTHORIZED_DELTA_HASH
    assert inheritance["authorized_delta_hash"] == (
        d4.AUTHORIZED_DELTA_HASH
    )
    assert inheritance["all_other_contract_paths_byte_equal"] is True


def test_parser_delta_is_eip_only_and_hash_bound() -> None:
    d3_parser = d3.build_preregistration()["parser_contract"]
    d4_parser = copy.deepcopy(d4.build_preregistration()["parser_contract"])
    assert d4_parser["reference_parser"]["version"] == d4.PARSER_VERSION
    assert d4_parser["reference_parser"]["eip_function"] == (
        "parse_eip_preamble_d4"
    )
    assert d4_parser["reference_parser"]["bip_function"] == (
        d3_parser["reference_parser"]["bip_function"]
    )
    assert d4_parser["reference_parser"].pop(
        "synthetic_probe_binding"
    ) == d4.PARSER_PROBE_BINDING
    assert d4_parser["eip_frontmatter"].pop(
        "normalized_empty_line_contract"
    ) == d4.PARSER_DELTA_CONTRACT
    d4_parser["reference_parser"]["version"] = d3_parser[
        "reference_parser"
    ]["version"]
    d4_parser["reference_parser"]["eip_function"] = d3_parser[
        "reference_parser"
    ]["eip_function"]
    assert d4_parser == d3_parser
    assert d4.canonical_hash(d4.PARSER_DELTA_CONTRACT) == (
        d4.PARSER_DELTA_CONTRACT_HASH
    )
    assert d4.PARSER_DELTA_CONTRACT["general_yaml_parser_adopted"] is False
    assert (
        d4.PARSER_DELTA_CONTRACT["current_eipw_compatibility_claim"] is False
    )


def test_d3_contracts_outside_parser_namespace_and_actions_are_identical() -> None:
    d3_payload = d3.build_preregistration()
    d4_payload = d4.build_preregistration()
    for key in (
        "availability_contract",
        "boundary_reset_contract",
        "bucket_contract",
        "daily_relation_contract",
        "event_contract",
        "excluded_feasibility_probe",
        "forbidden_access_contract",
        "official_sources",
        "representation_contract",
        "split_contract",
    ):
        assert d4_payload[key] == d3_payload[key]

    d3_support = copy.deepcopy(d3_payload["source_support_contract"])
    d4_support = copy.deepcopy(d4_payload["source_support_contract"])
    d3_support["first_failure_action"] = d4_support[
        "first_failure_action"
    ]
    d3_support["control_sensitivity_metric"]["first_failure_action"] = (
        d4_support["control_sensitivity_metric"]["first_failure_action"]
    )
    assert d4_support == d3_support


def test_hydration_mechanics_are_d3_identical_after_action_rebase() -> None:
    contract = d4.build_preregistration()["source_contract"][
        "batch_hydration_contract"
    ]
    assert contract == d4.BATCH_HYDRATION_CONTRACT
    assert d4._transport_contract_rebased_to_d3(contract) == (
        d3.BATCH_HYDRATION_CONTRACT
    )
    assert d4.canonical_hash(contract) == (
        d4.BATCH_HYDRATION_CONTRACT_HASH
    )
    assert contract["command"] == d3.BATCH_HYDRATION_CONTRACT["command"]
    assert contract["one_fetch_invocation_per_replica"] is True
    assert contract["maintenance_child_processes_allowed"] == 0
    assert contract["post_hydration_read"]["environment"] == (
        "GIT_NO_LAZY_FETCH=1"
    )
    assert "D1, D2, or D3 source-object reuse" in contract[
        "forbidden_transports"
    ]


def test_fresh_roots_refs_and_artifacts_forbid_d3_reuse() -> None:
    source = d4.build_preregistration()["source_contract"]
    assert source["source_root"] == "/tmp/psim-d4-source"
    assert source["bare_repository_contract"]["sealed_ref"] == (
        "refs/psim-d4/sealed-tip"
    )
    assert source["bare_repository_contract"]["source_traversal_ref"] == (
        "refs/psim-d4/sealed-tip"
    )
    assert source["bare_repository_contract"]["ref_roster"] == [
        "refs/heads/master",
        "refs/psim-d4/sealed-tip",
    ]
    assert source["bare_repository_contract"][
        "shared_objects_or_cache_allowed"
    ] is False
    assert all(
        row["sealed_ref"] == "refs/psim-d4/sealed-tip"
        for row in source["repositories"]
    )
    assert source["artifact_paths"] == d4.ARTIFACT_PATHS
    assert all("d4" in path for path in d4.ARTIFACT_PATHS.values())
    assert len(set(d4.ARTIFACT_PATHS.values())) == len(d4.ARTIFACT_PATHS)


def test_gate_control_split_and_clone_rosters_remain_d3_identical() -> None:
    d3_payload = d3.build_preregistration()
    d4_payload = d4.build_preregistration()
    d3_source = d3_payload["source_contract"]
    d4_source = d4_payload["source_contract"]
    assert d4_source["clone_arguments"] == d3_source["clone_arguments"]
    assert d4_source["repository_representation"] == d3_source[
        "repository_representation"
    ]
    assert d4_source["git_binary_binding"] == d3_source[
        "git_binary_binding"
    ]
    support = d4_payload["source_support_contract"]
    assert support["gates_in_order"] == list(d1.SOURCE_ONLY_GATES)
    assert len(support["gates_in_order"]) == 13
    assert support["relation_controls"] == list(d1.RELATION_CONTROLS)
    assert d4_payload["split_contract"] == d3_payload["split_contract"]


def test_preregistration_has_zero_forbidden_access() -> None:
    payload = d4.build_preregistration()
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
        "d3_forensic_root_opened": False,
        "official_historical_proposal_source_opened": False,
        "market_model_outcomes_opened": False,
    }


def test_manifest_hash_binds_exact_core() -> None:
    payload = d4.build_preregistration()
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    assert payload["manifest_hash"] == d4.canonical_hash(core)
    assert payload == d4.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (d4.REPO_ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert raw == d4.canonical_json_bytes(d4.build_preregistration())
    assert json.loads(raw) == d4.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "psim-d4.json"
    first = d4.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d4.write_preregistration(destination)
    assert first == second == destination
    assert first_bytes == second.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == d4.build_preregistration()


def test_write_rejects_conflicting_existing_artifact(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "psim-d4.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D4 preregistration differs",
    ):
        d4.write_preregistration(destination)


def test_write_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "psim-d4.json"
    destination.symlink_to(target)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D4 preregistration differs",
    ):
        d4.write_preregistration(destination)


def test_d3_terminal_authority_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d4, "D3_TERMINAL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="terminal authority changed"):
        d4.build_preregistration()


def test_parser_probe_authority_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d4, "PARSER_PROBE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="parser probe authority changed"):
        d4.build_preregistration()


def test_authorized_delta_roster_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d4,
        "AUTHORIZED_DELTA_PATHS",
        d4.AUTHORIZED_DELTA_PATHS[:-1],
    )
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d4.build_preregistration()


def test_parser_contract_value_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = copy.deepcopy(d4.PARSER_DELTA_CONTRACT)
    mutated["general_yaml_parser_adopted"] = True
    monkeypatch.setattr(d4, "PARSER_DELTA_CONTRACT", mutated)
    with pytest.raises(
        RuntimeError,
        match="authorized parser delta hash changed",
    ):
        d4.build_preregistration()


def test_transport_contract_value_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutated = copy.deepcopy(d4.BATCH_HYDRATION_CONTRACT)
    mutated["one_fetch_invocation_per_replica"] = False
    monkeypatch.setattr(d4, "BATCH_HYDRATION_CONTRACT", mutated)
    with pytest.raises(
        RuntimeError,
        match=(
            "inherited-contract delta changed|"
            "authorized parser delta hash changed|"
            "changed D3 hydration mechanics"
        ),
    ):
        d4.build_preregistration()


def test_unapproved_split_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d4._successor_core

    def mutate(payload: dict[str, object]) -> dict[str, object]:
        successor = original(payload)
        successor["split_contract"][
            "later_test_eval_minimum_cagr_strict_mdd"
        ] = "2.9"
        return successor

    monkeypatch.setattr(d4, "_successor_core", mutate)
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d4.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d4.REPO_ROOT / d4.SCRIPT_PATH).read_text(encoding="utf-8")
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


def test_preregistration_document_binds_machine_contract() -> None:
    text = PREREGISTRATION_DOCUMENT_PATH.read_text(encoding="utf-8")
    payload = d4.build_preregistration()
    assert "PSIM-D4" in text
    assert d4.DECISION_COMMIT in text
    assert d4.D3_TERMINAL_RESULT_HASH in text
    assert d4.PARSER_PROBE_RESULT_HASH in text
    assert payload["manifest_hash"] in text
    assert d4.AUTHORIZED_DELTA_HASH in text
    assert d4.PARSER_DELTA_CONTRACT_HASH in text
    assert d4.BATCH_HYDRATION_CONTRACT_HASH in text
    assert RESULT_SHA256 in text
    assert d4.sha256_file(d4.SCRIPT_PATH) in text
    assert d4.sha256_file(Path(__file__).relative_to(d4.REPO_ROOT)) in text
    assert "/tmp/psim-d4-source" in text
    assert "official source execution is not authorized" in text
    assert "https://eips.ethereum.org/EIPS/eip-1" in text
