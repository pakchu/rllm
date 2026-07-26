from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_protocol_specification_intent_maturity as d1
from training import preregister_protocol_specification_intent_maturity_d2 as d2


RESULT_PATH = Path(
    "results/protocol_specification_intent_maturity_d2_preregistration_"
    "2026-07-25.json"
)
RESULT_SHA256 = (
    "3b405de2bcdc1979855e8505148f7de3fbee366cb126e78b1b23e10f84cf470a"
)


def _without_manifest(payload: dict[str, object]) -> dict[str, object]:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash")
    return core


def test_decision_and_terminal_authorities_are_exactly_bound() -> None:
    assert d2.sha256_file(d2.DECISION_PATH) == d2.DECISION_SHA256
    assert d2.DECISION_COMMIT == (
        "73de336a1d24399927d43e08c8394450b1cd1cb0"
    )
    assert d2.DECISION_SHA256 == (
        "e68c0217a6aa3927c88c1f48d9c45ed0b2be3cee4bc3c86d3cb4c6a88e1f8598"
    )
    assert d2.sha256_file(d2.D1_PREREGISTRATION_PATH) == (
        d2.D1_PREREGISTRATION_SHA256
    )
    assert d2.sha256_file(d2.D1_TERMINAL_PATH) == d2.D1_TERMINAL_SHA256
    terminal = d2._read_canonical_json(d2.D1_TERMINAL_PATH)
    assert terminal["result_hash"] == d2.D1_TERMINAL_RESULT_HASH
    assert terminal["decision"] == "reject"
    assert terminal["first_failure"]["gate_id"] == 1
    assert terminal["source_incidence_opened"] is False


def test_candidate_and_next_step_are_source_only() -> None:
    payload = d2.build_preregistration()
    assert payload["protocol_version"] == (
        "psim_d2_source_preregistration_v1"
    )
    assert payload["candidate"] == {
        "id": "PSIM-D2",
        "name": (
            "Protocol Specification Intent-Maturity relation RLLM, "
            "bare object-database replay"
        ),
        "selection_commit": d2.DECISION_COMMIT,
        "source_axis": "official_eip_bip_specification_revision_relation",
        "stage": "source_support_only",
    }
    assert payload["next_authorized_step"] == (
        "implement and seal synthetic-only PSIM-D2 bare source-support "
        "evaluator"
    )


def test_inheritance_delta_is_exact_and_exhaustive() -> None:
    d1_payload = d1.build_preregistration()
    d2_payload = d2.build_preregistration()
    successor = copy.deepcopy(d2_payload)
    successor.pop("manifest_hash")
    inheritance = successor.pop("inheritance_proof")
    delta = d2._diff_values(_without_manifest(d1_payload), successor)
    assert tuple(sorted(delta)) == tuple(sorted(d2.AUTHORIZED_DELTA_PATHS))
    assert delta == inheritance["authorized_delta"]
    assert list(d2.AUTHORIZED_DELTA_PATHS) == inheritance[
        "authorized_delta_paths"
    ]
    assert inheritance["authorized_delta_hash"] == d2.canonical_hash(delta)
    assert inheritance["all_other_paths_byte_equal"] is True


def test_d1_parser_split_and_support_thresholds_are_byte_equal() -> None:
    d1_payload = d1.build_preregistration()
    d2_payload = d2.build_preregistration()
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
        assert d2_payload[key] == d1_payload[key]

    d1_support = copy.deepcopy(d1_payload["source_support_contract"])
    d2_support = copy.deepcopy(d2_payload["source_support_contract"])
    d1_support["first_failure_action"] = d2_support["first_failure_action"]
    d1_support["control_sensitivity_metric"]["first_failure_action"] = (
        d2_support["control_sensitivity_metric"]["first_failure_action"]
    )
    assert d2_support == d1_support


def test_bare_clone_shape_and_repository_invariants_are_exact() -> None:
    source = d2.build_preregistration()["source_contract"]
    assert source["clone_arguments"] == [
        "--bare",
        "--filter=blob:none",
        "--single-branch",
        "--branch",
        "master",
        "--no-tags",
    ]
    assert source["source_root"] == "/tmp/psim-d2-source"
    assert source["repository_representation"] == (
        "BARE_OBJECT_DATABASE_NO_WORKTREE_NO_INDEX"
    )
    assert source["git_status_allowed"] is False
    assert source["shared_git_environment_scrubbed"] is True
    assert source["git_version"] == "git version 2.43.0"
    assert source["bare_repository_contract"] == {
        "absolute_git_dir_must_equal_configured_root": True,
        "checkout_allowed": False,
        "forbidden_paths": [
            ".git",
            "index",
            "worktrees",
            "commondir",
            "gitdir",
            "objects/info/alternates",
            "shallow",
        ],
        "fresh_independent_roots_required": True,
        "git_common_dir": ".",
        "git_fsck_no_dangling_required": True,
        "git_status_allowed": False,
        "is_bare_repository": True,
        "is_inside_work_tree": False,
        "ref_roster": [
            "refs/heads/master",
            "refs/psim-d2/sealed-tip",
        ],
        "root_names": {
            "bitcoin_a": "bitcoin-a.git",
            "bitcoin_b": "bitcoin-b.git",
            "ethereum_a": "ethereum-a.git",
            "ethereum_b": "ethereum-b.git",
        },
        "sealed_ref": "refs/psim-d2/sealed-tip",
        "shared_objects_or_cache_allowed": False,
        "source_traversal_ref": "refs/psim-d2/sealed-tip",
        "symbolic_head": "refs/heads/master",
    }


def test_repository_source_identities_are_inherited_except_local_refs() -> None:
    d1_rows = d1.build_preregistration()["source_contract"]["repositories"]
    d2_rows = d2.build_preregistration()["source_contract"]["repositories"]
    assert len(d1_rows) == len(d2_rows) == 2
    for before, after in zip(d1_rows, d2_rows, strict=True):
        normalized = copy.deepcopy(after)
        normalized["remote_head_symref"] = before["remote_head_symref"]
        normalized.pop("local_branch_ref")
        normalized.pop("sealed_ref")
        assert normalized == before
        assert after["remote_head_symref"] == "refs/heads/master"
        assert after["local_branch_ref"] == "refs/heads/master"
        assert after["sealed_ref"] == "refs/psim-d2/sealed-tip"


def test_artifact_paths_are_d2_namespaced() -> None:
    paths = d2.build_preregistration()["source_contract"]["artifact_paths"]
    assert paths == d2.ARTIFACT_PATHS
    assert all("d2" in path for path in paths.values())
    assert len(set(paths.values())) == len(paths)


def test_gate_and_control_rosters_remain_identical_to_d1() -> None:
    d1_support = d1.build_preregistration()["source_support_contract"]
    d2_support = d2.build_preregistration()["source_support_contract"]
    assert d2_support["gates_in_order"] == list(d1.SOURCE_ONLY_GATES)
    assert d2_support["gates_in_order"] == d1_support["gates_in_order"]
    assert len(d2_support["gates_in_order"]) == 13
    assert d2_support["relation_controls"] == list(d1.RELATION_CONTROLS)
    assert d2_support["relation_controls"] == d1_support[
        "relation_controls"
    ]
    assert d2_support["relation_control_transforms"] == d1_support[
        "relation_control_transforms"
    ]
    assert d2_support["relation_control_eligibility"] == d1_support[
        "relation_control_eligibility"
    ]
    assert d2_support["first_failure_action"] == d2.FAILURE_ACTION
    assert d2_support["control_sensitivity_metric"][
        "first_failure_action"
    ] == d2.FAILURE_ACTION


def test_local_probe_is_shape_evidence_not_contract_discretion() -> None:
    probe = d2.build_preregistration()["inheritance_proof"][
        "local_bare_probe"
    ]
    assert probe == {
        "alternates_present": False,
        "dot_git_present": False,
        "fsck_no_dangling_passed": True,
        "git_common_dir": ".",
        "git_version": "git version 2.43.0",
        "index_present": False,
        "is_bare_repository": True,
        "is_inside_work_tree": False,
        "linked_worktrees_present": False,
        "official_source_opened": False,
        "probe_may_change_inherited_contract": False,
        "symbolic_head": "refs/heads/master",
    }


def test_preregistration_has_zero_forbidden_access() -> None:
    access = d2.build_preregistration()["forbidden_access_contract"]
    assert set(access["counters"]) == set(d1.FORBIDDEN_COUNTERS)
    assert set(access["counters"].values()) == {0}
    assert access["network_calls_during_preregistration"] == 0
    assert access["git_commands_during_preregistration"] == 0
    assert access["source_incidence_opened"] is False
    assert access["proposal_blobs_opened"] is False
    assert access["btc_or_funding_outcomes_opened"] is False
    assert access["models_loaded"] == 0


def test_manifest_hash_binds_exact_core() -> None:
    payload = d2.build_preregistration()
    core = _without_manifest(payload)
    assert payload["manifest_hash"] == d2.canonical_hash(core)
    assert payload == d2.build_preregistration()


def test_committed_artifact_matches_exact_builder() -> None:
    raw = (d2.REPO_ROOT / RESULT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert raw == d2.canonical_json_bytes(d2.build_preregistration())
    assert json.loads(raw) == d2.build_preregistration()


def test_write_is_canonical_and_idempotent(tmp_path: Path) -> None:
    destination = tmp_path / "psim-d2.json"
    first = d2.write_preregistration(destination)
    first_bytes = first.read_bytes()
    second = d2.write_preregistration(destination)
    assert first == second == destination
    assert first_bytes == second.read_bytes()
    assert first_bytes.endswith(b"\n")
    assert json.loads(first_bytes) == d2.build_preregistration()


def test_write_rejects_conflicting_existing_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "psim-d2.json"
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D2 preregistration differs",
    ):
        d2.write_preregistration(destination)


def test_write_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "psim-d2.json"
    destination.symlink_to(target)
    with pytest.raises(
        RuntimeError,
        match="existing PSIM-D2 preregistration differs",
    ):
        d2.write_preregistration(destination)


def test_authority_hash_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(d2, "D1_TERMINAL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="terminal authority changed"):
        d2.build_preregistration()


def test_authorized_delta_roster_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d2,
        "AUTHORIZED_DELTA_PATHS",
        d2.AUTHORIZED_DELTA_PATHS[:-1],
    )
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d2.build_preregistration()


def test_unapproved_split_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d2._successor_core

    def mutate(payload: dict[str, object]) -> dict[str, object]:
        successor = original(payload)
        successor["split_contract"]["later_test_eval_minimum_cagr_strict_mdd"] = (
            "2.9"
        )
        return successor

    monkeypatch.setattr(d2, "_successor_core", mutate)
    with pytest.raises(RuntimeError, match="inherited-contract delta changed"):
        d2.build_preregistration()


def test_module_has_no_network_git_market_or_model_execution_imports() -> None:
    source = (d2.REPO_ROOT / d2.SCRIPT_PATH).read_text(encoding="utf-8")
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
