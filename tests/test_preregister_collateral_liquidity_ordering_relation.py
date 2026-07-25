from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from training import preregister_collateral_liquidity_ordering_relation as p


FAKE_PRODUCER = {
    "path": p.PRODUCER_SCRIPT,
    "commit": "a" * 40,
    "sha256": "b" * 64,
}


def synthetic_manifest() -> dict:
    core = p.manifest_core(FAKE_PRODUCER)
    return {**core, "manifest_hash": p.canonical_hash(core)}


def test_runtime_authority_uses_sanitized_hash_bound_git() -> None:
    authority = p.validate_runtime_authority()
    assert authority["path"] == "/usr/bin/git"
    assert authority["sha256"] == p.GIT_EXECUTABLE_SHA256
    assert authority["path_component"] == "/usr/bin"
    assert authority["version"].startswith("git version ")
    assert authority["top_level_matches_repository_root"] is True
    assert (
        authority["ambient_git_environment"]
        == "removed_before_every_git_subprocess"
    )
    assert authority["system_and_user_git_config"] == "disabled"


def test_git_subprocess_ignores_ambient_repository_redirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "/definitely/not/the/clor/repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/definitely/not/the/clor/worktree")
    assert Path(p._git_output("rev-parse", "--show-toplevel")).resolve() == (
        p.REPOSITORY_ROOT.resolve()
    )


def test_runtime_rejects_path_without_usr_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/bin")
    with pytest.raises(RuntimeError, match="lacks exact /usr/bin"):
        p.validate_runtime_authority()
    assert p._git_output("--version").startswith("git version ")


def test_frozen_authority_binds_sources_predecessors_and_window_policy() -> None:
    authority = p.validate_frozen_authority()
    assert authority["selection"]["commit"] == p.SELECTION_COMMIT
    assert authority["boundary"]["commit"] == p.BOUNDARY_COMMIT
    assert authority["common_window_policy"]["sha256"] == (
        p.COMMON_WINDOW_SHA256
    )
    assert authority["sources"]["treasury"]["sha256"] == p.TREASURY_SHA256
    assert authority["sources"]["ofr"]["sha256"] == p.OFR_SHA256
    assert set(authority["predecessors"]) == set(p.PREDECESSOR_BINDINGS)
    assert authority["predecessors"]["TADI"]["sha256"] == (
        p.PREDECESSOR_BINDINGS["TADI"]["sha256"]
    )
    assert authority["predecessors"]["TADI"]["parser"] == (
        p.PREDECESSOR_PARSERS["TADI"]
    )


def test_source_and_predecessor_headers_are_exact_without_value_decode() -> None:
    assert p.csv_header(p.TREASURY_PATH) == p.TREASURY_PHYSICAL_HEADER
    assert p.csv_header(p.SOMA_OPERATIONS_PATH) == (
        p.SOMA_OPERATION_PHYSICAL_HEADER
    )
    assert p.csv_header(p.SOMA_DETAILS_PATH) == p.SOMA_DETAIL_PHYSICAL_HEADER
    assert p.csv_header(p.OFR_PATH) == p.OFR_PHYSICAL_HEADER
    p._validate_source_manifests()
    p._validate_predecessor_headers()


def test_scientific_contract_freezes_relational_target_policy() -> None:
    contract = p.scientific_contract()
    assert contract["decision"]["source_incidence_informed"] is True
    assert contract["decision"]["independent_or_pristine_claim"] is False
    assert contract["sources"]["treasury"]["allowlist"] == list(
        p.TREASURY_ALLOWLIST
    )
    assert contract["sources"]["ofr"]["mnemonics"] == list(p.OFR_MNEMONICS)
    sequence = contract["sequence_language"]
    assert sequence["length"] == 12
    assert sequence["first_decision_valid_line_number"] == 12
    assert sequence["decision_uses_current_line"] is True
    assert sequence["prompt_template"] == p.PROMPT_TEMPLATE
    assert sequence["prompt_template"].endswith("TARGET=")
    assert not sequence["prompt_template"].endswith("\n")
    assert sequence["action_space"] == list(p.ACTION_SPACE)
    clock = contract["clock"]
    assert clock["invalid_line_action"] == "TARGET_FLAT"
    assert clock["maximum_target_age_elapsed_hours"] == 72
    assert clock["same_timestamp_order"] == "source_group_then_old_expiry"
    assert clock["expiry_transition"]["source_line_emitted"] is False
    assert contract["support_gates"]["minimum_model_decisions"] == {
        "TRAIN": 450,
        "TEST": 180,
        "EVAL": 180,
    }
    assert contract["support_gates"]["relation_controls"] == list(
        p.RELATION_FALSIFICATION_CONTROL_IDS
    )
    assert (
        contract["support_gates"]["append_invariance_changed_hash_floor"]
        is None
    )
    assert contract["controls"]["ordered_ids"] == list(p.CONTROL_IDS)
    assert contract["controls"]["definitions"] == p.CONTROL_DEFINITIONS
    gates = contract["predecessor_and_ablation_gates"]
    assert gates["ablations"] == ["no_Treasury", "no_SOMA", "no_OFR"]
    assert gates["exact_entry_key"] == "canonical_entry_timestamp_signless"
    assert gates["ablation_mask"]["canonical_line_grammar"] == (
        p.ABLATION_LINE_GRAMMAR
    )
    assert gates["source_support_uses_primary_nonempty_UPDATED_only"] is True
    assert gates["ablations_may_affect_or_rescue_source_support"] is False


def test_synthetic_manifest_structure_is_source_and_outcome_blind() -> None:
    payload = synthetic_manifest()
    p._validate_manifest_structure(payload)
    assert payload["source_rows_parsed"] == 0
    assert payload["source_values_opened"] is False
    assert payload["joint_state_rows_built"] == 0
    assert payload["outcomes_opened"] is False
    assert all(value == 0 for value in payload["forbidden_access"].values())
    assert payload["terminal_actions"] == p.TERMINAL_ACTIONS


def test_public_manifest_validator_rejects_forged_producer() -> None:
    with pytest.raises(RuntimeError, match="producer commit is unreadable"):
        p.validate_manifest(synthetic_manifest())


def test_manifest_tamper_is_rejected_before_producer_lookup() -> None:
    payload = synthetic_manifest()
    payload["scientific_contract"] = copy.deepcopy(
        payload["scientific_contract"]
    )
    payload["scientific_contract"]["sequence_language"]["length"] = 13
    payload["scientific_contract_hash"] = p.canonical_hash(
        payload["scientific_contract"]
    )
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="invariant mismatch"):
        p.validate_manifest(payload)


def test_repository_path_rejects_absolute_and_escape() -> None:
    with pytest.raises(RuntimeError, match="must be relative"):
        p.repository_path("/tmp/not-clor")
    with pytest.raises(RuntimeError, match="escaped root"):
        p.repository_path("../not-clor")


def test_atomic_write_is_idempotent_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    assert p._write_once_bytes("results/artifact.json", b"one") == (
        hashlib.sha256(b"one").hexdigest()
    )
    assert (tmp_path / "results/artifact.json").read_bytes() == b"one"
    assert p._write_once_bytes("results/artifact.json", b"one") == (
        hashlib.sha256(b"one").hexdigest()
    )
    with pytest.raises(RuntimeError, match="artifact drift"):
        p._write_once_bytes("results/artifact.json", b"two")


def test_atomic_write_rejects_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="output parent is missing or unsafe"):
        p._write_once_bytes("results/artifact.json", b"one")
    assert not (outside / "artifact.json").exists()


def test_real_manifest_build_after_producer_is_committed() -> None:
    tracked = p._run_git(
        "ls-files",
        "--error-unmatch",
        "--",
        p.PRODUCER_SCRIPT,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("producer is intentionally not sealed until work-unit commit")
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["authority"]["producer"] == p.producer_binding()


def test_preregistration_output_path_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="path is frozen"):
        p.write_once("results/not-clor-d1.json", synthetic_manifest())
