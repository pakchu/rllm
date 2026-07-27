from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as s6,
)
from training import (
    preregister_psim_d8_rllm2_s6r1_action_mean_residual_ridge_fqi as prereg,
)


def test_s6_failure_is_exact_and_outcome_closed() -> None:
    evidence = prereg.validate_s6_implementation_failure()
    failure = evidence["failure"]

    assert evidence["registration"]["manifest_hash"] == (
        prereg.S6_PREREGISTRATION_MANIFEST_HASH
    )
    assert evidence["attempt"]["attempt_hash"] == prereg.S6_ATTEMPT_HASH
    assert failure["failure_hash"] == prereg.S6_FAILURE_HASH
    boundary = failure["access_boundary_after_failure"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_reward_rows_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0
    assert boundary["2021_policy_specific_outcomes_opened"] is False


def test_repair_preserves_every_scientific_contract_section() -> None:
    repaired = prereg.build_preregistration()
    original = s6.build_preregistration()
    sections = (
        "frozen_source_and_outcome_artifacts",
        "action_mean_residual_reward_contract",
        "fitted_q_contract",
        "control_family",
        "pre2021_schedule_readiness_gate",
        "future_2021_transfer_gate",
    )

    for section in sections:
        assert repaired[section] == original[section]
    unchanged = {section: original[section] for section in sections}
    assert repaired["scientific_contract_identity"] == {
        "source_stage": s6.STAGE_ID,
        "unchanged_sections": list(sections),
        "unchanged_contract_hash": prereg.canonical_hash(unchanged),
    }
    repair = repaired["repair_contract"]
    assert repair["authorized_change_count"] == 1
    assert repair["authorized_change"] == (
        "replace residual.reconstruct_reward_tensor with "
        "residual.s5_core.reconstruct_reward_tensor"
    )
    assert all(
        repair[key] is False
        for key in (
            "model_changed",
            "source_features_changed",
            "reward_formula_changed",
            "fit_rows_changed",
            "algorithm_or_hyperparameters_changed",
            "policy_or_control_family_changed",
            "readiness_gate_changed",
            "future_transfer_gate_changed",
            "failed_attempt_deleted_or_overwritten",
        )
    )
    assert repair["distinct_write_once_paths"] is True


def test_repair_uses_distinct_write_once_paths_and_no_output_override() -> None:
    payload = prereg.build_preregistration()
    artifact = payload["artifact_contract"]
    original_paths = {
        s6.ATTEMPT_PATH,
        s6.RESULT_PATH,
        s6.RESIDUAL_LEDGER_PATH,
        s6.SCHEDULE_PATH,
        s6.DELAYED_SCHEDULE_PATH,
        s6.SCHEDULE_MANIFEST_PATH,
    }
    repaired_paths = {
        Path(artifact[key])
        for key in (
            "attempt",
            "result",
            "residual_reward_ledger_2020",
            "base_schedules_2021",
            "delayed_primary_schedule_2021",
            "schedule_gate_manifest_2021",
        )
    }

    assert repaired_paths.isdisjoint(original_paths)
    assert len(repaired_paths) == 6
    assert artifact["fixed_paths_no_output_override"] is True
    assert artifact["write_once"] is True
    assert artifact["all_paths_distinct_from_failed_s6"] is True
    assert payload["execution_contract"][
        "full_execute_smoke_test_before_commit_required"
    ]


def test_preregistration_is_self_hashed_deterministic_and_write_once(
    tmp_path: Path,
) -> None:
    first = prereg.build_preregistration()
    second = prereg.build_preregistration()
    core = {
        key: value
        for key, value in first.items()
        if key != "manifest_hash"
    }

    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(core)
    output = tmp_path / "registration.json"
    written = prereg.write_preregistration(output)
    assert written == first
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert prereg.write_preregistration(output) == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="drift"):
        prereg.write_preregistration(output)


def test_exact_failure_hash_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    failure = json.loads(
        prereg.repository_path(prereg.S6_FAILURE_PATH).read_text(
            encoding="utf-8"
        )
    )
    failure["access_boundary_after_failure"][
        "2021_economic_metrics_computed"
    ] = 1
    core = {
        key: value
        for key, value in failure.items()
        if key != "failure_hash"
    }
    failure["failure_hash"] = prereg.canonical_hash(core)
    target = tmp_path / "failure.json"
    target.write_bytes(prereg.canonical_bytes(failure, pretty=True))
    monkeypatch.setattr(prereg, "S6_FAILURE_PATH", target)
    monkeypatch.setattr(
        prereg,
        "S6_FAILURE_SHA256",
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prereg,
        "S6_FAILURE_HASH",
        failure["failure_hash"],
    )

    with pytest.raises(RuntimeError, match="failure boundary changed"):
        prereg.validate_s6_implementation_failure()
