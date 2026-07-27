from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    audit_psim_d8_rllm2_s4_pre2021_readiness as s4_audit,
)
from training import (
    preregister_psim_d8_rllm2_s5_direction_residual_ridge_fqi as prereg,
)


def test_build_binds_terminal_s4_rejection_and_keeps_raw_outcomes_closed() -> None:
    payload = prereg.build_preregistration()

    assert payload["candidate"] == {
        "id": prereg.STAGE_ID,
        "predecessor": s4_audit.STAGE_ID,
        "stage": (
            "reuse_2020_ledger_fit_direction_residual_ridge_and_seal_2021"
        ),
        "profitability_claim": False,
        "single_promotable_primary": prereg.PRIMARY_POLICY_ID,
        "selection_from_2020_metrics": False,
        "selection_from_known_2021_metrics": False,
        "globally_pristine_2021_claim": False,
    }
    evidence = payload["predecessor_terminal_evidence"]
    assert (
        evidence["terminal_record_commit"]
        == prereg.S4_TERMINAL_RECORD_COMMIT
    )
    assert (
        evidence["s4_readiness_rejection"]["result_hash"]
        == prereg.S4_REJECTION_RESULT_HASH
    )
    assert evidence["s4_readiness_rejection"]["terminal_action"] == (
        s4_audit.REJECT_ACTION
    )
    boundary = payload["access_boundary"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["2020_outcome_derived_artifact_bytes_hashed"] is True
    assert boundary["2020_transition_reward_rows_parsed"] == 0
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_market_rows_parsed"] == 0
    assert boundary["2021_funding_rows_parsed"] == 0
    assert boundary["2021_rewards_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0
    assert boundary["2021_policy_specific_outcomes_opened"] is False
    assert boundary["2022_or_later_outcomes_opened"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_forwards_started"] == 0
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == prereg.canonical_hash(core)


def test_direction_residual_reward_and_single_ridge_primary_are_exact() -> None:
    payload = prereg.build_preregistration()
    reward = payload["direction_residual_reward_contract"]

    assert reward["positions"] == [
        "POSITION_SHORT",
        "POSITION_FLAT",
        "POSITION_LONG",
    ]
    assert reward["actions"] == [
        "TARGET_FLAT",
        "TARGET_SHORT",
        "TARGET_LONG",
    ]
    assert reward["delta_formula"] == (
        "for each current position p, delta_p = 0.5 * mean_t("
        "R[t,p,TARGET_LONG] - R[t,p,TARGET_SHORT]) over all finite "
        "reachable 2020 state rows"
    )
    assert reward["transform"] == {
        "TARGET_FLAT": "R_star[t,p,flat] = R[t,p,flat]",
        "TARGET_SHORT": (
            "R_star[t,p,short] = R[t,p,short] + delta_p"
        ),
        "TARGET_LONG": (
            "R_star[t,p,long] = R[t,p,long] - delta_p"
        ),
    }
    assert reward["fit_rows"] == "all 366 2020 source rows"
    assert reward["clipping"] is False
    assert reward["scaling"] is False
    assert reward["threshold_search"] is False
    assert reward["hyperparameter_search"] is False
    fitted = payload["fitted_q_contract"]
    assert fitted == {
        "primary_policy_id": prereg.PRIMARY_POLICY_ID,
        "algorithm": "ridge",
        "alpha": 100.0,
        "unpenalized_intercept": True,
        "features": (
            "exact frozen S4 PCA32 semantic coordinates plus separate "
            "current-position one-hot"
        ),
        "bellman_iterations": 25,
        "discount": 0.99,
        "action_order": [
            "TARGET_FLAT",
            "TARGET_SHORT",
            "TARGET_LONG",
        ],
        "target_account_gross": [0.0, -0.5, 0.5],
        "tie_break": [
            "TARGET_FLAT",
            "current_target",
            "TARGET_SHORT",
            "TARGET_LONG",
        ],
        "extra_trees_authorized": False,
        "qlora_authorized": False,
        "model_load_or_new_gemma_forward_authorized": False,
    }


def test_control_family_and_preoutcome_schedule_gate_are_frozen() -> None:
    payload = prereg.build_preregistration()
    family = payload["control_family"]

    assert tuple(family["new_policy_family_ids"]) == prereg.POLICY_FAMILY_IDS
    assert tuple(family["new_control_policy_ids"]) == (
        prereg.CONTROL_POLICY_IDS
    )
    assert tuple(family["combined_2021_family_ids"]) == (
        prereg.COMBINED_2021_FAMILY_IDS
    )
    assert family["combined_2021_family_count"] == 33
    assert len(prereg.COMBINED_2021_FAMILY_IDS) == len(
        set(prereg.COMBINED_2021_FAMILY_IDS)
    )
    assert family["fixed_circular_reward_shift"] == 21
    assert family["fixed_within_month_shuffle_seed"] == 20_260_727
    gate = payload["pre2021_schedule_readiness_gate"]
    assert gate["must_run_before_any_2021_market_or_funding_read"] is True
    assert gate["base_primary_schedule_rows"] == 365
    assert gate["delayed_primary_schedule_rows"] == 365
    assert gate["minimum_nonflat_target_rows"] == 80
    assert gate["minimum_long_share_of_nonflat_targets"] == 0.20
    assert gate["minimum_short_share_of_nonflat_targets"] == 0.20
    assert gate["delayed_target_counts_must_equal_base"] is True
    assert gate["action_code_permutation_exact_target_identity"] is True
    assert (
        gate["minimum_target_hamming_distance_from_each_degenerate_control"]
        == 1
    )
    assert gate["failure_action"].startswith("TERMINAL_REJECT_S5")
    future = payload["future_2021_transfer_gate"]
    assert future["absolute_return_positive"] is True
    assert future["stress_return_positive"] is True
    assert future["delay_return_positive"] is True
    assert future["both_half_returns_positive"] is True
    assert future["cagr_to_strict_mdd_minimum"] == 1.0
    assert future["familywise_p_max_strictly_below"] == 0.25
    assert future["must_beat_strongest_nonsemantic_control"] is True
    assert future["success_is_live_promotion"] is False


def test_global_2021_contamination_is_explicit_and_fail_closed() -> None:
    payload = prereg.build_preregistration()
    disclosure = payload["global_outcome_contamination_disclosure"]

    assert disclosure["globally_pristine_2021_claim_allowed"] is False
    assert disclosure["prior_2021_transfer_result_exists"] is True
    assert disclosure["prior_2021_stage_market_rows"] == 105_120
    assert disclosure["prior_2021_stage_funding_rows"] == 1_095
    assert disclosure["s5_policy_specific_2021_metrics_already_exist"] is False
    assert disclosure["known_unrelated_2021_metrics_may_not_inform_s5_design"]
    assert disclosure["2021_may_be_called_globally_pristine"] is False
    assert disclosure["globally_pristine_evidence_requires_forward_or_live_data"]
    assert payload["candidate"]["globally_pristine_2021_claim"] is False


def test_artifact_paths_are_fixed_and_terminal_outputs_exist() -> None:
    payload = prereg.build_preregistration()
    artifacts = payload["artifact_contract"]

    assert artifacts == {
        "runner": prereg.RUNNER_PATH.as_posix(),
        "attempt": prereg.ATTEMPT_PATH.as_posix(),
        "result": prereg.RESULT_PATH.as_posix(),
        "residual_reward_ledger_2020": (
            prereg.RESIDUAL_LEDGER_PATH.as_posix()
        ),
        "base_schedules_2021": prereg.SCHEDULE_PATH.as_posix(),
        "delayed_primary_schedule_2021": (
            prereg.DELAYED_SCHEDULE_PATH.as_posix()
        ),
        "schedule_gate_manifest_2021": (
            prereg.SCHEDULE_MANIFEST_PATH.as_posix()
        ),
        "fixed_paths_no_output_override": True,
        "deterministic_csv_gzip_and_json": True,
        "write_once": True,
        "result_published_last": True,
    }
    assert prereg.repository_path(prereg.ATTEMPT_PATH).is_file()
    assert prereg.repository_path(prereg.RESULT_PATH).is_file()
    assert prereg.repository_path(prereg.RESIDUAL_LEDGER_PATH).is_file()
    assert prereg.repository_path(prereg.SCHEDULE_PATH).is_file()
    assert prereg.repository_path(prereg.DELAYED_SCHEDULE_PATH).is_file()
    assert prereg.repository_path(prereg.SCHEDULE_MANIFEST_PATH).is_file()


def test_write_is_deterministic_and_drift_closed(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    first = prereg.write_preregistration(output)
    raw = output.read_bytes()
    second = prereg.write_preregistration(output)

    assert first == second
    assert output.read_bytes() == raw
    assert json.loads(raw) == first
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="preregistration drift"):
        prereg.write_preregistration(output)


def test_committed_generated_artifact_matches_builder() -> None:
    payload = prereg.build_preregistration()
    target = prereg.repository_path(prereg.DEFAULT_OUTPUT)

    assert target.read_bytes() == prereg.canonical_bytes(
        payload,
        pretty=True,
    )
    assert json.loads(target.read_bytes())["manifest_hash"] == payload[
        "manifest_hash"
    ]


def test_predecessor_rejection_replay_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = s4_audit.build_audit()
    changed["eligible_primary_count"] = 1
    monkeypatch.setattr(s4_audit, "build_audit", lambda: changed)

    with pytest.raises(RuntimeError, match="no longer replays"):
        prereg.validate_s4_rejection()


def test_transition_ledger_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = tmp_path / "ledger.csv.gz"
    fake.write_bytes(b"tampered")
    monkeypatch.setattr(
        prereg.s4,
        "TRANSITION_LEDGER_PATH",
        fake,
    )

    with pytest.raises(RuntimeError, match="predecessor artifact changed"):
        prereg.build_preregistration()
