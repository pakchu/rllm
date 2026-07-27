from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import (
    preregister_psim_d8_rllm2_s6_action_mean_residual_ridge_fqi as prereg,
)


def test_build_binds_exact_s5_failure_and_keeps_outcomes_closed() -> None:
    payload = prereg.build_preregistration()

    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["candidate"]["predecessor"] == prereg.s5.STAGE_ID
    assert payload["candidate"]["profitability_claim"] is False
    assert payload["candidate"]["selection_from_known_2021_metrics"] is False
    assert payload["candidate"]["globally_pristine_2021_claim"] is False
    assert payload["candidate"]["s5_activity_threshold_lowered"] is False
    assert payload["candidate"]["target_quota_or_q_margin_calibration"] is False
    evidence = payload["predecessor_terminal_evidence"]
    assert evidence["terminal_record_commit"] == (
        prereg.S5_TERMINAL_RECORD_COMMIT
    )
    assert evidence["s5_result"] == {
        "path": prereg.s5.RESULT_PATH.as_posix(),
        "sha256": prereg.S5_RESULT_SHA256,
        "result_hash": prereg.S5_RESULT_HASH,
        "decision": "reject",
        "failed_gate": "minimum_nonflat_target_rows",
        "observed_nonflat_target_rows": 74,
        "required_nonflat_target_rows": 80,
    }
    boundary = payload["access_boundary"]
    assert boundary["raw_market_or_funding_paths_read"] == []
    assert boundary["original_2020_transition_reward_rows_parsed"] == 0
    assert boundary["2021_market_or_funding_paths_read"] == []
    assert boundary["2021_rewards_created"] == 0
    assert boundary["2021_economic_metrics_computed"] == 0
    assert boundary["2021_policy_specific_outcomes_opened"] is False
    assert boundary["2022_or_later_outcomes_opened"] is False
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == prereg.canonical_hash(core)


def test_all_action_mean_residual_formula_is_single_fixed_repair() -> None:
    payload = prereg.build_preregistration()
    reward = payload["action_mean_residual_reward_contract"]

    assert reward["action_mean_formula"] == (
        "for each current position p and action a, mu[p,a] = "
        "mean_t R[t,p,a] over all finite reachable 2020 state rows"
    )
    assert reward["position_grand_mean_formula"] == (
        "bar_mu[p] = (mu[p,flat] + mu[p,short] + mu[p,long]) / 3"
    )
    assert reward["transform"] == (
        "R_star[t,p,a] = R[t,p,a] - mu[p,a] + bar_mu[p]"
    )
    assert reward["fit_rows"] == "all 366 2020 source rows"
    assert reward["clipping"] is False
    assert reward["scaling"] is False
    assert reward["threshold_search"] is False
    assert reward["hyperparameter_search"] is False
    assert reward["target_quota"] is False
    assert reward["q_margin_calibration"] is False
    assert reward["s5_delta_reuse"] is False
    original = payload["frozen_source_and_outcome_artifacts"][
        "original_transition_ledger_2020"
    ]
    assert original["sha256"] == prereg.s5.S4_TRANSITION_LEDGER_SHA256
    assert original["reuse_s5_residual_rewards"] is False


def test_model_family_and_unchanged_readiness_gate_are_exact() -> None:
    payload = prereg.build_preregistration()
    fitted = payload["fitted_q_contract"]

    assert fitted["primary_policy_id"] == prereg.PRIMARY_POLICY_ID
    assert fitted["algorithm"] == "ridge"
    assert fitted["alpha"] == 100.0
    assert fitted["bellman_iterations"] == 25
    assert fitted["discount"] == 0.99
    assert fitted["extra_trees_authorized"] is False
    assert fitted["qlora_authorized"] is False
    family = payload["control_family"]
    assert tuple(family["new_policy_family_ids"]) == prereg.POLICY_FAMILY_IDS
    assert tuple(family["combined_2021_family_ids"]) == (
        prereg.COMBINED_2021_FAMILY_IDS
    )
    assert family["combined_2021_family_count"] == 41
    assert len(prereg.COMBINED_2021_FAMILY_IDS) == len(
        set(prereg.COMBINED_2021_FAMILY_IDS)
    )
    gate = payload["pre2021_schedule_readiness_gate"]
    assert gate["minimum_nonflat_target_rows"] == 80
    assert gate["minimum_long_share_of_nonflat_targets"] == 0.20
    assert gate["minimum_short_share_of_nonflat_targets"] == 0.20
    assert gate["action_code_permutation_exact_target_identity"] is True
    assert gate["minimum_hamming_distance_from_s5_primary"] == 1
    assert tuple(gate["degenerate_control_ids"]) == (
        prereg.DEGENERATE_CONTROL_IDS
    )
    assert gate["failure_action"].startswith("TERMINAL_REJECT_S6")


def test_future_gate_and_global_contamination_remain_fail_closed() -> None:
    payload = prereg.build_preregistration()
    future = payload["future_2021_transfer_gate"]

    assert future["absolute_return_positive"] is True
    assert future["stress_return_positive"] is True
    assert future["delay_return_positive"] is True
    assert future["both_half_returns_positive"] is True
    assert future["cagr_to_strict_mdd_minimum"] == 1.0
    assert future["minimum_nonflat_intervals"] == 80
    assert future["minimum_each_direction_share"] == 0.20
    assert future["familywise_p_max_strictly_below"] == 0.25
    assert future["must_beat_strongest_nonsemantic_control"] is True
    assert future["success_is_live_promotion"] is False
    disclosure = payload["global_outcome_contamination_disclosure"]
    assert disclosure["globally_pristine_2021_claim_allowed"] is False
    assert disclosure["s6_policy_specific_2021_metrics_already_exist"] is False
    assert disclosure["known_unrelated_2021_metrics_may_not_inform_s6_design"]
    assert disclosure["known_s5_schedule_counts_are_outcome_blind"]
    assert disclosure["2021_may_be_called_globally_pristine"] is False


def test_artifact_paths_are_fixed_and_only_failed_attempt_exists() -> None:
    artifacts = prereg.build_preregistration()["artifact_contract"]

    assert artifacts["runner"] == prereg.RUNNER_PATH.as_posix()
    assert artifacts["attempt"] == prereg.ATTEMPT_PATH.as_posix()
    assert artifacts["result"] == prereg.RESULT_PATH.as_posix()
    assert artifacts["residual_reward_ledger_2020"] == (
        prereg.RESIDUAL_LEDGER_PATH.as_posix()
    )
    assert artifacts["base_schedules_2021"] == (
        prereg.SCHEDULE_PATH.as_posix()
    )
    assert artifacts["fixed_paths_no_output_override"] is True
    assert artifacts["write_once"] is True
    attempt_path = prereg.repository_path(prereg.ATTEMPT_PATH)
    assert attempt_path.is_file()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["stage_id"] == prereg.STAGE_ID
    assert attempt["execution_commit"] == (
        "31cd9ba330a7f3c53b7a5a642d365e729d1e7cca"
    )
    assert attempt["attempt_hash"] == (
        "0b686a89dc796800422b218888fd904a24ddbfa6c7ca2e350662621085e7c45d"
    )
    for path in (
        prereg.RESULT_PATH,
        prereg.RESIDUAL_LEDGER_PATH,
        prereg.SCHEDULE_PATH,
        prereg.DELAYED_SCHEDULE_PATH,
        prereg.SCHEDULE_MANIFEST_PATH,
    ):
        assert not prereg.repository_path(path).exists()


def test_write_is_deterministic_and_drift_closed(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    first = prereg.write_preregistration(output)
    raw = output.read_bytes()
    second = prereg.write_preregistration(output)

    assert first == second
    assert output.read_bytes() == raw
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
    assert json.loads(target.read_bytes())["manifest_hash"] == (
        "e35975dc79e6dd0dd694f0998cb1fee7100c8e0d8d98585c74cf025fb501abdb"
    )


def test_s5_gate_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = prereg._read_exact_json
    result = json.loads(
        prereg.repository_path(prereg.s5.RESULT_PATH).read_text(
            encoding="utf-8"
        )
    )
    result["schedule_readiness"]["nonflat_target_rows"] = 80
    core = {
        key: value
        for key, value in result.items()
        if key != "result_hash"
    }
    result["result_hash"] = prereg.canonical_hash(core)

    def fake_reader(
        path: str | Path,
        *,
        expected_sha256: str,
        self_hash_field: str,
        expected_self_hash: str,
    ) -> dict:
        if Path(path) == prereg.s5.RESULT_PATH:
            return result
        return original(
            path,
            expected_sha256=expected_sha256,
            self_hash_field=self_hash_field,
            expected_self_hash=expected_self_hash,
        )

    monkeypatch.setattr(prereg, "_read_exact_json", fake_reader)
    monkeypatch.setattr(prereg, "S5_RESULT_HASH", result["result_hash"])

    with pytest.raises(RuntimeError, match="rejection boundary changed"):
        prereg.validate_s5_rejection()
