from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_soma_lending_collateral_scarcity as prereg


def test_candidate_and_research_disclosure_are_honest() -> None:
    payload = prereg.build_preregistration()
    prereg.validate_preregistration(payload)
    assert payload["candidate"] == "SLCS-72-NEW-SOURCE"
    assert payload["source_family_values_previously_opened"] is True
    assert payload["source_family_market_outcomes_previously_opened"] is False
    assert payload["exact_source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    disclosure = payload["prior_research_disclosure"]
    assert disclosure["exact_source_family_new_to_repository"] is True
    assert disclosure["candidate_ratios_ranks_states_and_incidence_opened"] is False
    assert disclosure["pristine_broad_liquidity_claim"] is False


def test_policy_freezes_four_weak_components_and_prior_rank() -> None:
    policy = prereg.policy_payload()
    assert set(policy["components"]) == {
        "demand_intensity",
        "weighted_fee",
        "carry_intensity",
        "demand_breadth",
    }
    normalization = policy["normalization"]
    assert normalization["history_complete_operations"] == 252
    assert normalization["strict_prior_only"] is True
    assert normalization["current_operation_excluded"] is True
    assert normalization["expanding_fallback"] is False
    state = policy["state"]
    assert state["positive_votes_minimum"] == 3
    assert state["negative_votes_minimum"] == 3
    assert state["positive_score_minimum"] == 0.50
    assert state["negative_score_maximum"] == -0.50
    assert state["first_state_after_break_can_trigger"] is False
    assert state["persistence_inside_same_state_trades"] is False


def test_execution_support_novelty_and_rllm_are_frozen() -> None:
    policy = prereg.policy_payload()
    execution = policy["execution"]
    assert execution["entry_time"] == "ceil_to_5m(signal_time) + 5 elapsed minutes"
    assert execution["exact_grid_signal_still_waits_one_bar"] is True
    assert execution["hold_elapsed_hours"] == 72
    assert execution["notional_exposure"] == 0.5
    assert execution["global_nonoverlap"] is True
    gates = policy["source_support_gates"]
    assert gates["train_total_minimum"] == 60
    assert gates["each_train_year_minimum"] == 15
    assert gates["selection_total_minimum"] == 18
    assert gates["each_selection_half_minimum"] == 7
    novelty = policy["novelty"]
    assert novelty["comparators"] == [
        spec["name"] for spec in prereg.COMPARATOR_SPECS
    ]
    assert novelty["maximum_exact_entry_jaccard"] == 0.10
    assert novelty["maximum_slcs_one_day_containment"] == 0.35
    assert novelty["maximum_absolute_signed_exposure_correlation"] == 0.35
    assert policy["strict_economic_gates"]["cagr_to_strict_mdd_minimum"] == 3.0
    assert policy["strict_economic_gates"]["strict_mdd_pct_maximum"] == 15.0
    assert policy["rllm_boundary"][
        "authorized_before_deterministic_train_and_selection_pass"
    ] is False
    assert policy["rllm_boundary"]["later_actions"] == [
        "TRADE_FIXED_SIDE",
        "ABSTAIN",
    ]


def test_controls_are_complete_and_immutable() -> None:
    policy = prereg.policy_payload()
    assert set(policy["source_controls"]) == {
        "demand_intensity_only",
        "weighted_fee_only",
        "carry_intensity_only",
        "demand_breadth_only",
        "mean_without_consensus",
        "same_sign_without_magnitude",
        "one_operation_stale",
        "five_operation_stale",
        "year_component_permutation",
    }
    assert set(policy["economic_controls"]) >= {
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
        "year_component_permutation",
    }
    assert policy["mutable_parameters"] == []


def test_bindings_are_hash_only_and_source_rows_remain_unread() -> None:
    payload = prereg.build_preregistration()
    source = payload["source_binding"]
    assert source["manifest_operation_rows"] == 1259
    assert source["manifest_detail_rows"] == 182616
    assert source["operation_value_rows_read_during_preregistration"] == 0
    assert source["detail_value_rows_read_during_preregistration"] == 0
    assert all(
        row["value_rows_read_during_preregistration"] == 0
        for row in payload["comparator_bindings"]
    )
    assert all(
        row["values_read_during_slcs_preregistration"] == 0
        for row in payload["history_bindings"]
    )
    assert payload["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY


def test_tampering_fails_canonical_validation() -> None:
    payload = prereg.build_preregistration()
    payload["policy"]["state"]["positive_score_minimum"] = 0.40
    with pytest.raises(RuntimeError, match="frozen policy drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_false_clean_research_claim_is_rejected() -> None:
    payload = prereg.build_preregistration()
    payload["prior_research_disclosure"]["pristine_broad_liquidity_claim"] = True
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match="prior-research disclosure drift"):
        prereg.validate_preregistration(payload, verify_sources=False)


def test_write_once_verifies_identical_and_refuses_drift(tmp_path: Path) -> None:
    relative = Path(".pytest_cache") / f"{tmp_path.name}-slcs-preregistration.json"
    output = prereg.REPOSITORY_ROOT / relative
    output.unlink(missing_ok=True)
    cfg = prereg.Config(output=str(relative))
    try:
        payload, status = prereg.write_preregistration(cfg)
        assert status == "created"
        second, status = prereg.write_preregistration(cfg)
        assert status == "verified_existing"
        assert second == payload
        stored = json.loads(output.read_text())
        stored["manifest_hash"] = "0" * 64
        output.write_text(json.dumps(stored))
        with pytest.raises(RuntimeError, match="canonical hash mismatch"):
            prereg.write_preregistration(cfg)
    finally:
        output.unlink(missing_ok=True)


def test_repository_path_rejects_escape() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        prereg._repository_path("../outside.json")
