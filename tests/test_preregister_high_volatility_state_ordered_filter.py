from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_state_ordered_filter as p


def _rows() -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "source_pass": True,
            "gross9_pass": True,
            "train_cagr_to_strict_mdd": float(index + 1),
            "train_absolute_return": 0.01 * (index + 1),
            "train_economic_pass": True,
        }
        for index, candidate in enumerate(p.CANDIDATE_FAMILY)
    ]


def test_manifest_fixes_exact_ordered_action_by_filter_family() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVSOF-8"
    assert value["action_order"] == ["HVRSSR-8", "HVTFR-8", "HVSVF-8"]
    assert value["filter_order"] == ["HVTCCR-8", "HVLZC-8"]
    assert value["candidate_family"] == [
        "HVRSSR-8__FILTERED_BY__HVTCCR-8",
        "HVRSSR-8__FILTERED_BY__HVLZC-8",
        "HVTFR-8__FILTERED_BY__HVTCCR-8",
        "HVTFR-8__FILTERED_BY__HVLZC-8",
        "HVSVF-8__FILTERED_BY__HVTCCR-8",
        "HVSVF-8__FILTERED_BY__HVLZC-8",
    ]
    assert value["candidate_family_size"] == 6
    assert all(
        status
        == {
            "source_support_passed": True,
            "gross9_novelty_passed": True,
            "primary_clock_immutable": True,
        }
        for status in value["component_gate_status"]["actions"].values()
    )


def test_filters_are_frozen_states_only_and_preserve_action() -> None:
    value = p.build()
    construction = value["construction"]
    conditions = value["eligibility_conditions"]
    assert construction["decision_join"].startswith("exact equality")
    assert construction["action_entry_side_hold"] == "retain the action entry_time, side, and hold exactly"
    assert construction["filter_has_side_or_action"] is False
    assert construction["filter_may_change_variation_direction_onset_or_reservation"] is False
    assert conditions["HVTCCR-8"] == {
        "state_true": "source_valid is true and concentration_rank >= 0.80",
        "source_valid_required": True,
        "field": "concentration_rank",
        "operator": ">=",
        "threshold": 0.80,
        "uses_original_candidate_eligible_or_onset": False,
    }
    assert conditions["HVLZC-8"] == {
        "state_true": "source_valid is true and complexity_rank <= 0.25",
        "source_valid_required": True,
        "field": "complexity_rank",
        "operator": "<=",
        "threshold": 0.25,
        "uses_original_candidate_eligible_or_onset": False,
    }
    assert value["clock"]["action_decisions"] == "exact 00:00, 08:00 and 16:00 UTC"
    assert value["clock"]["entry"] == "exact action D+5m entry"
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["filter_adds_or_changes_high_volatility_gate"] is False


def test_manifest_hash_pins_requested_artifacts_and_discloses_boundary() -> None:
    value = p.build()
    assert all(
        set(artifacts) == {"preregistration", "support", "gross9", "clock"}
        for artifacts in value["action_artifacts"].values()
    )
    assert all(
        set(artifacts)
        == {"preregistration", "support", "gross9", "state_panel", "source_manifest"}
        for artifacts in value["eligibility_artifacts"].values()
    )
    boundary = value["research_boundary"]
    assert boundary["all_component_outcomes_known"] is True
    assert boundary["prior_HVMCPAC_outcomes_known"] is True
    assert boundary["prior_HVMDPAC_outcomes_known"] is True
    assert boundary["filter_incidence_opened"] is False
    assert boundary["filter_postentry_returns_or_pnl_opened"] is False
    assert value["filter_incidence_opened"] is False
    assert value["filter_outcomes_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False


def test_train_selection_and_bonferroni_are_fixed_without_substitution() -> None:
    value = p.build()
    assert value["familywise_multiplicity"]["rule"] == "Bonferroni"
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 6
    assert value["familywise_multiplicity"]["winner_raw_weekly_signflip_p_max"] == pytest.approx(0.10 / 6)
    assert value["train_only_selection"]["future_reselection_or_repair"] is False

    rows = _rows()
    rows[-1]["source_pass"] = False
    rows[-2]["train_cagr_to_strict_mdd"] = 99.0
    rows[-2]["train_absolute_return"] = 0.3
    rows[-3]["train_cagr_to_strict_mdd"] = 99.0
    rows[-3]["train_absolute_return"] = 0.3
    winner = p.select_train_winner(rows)
    assert winner["candidate"] == p.CANDIDATE_FAMILY[-3]
    assert winner["frozen_before_test"] is True
    assert winner["substitution_authorized"] is False


def test_raw_rank_one_failure_is_terminal_and_hash_detects_drift() -> None:
    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(RuntimeError, match="raw rank one failed train; no substitution"):
        p.select_train_winner(rows)

    drifted = copy.deepcopy(p.build())
    drifted["eligibility_conditions"]["HVLZC-8"]["threshold"] = 0.30
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
