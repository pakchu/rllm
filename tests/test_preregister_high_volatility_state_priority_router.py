from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_state_ordered_filter as hvsof
from training import preregister_high_volatility_state_priority_router as p


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


def test_manifest_fixes_exact_four_candidate_family() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVSPR-8"
    assert value["action_order"] == ["HVTFR-8", "HVSVF-8", "HVRSSR-8"]
    assert value["priority_orders"] == [
        ["HVTFR-8", "HVSVF-8", "HVRSSR-8"],
        ["HVRSSR-8", "HVSVF-8", "HVTFR-8"],
    ]
    assert value["eligibility_order"] == ["HVTCCR-8", "HVLZC-8"]
    assert value["candidate_family"] == [
        p.candidate_id(priority, eligibility)
        for priority in p.PRIORITY_ORDERS
        for eligibility in p.ELIGIBILITY_ORDER
    ]
    assert value["candidate_family_size"] == 4
    assert len(value["candidates"]) == 4


def test_router_is_strict_priority_only_and_preserves_frozen_actions() -> None:
    construction = p.build()["construction"]
    assert construction["decision_join"].startswith("exact equality")
    assert (
        construction["routing_rule"]
        == "choose the first active action in the candidate priority_order"
    )
    assert (
        construction["simultaneous_or_conflicting_actions"]
        == "resolve only by priority_order"
    )
    assert construction["no_active_action"] == "cash"
    assert (
        construction["action_entry_side_hold"]
        == "retain the chosen action entry_time, side, and 8 elapsed hour hold exactly"
    )
    assert construction["never_average"] is True
    assert construction["never_intersect"] is True
    assert construction["never_flip"] is True


def test_eligibility_clocks_gates_and_stages_match_hvsof() -> None:
    value = p.build()
    prior = hvsof.build()
    for eligibility in p.ELIGIBILITY_ORDER:
        expected = dict(prior["eligibility_conditions"][eligibility])
        assert value["eligibility_conditions"][eligibility] == expected
    assert value["clock"]["action_decisions"] == prior["clock"]["action_decisions"]
    assert value["clock"]["hold"] == prior["clock"]["hold"]
    assert value["stages"] == prior["stages"]
    assert value["source_support_gates"] == prior["source_support_gates"]
    assert value["gross9_novelty_gates"] == prior["gross9_novelty_gates"]
    for key, expected in prior["economic_gates"].items():
        if key != "weekly_signflip_one_sided_p_max":
            assert value["economic_gates"][key] == expected
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == 0.025


def test_exact_hvsof_artifacts_are_reused_and_boundary_is_disclosed() -> None:
    value = p.build()
    assert p.ACTION_ARTIFACTS is hvsof.ACTION_ARTIFACTS
    assert p.ELIGIBILITY_ARTIFACTS is hvsof.ELIGIBILITY_ARTIFACTS
    assert value["action_artifacts"] == hvsof.ACTION_ARTIFACTS
    assert value["eligibility_artifacts"] == hvsof.ELIGIBILITY_ARTIFACTS
    boundary = value["research_boundary"]
    assert boundary["all_component_outcomes_known"] is True
    assert boundary["all_constituent_outcomes_known"] is True
    assert boundary["all_prior_outcomes_known"] is True
    assert boundary["all_prior_family_outcomes_known"] is True
    assert boundary["prior_HVSOF_outcomes_known"] is True
    assert boundary["router_incidence_opened"] is False
    assert boundary["router_postentry_returns_or_pnl_opened"] is False
    assert value["router_incidence_opened"] is False
    assert value["router_outcomes_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False


def test_bonferroni_and_train_rank_one_are_fixed_without_substitution() -> None:
    value = p.build()
    multiplicity = value["familywise_multiplicity"]
    assert multiplicity["rule"] == "Bonferroni"
    assert multiplicity["number_of_hypotheses"] == 4
    assert multiplicity["winner_raw_weekly_signflip_p_max"] == 0.025
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
    with pytest.raises(
        RuntimeError, match="raw rank one failed train; no substitution"
    ):
        p.select_train_winner(rows)

    drifted = copy.deepcopy(p.build())
    drifted["construction"]["never_flip"] = False
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
