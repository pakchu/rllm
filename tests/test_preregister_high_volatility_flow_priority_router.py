from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_flow_priority_router as p


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


def test_manifest_fixes_exact_two_candidate_family() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVFPR-6"
    assert value["action_order"] == ["HVAFC-6", "HVELR-6", "RIVSCR-6"]
    assert value["priority_orders"] == [
        ["HVAFC-6", "HVELR-6", "RIVSCR-6"],
        ["RIVSCR-6", "HVELR-6", "HVAFC-6"],
    ]
    assert value["eligibility_policy"] == "HVTCCR-8"
    assert value["eligibility_order"] == ["HVTCCR-8"]
    assert value["candidate_family"] == [
        p.candidate_id(priority) for priority in p.PRIORITY_ORDERS
    ]
    assert value["candidate_family_size"] == 2
    assert value["candidates"] == [
        {
            "candidate": p.candidate_id(priority),
            "priority_order": list(priority),
            "eligibility": "HVTCCR-8",
        }
        for priority in p.PRIORITY_ORDERS
    ]


def test_router_is_first_active_only_and_preserves_frozen_action() -> None:
    value = p.build()
    construction = value["construction"]
    assert construction["decision_join"].startswith("exact equality")
    assert construction["routing_rule"] == (
        "choose the first active action in the candidate priority_order"
    )
    assert construction["simultaneous_or_conflicting_actions"] == (
        "resolve only by priority_order"
    )
    assert construction["no_active_action"] == "cash; emit no action"
    assert construction["action_entry_side_hold"] == (
        "retain the chosen action side, exact D+5m entry, and 6 elapsed hour hold"
    )
    assert construction["never_average"] is True
    assert construction["never_intersect"] is True
    assert construction["never_flip"] is True
    assert value["clock"] == {
        "action_decisions": "exact 00:00, 08:00 and 16:00 UTC",
        "entry": "exact chosen-action D+5m entry",
        "hold": "6 elapsed hours",
        "sides": "exact chosen-action side",
        "action_already_supplies_high_volatility_gate": True,
        "eligibility_adds_or_changes_high_volatility_gate": False,
        "funding": "not a signal input; exact held settlements only after source and Gross9 pass",
    }


def test_only_same_decision_hvtccr_source_state_is_eligible() -> None:
    condition = p.build()["eligibility_condition"]
    assert condition == {
        "policy_id": "HVTCCR-8",
        "state_true": "source_valid is true and concentration_rank >= 0.80",
        "source_valid_required": True,
        "field": "concentration_rank",
        "operator": ">=",
        "threshold": 0.80,
        "same_decision_required": True,
        "uses_original_candidate_eligible_or_onset": False,
        "no_side_or_action": True,
    }


def test_strict_gates_stages_and_family_two_bonferroni_are_frozen() -> None:
    value = p.build()
    assert value["stages"] == {
        "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
        "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
        "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
    }
    assert value["source_support_gates"] == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.20,
        "max_month_share": 0.45,
    }
    assert value["gross9_novelty_gates"]["must_pass_before_economics"] is True
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == 0.05
    multiplicity = value["familywise_multiplicity"]
    assert multiplicity["rule"] == "Bonferroni"
    assert multiplicity["familywise_alpha"] == 0.10
    assert multiplicity["number_of_hypotheses"] == 2
    assert multiplicity["winner_raw_weekly_signflip_p_max"] == 0.05


def test_all_referenced_artifacts_are_hash_pinned() -> None:
    value = p.build()
    assert set(value["action_artifacts"]) == set(p.ACTION_ORDER)
    assert set(value["eligibility_artifacts"]) == {"HVTCCR-8"}
    for artifacts in (
        *value["action_artifacts"].values(),
        *value["eligibility_artifacts"].values(),
    ):
        for artifact in artifacts.values():
            assert set(artifact) == {"path", "sha256"}
            assert len(artifact["sha256"]) == 64
            assert p.sha256_file(artifact["path"]) == artifact["sha256"]


def test_prior_outcomes_disclosed_while_router_is_unopened_and_exploratory() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["all_constituent_outcomes_known"] is True
    assert boundary["all_prior_outcomes_known"] is True
    assert boundary["all_prior_family_outcomes_known"] is True
    assert boundary["prior_HVAFC_outcomes_known"] is True
    assert boundary["prior_HVELR_outcomes_known"] is True
    assert boundary["prior_RIVSCR_outcomes_known"] is True
    assert boundary["prior_HVTCCR_outcomes_known"] is True
    assert boundary["router_incidence_opened"] is False
    assert boundary["router_postentry_returns_or_pnl_opened"] is False
    assert value["router_incidence_opened"] is False
    assert value["router_outcomes_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False


def test_raw_rank_one_is_frozen_without_substitution() -> None:
    rows = _rows()
    rows[-1]["source_pass"] = False
    winner = p.select_train_winner(rows)
    assert winner["candidate"] == p.CANDIDATE_FAMILY[0]
    assert winner["frozen_before_test"] is True
    assert winner["substitution_authorized"] is False

    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(
        RuntimeError, match="raw rank one failed train; no substitution"
    ):
        p.select_train_winner(rows)


def test_manifest_hash_detects_drift() -> None:
    drifted = copy.deepcopy(p.build())
    drifted["construction"]["never_flip"] = False
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
