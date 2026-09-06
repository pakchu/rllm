from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_multi_condition_pairwise_concordance as p


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


def test_manifest_fixes_exact_pairwise_and_family_and_frozen_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVMCPAC-8"
    assert value["component_order"] == ["CARSC-8", "HVTCCR-8", "HVTFR-8", "HVLZC-8"]
    assert value["candidate_family"] == [
        "CARSC-8__AND__HVTCCR-8",
        "CARSC-8__AND__HVTFR-8",
        "CARSC-8__AND__HVLZC-8",
        "HVTCCR-8__AND__HVTFR-8",
        "HVTCCR-8__AND__HVLZC-8",
        "HVTFR-8__AND__HVLZC-8",
    ]
    assert value["construction"]["intersection"] == "retain only rows with exactly equal entry_time and exactly equal side"
    assert value["construction"]["component_formula_threshold_clock_mutability"] == "immutable"
    assert value["clock"]["hold"] == "8 elapsed hours"
    assert value["clock"]["gross_exposure"] == 0.5
    assert all(set(items) == {"preregistration", "support", "gross9", "clock"} for items in value["component_artifacts"].values())


def test_manifest_discloses_boundary_selection_and_familywise_rule() -> None:
    value = p.build()
    boundary = value["research_boundary"]
    assert boundary["all_component_standalone_outcomes_known"] is True
    assert boundary["combination_incidence_opened"] is False
    assert boundary["combination_postentry_returns_or_pnl_opened"] is False
    assert value["exploratory_discovery"] is True
    assert value["fresh_confirmatory_evidence"] is False
    assert value["train_only_selection"]["future_reselection_or_repair"] is False
    assert "before any test outcome is opened" in value["train_only_selection"]["freeze_deadline"]
    assert value["familywise_multiplicity"]["rule"] == "Bonferroni"
    assert value["familywise_multiplicity"]["number_of_hypotheses"] == 6
    assert value["economic_gates"]["weekly_signflip_one_sided_p_max"] == 0.10
    assert value["familywise_multiplicity"]["winner_raw_weekly_signflip_p_max"] == pytest.approx(0.10 / 6)


def test_train_selection_uses_only_eligible_pairs_and_fixed_ties() -> None:
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


def test_raw_rank_one_failure_is_terminal_without_substitution() -> None:
    rows = _rows()
    rows[-1]["train_economic_pass"] = False
    with pytest.raises(RuntimeError, match="raw rank one failed train; no substitution"):
        p.select_train_winner(rows)


def test_manifest_hash_detects_drift() -> None:
    value = p.build()
    drifted = copy.deepcopy(value)
    drifted["clock"]["hold"] = "9 elapsed hours"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
