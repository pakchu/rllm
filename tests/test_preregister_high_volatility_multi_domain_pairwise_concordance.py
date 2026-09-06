from __future__ import annotations

import copy

import pytest

from training import preregister_high_volatility_multi_domain_pairwise_concordance as p


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


def test_manifest_fixes_second_exact_pairwise_and_family_and_frozen_clock() -> None:
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVMDPAC-6"
    assert value["battery_ordinal"] == 2
    assert value["independent_fixed_battery"] is True
    assert value["component_order"] == ["HVAFC-6", "HVCBR-6", "HVELR-6", "RIVSCR-6"]
    assert value["candidate_family"] == [
        "HVAFC-6__AND__HVCBR-6",
        "HVAFC-6__AND__HVELR-6",
        "HVAFC-6__AND__RIVSCR-6",
        "HVCBR-6__AND__HVELR-6",
        "HVCBR-6__AND__RIVSCR-6",
        "HVELR-6__AND__RIVSCR-6",
    ]
    assert value["construction"]["intersection"] == "retain only rows with exactly equal entry_time and exactly equal side"
    assert value["construction"]["timestamp_tolerance"] == "none"
    assert value["construction"]["side_tolerance"] == "none"
    assert value["construction"]["repairs"] == "none"
    assert value["construction"]["higher_order_combinations"] == "none"
    assert value["construction"]["component_formula_threshold_clock_mutability"] == "immutable"
    assert value["clock"]["decision"] == "exact shared 00:00, 08:00, or 16:00 UTC component decision"
    assert value["clock"]["entry"] == "exact shared decision+5m component entry timestamp"
    assert value["clock"]["hold"] == "6 elapsed hours"
    assert value["clock"]["gross_exposure"] == 0.5


def test_manifest_pins_qualified_component_artifacts() -> None:
    value = p.build()
    contract = value["component_frozen_contract"]
    assert contract == {
        "decision_times": "exact 00:00, 08:00, and 16:00 UTC",
        "entry": "exact decision+5m BTCUSDT open",
        "hold": "6 elapsed hours",
        "source_support": "passed for every component",
        "gross9_novelty": "passed for every component",
    }
    assert all(
        set(items) == {"preregistration", "support", "gross9", "clock"}
        for items in value["component_artifacts"].values()
    )
    assert all(
        len(artifact["sha256"]) == 64
        for items in value["component_artifacts"].values()
        for artifact in items.values()
    )


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
    drifted["clock"]["hold"] = "7 elapsed hours"
    with pytest.raises(RuntimeError, match="preregistration drift"):
        p.validate(drifted)
