from training import preregister_high_volatility_treasury_belly_curvature_relay as prereg


def test_preregistration_contract_is_singleton_and_outcome_blind():
    payload = prereg.build()
    prereg.validate(payload)
    assert payload["policy_id"] == "HVTBCR-24"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["prior_event_sets_or_controls_reused"] is False


def test_preregistered_curvature_and_gates_are_frozen():
    payload = prereg.build()
    assert payload["features"]["curvature"] == "2*yield_5y-yield_2y-yield_10y"
    assert payload["features"]["direction"] == "-sign(curvature_change)"
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "24 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8
    }
    assert payload["novelty_gates"]["absolute_signed_exposure_pearson_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
