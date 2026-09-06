from training import preregister_high_volatility_roll_spread_shock_reversal as prereg


def test_preregistration_is_singleton_outcome_blind_and_terminal():
    payload = prereg.build(); prereg.validate(payload)
    assert payload["policy_id"] == "HVRSSR-8"
    assert payload["singleton"] is True
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["prior_event_sets_or_controls_reused"] is False


def test_roll_spread_policy_and_gates_are_frozen():
    payload = prereg.build()
    assert payload["policy"]["window_minutes"] == 480
    assert payload["policy"]["spread_rank_min"] == 0.80
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {"train": 8, "test": 12, "eval": 12, "final": 8}
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
