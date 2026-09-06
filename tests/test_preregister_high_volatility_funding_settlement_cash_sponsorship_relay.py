from training import preregister_high_volatility_funding_settlement_cash_sponsorship_relay as prereg


def test_singleton_boundary_is_outcome_blind():
    value = prereg.build()
    prereg.validate(value)
    assert value["policy_id"] == "HVFSCS-6"
    assert value["singleton"] is True
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["outcomes_opened"] is False


def test_two_stage_handoff_and_side_are_frozen():
    value = prereg.build()
    assert value["policy"]["absolute_funding_rank_min"] == 0.60
    assert value["policy"]["variation_rank_min"] == 0.65
    assert value["policy"]["cash_confirmation_hours"] == 1
    assert value["policy"]["hold_hours"] == 6
    assert "spot aggressive quote flow" in value["mechanism"]["side"]


def test_standard_gates_and_no_repair_are_frozen():
    value = prereg.build()
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert value["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert value["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
