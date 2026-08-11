from training import preregister_high_volatility_london_fix_dollar_impulse_relay as prereg


def test_hvlfx_is_outcome_blind_singleton():
    value=prereg.build(); prereg.validate(value)
    assert value["policy_id"]=="HVLFX-12"
    assert value["singleton"] is True and value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["research_boundary"]["gross9_rows_opened"] is False
    assert value["research_boundary"]["grid"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False


def test_fix_window_clock_and_gates_are_frozen():
    value=prereg.build()
    assert "[15:55,16:05) Europe/London" in value["features"]["fixing_interval"]
    assert value["clock"]["entry"].startswith("exact 16:10 Europe/London")
    assert value["clock"]["hold"]=="12 elapsed hours"
    assert "rank>=0.65" in value["features"]["btc_variation_rank"]
    assert value["novelty_gates"]["candidate_near_6h_share_max"]==0.35
