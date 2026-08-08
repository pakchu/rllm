from training import preregister_sterling_euro_risk_beta_relay as prereg


def test_serbr_is_singleton_outcome_blind_and_independent():
    registration = prereg.build()
    prereg.validate(registration)
    assert registration["policy_id"] == "SERBR-12"
    assert registration["singleton"] is True
    assert registration["outcomes_opened"] is False
    assert registration["source_incidence_opened"] is False
    assert registration["research_boundary"]["candidate_count"] == 1
    assert registration["research_boundary"]["grid"] is False
    assert registration["research_boundary"]["repair_of_prior_candidate"] is False


def test_serbr_relative_session_clock_and_gates_are_frozen():
    registration = prereg.build()
    assert "[08:00,16:00) Europe/London" in registration["features"]["fx_session"]
    assert "GBPUSD log return minus" in registration["mechanism"]["side"]
    assert "rank>=0.65" in registration["features"]["btc_variation_rank"]
    assert registration["clock"]["hold"] == "12 elapsed hours"
    assert registration["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert registration["economic_gates"]["stop_on_first_failure"] is True
