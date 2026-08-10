from training import preregister_china_morning_yuan_risk_relay as prereg


def test_cymrr_is_singleton_outcome_blind_and_independent():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "CYMRR-12"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False


def test_cymrr_session_clock_and_gates_are_frozen():
    result = prereg.build()
    assert "[01:30,02:00)" in result["features"]["fx_session"]
    assert "negative sign" in result["mechanism"]["side"]
    assert "rank>=0.65" in result["features"]["btc_variation_rank"]
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert result["economic_gates"]["stop_on_first_failure"] is True
