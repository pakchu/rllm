from training import preregister_russell_volatility_spread_break_relay as prereg


def test_rvsbr_preregistration_is_singleton_and_outcome_blind():
    result = prereg.build()
    prereg.validate(result)
    assert result["policy_id"] == "RVSBR-12"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False


def test_rvsbr_frozen_clock_and_gates():
    result = prereg.build()
    assert result["features"]["shock_gate"].startswith("absolute shock_z >= 1.0")
    assert "rank >= 0.65" in result["features"]["btc_variation_rank"]
    assert result["clock"]["hold"] == "12 elapsed hours"
    assert result["clock"]["gross_exposure"] == 0.5
    assert result["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert result["economic_gates"]["stop_on_first_failure"] is True
    assert result["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
