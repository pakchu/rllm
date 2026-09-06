from training import preregister_high_volatility_premium_open_interest_unwind_reversal as p


def test_frozen_contract():
    value = p.build(); p.validate(value)
    assert value["policy_id"] == "HVPOIUR-8"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["features"]["eligibility"] == "abs(premium displacement) rank>=0.60, DVOL close rank>=0.60, and net OI change<0"
    assert value["clock"]["side"] == "opposite completed premium displacement"
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
    assert value["economic_gates"]["mean_gross_underlying_min_bp"] == 20.0
