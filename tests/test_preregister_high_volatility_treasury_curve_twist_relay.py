from training import preregister_high_volatility_treasury_curve_twist_relay as prereg

def test_preregistration_is_bound_and_outcome_blind():
    policy=prereg.build();prereg.validate(policy)
    assert policy["policy_id"]=="HVTCTR-24"
    assert not policy["outcomes_opened"] and not policy["source_incidence_opened"] and not policy["gross9_rows_opened"]
    assert policy["diagnostic_controls"]["cannot_be_promoted"]
    assert policy["clock"]["entry"]=="decision+5m exact BTCUSDT open"
