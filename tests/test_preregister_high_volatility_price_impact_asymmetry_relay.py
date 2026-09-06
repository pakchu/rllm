from training import preregister_high_volatility_price_impact_asymmetry_relay as prereg
def test_preregistration_is_outcome_blind():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVPIAR-8";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["policy"]["minimum_bars_each_sign"]==16
