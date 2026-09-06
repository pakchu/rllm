from training import preregister_high_volatility_kurtosis_regime_momentum_relay as prereg
def test_preregistration_is_outcome_blind():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVKRMR-24";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["policy"]["kurtosis_rank_min"]==.8
