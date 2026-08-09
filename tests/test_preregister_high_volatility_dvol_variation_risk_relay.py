from training import preregister_high_volatility_dvol_variation_risk_relay as prereg
def test_preregistration_is_outcome_blind_singleton():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVDVVR-12";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["policy"]["dvol_variation_rank_min"]==.75;assert p["diagnostic_controls"]["cannot_be_promoted"]
