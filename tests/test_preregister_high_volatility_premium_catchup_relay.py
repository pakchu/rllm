from training import preregister_high_volatility_premium_catchup_relay as prereg
def test_outcome_blind_singleton():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVPCR-8" and p["singleton"];assert not p["outcomes_opened"] and not p["source_incidence_opened"] and not p["gross9_rows_opened"];assert p["diagnostic_controls"]["cannot_be_promoted"]
