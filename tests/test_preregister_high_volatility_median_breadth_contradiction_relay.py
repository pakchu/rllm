from training import preregister_high_volatility_median_breadth_contradiction_relay as prereg
def test_preregistration_is_singleton_and_outcome_blind():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVMBC-8";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["diagnostic_controls"]["cannot_be_promoted"]
