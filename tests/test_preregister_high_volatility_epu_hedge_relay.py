from training import preregister_high_volatility_epu_hedge_relay as prereg
def test_preregistration_is_outcome_blind_before_source_download():
 p=prereg.build();prereg.validate(p);assert p["policy_id"]=="HVEPUH-24";assert p["singleton"];assert not p["outcomes_opened"];assert not p["source_incidence_opened"];assert not p["gross9_rows_opened"];assert p["policy"]["publication_delay_days"]==2;assert p["source_plan"]["epu"]["download_after_preregistration_commit"]
