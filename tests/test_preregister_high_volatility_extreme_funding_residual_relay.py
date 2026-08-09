from training import preregister_high_volatility_extreme_funding_residual_relay as prereg
def test_outcome_blind_singleton():
 report=prereg.build();prereg.validate(report)
 assert report["policy_id"]=="HVEFR-8" and report["singleton"]
 assert not report["outcomes_opened"] and not report["source_incidence_opened"] and not report["gross9_rows_opened"]
