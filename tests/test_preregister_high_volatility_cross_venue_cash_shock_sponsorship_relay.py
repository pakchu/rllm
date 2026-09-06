from training import preregister_high_volatility_cross_venue_cash_shock_sponsorship_relay as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_fixed_policy_and_gates():
 x=s.build();assert x["policy"]["spot_shock_quantile"]==.95 and x["policy"]["perpetual_quiet_quantile"]==.75;assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert not x["research_boundary"]["repair_of_prior_candidate"]
