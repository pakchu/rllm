from training import preregister_high_volatility_premium_price_phase_loop_relay as s
def test_blind_canonical_registration():
 x=s.build();assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_fixed_phase_loop_contract():
 x=s.build();assert x["policy_id"]=="HVPPL-8";assert x["policy"]["decision_hours_minutes_utc"]==[[3,30],[11,30],[19,30]];assert x["policy"]["area_magnitude_rank_min"]==.75;assert x["clock"]["hold"]=="8 elapsed hours";assert not x["research_boundary"]["repair_of_prior_candidate"]
