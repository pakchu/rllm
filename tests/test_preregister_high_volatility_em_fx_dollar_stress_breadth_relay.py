from training import preregister_high_volatility_em_fx_dollar_stress_breadth_relay as s

def test_blind_canonical_registration():
 x=s.build();assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]

def test_fixed_em_fx_contract():
 x=s.build();assert x["policy_id"]=="HVEMFX-12";assert x["source_plan"]["fx"]["symbols"]==["USDMXN","USDKRW","USDINR","USDCNY"];assert x["policy"]["minimum_agreeing_pairs"]==3 and x["policy"]["median_absolute_pair_z_min"]==.9;assert x["policy"]["decision_hour_utc"]==22 and x["policy"]["decision_minute"]==30 and x["clock"]["hold"]=="12 elapsed hours";assert not x["research_boundary"]["repair_of_prior_candidate"]
