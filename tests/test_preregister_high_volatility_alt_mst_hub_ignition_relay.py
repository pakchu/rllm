from training import preregister_high_volatility_alt_mst_hub_ignition_relay as s
def test_blind_canonical_registration():
 x=s.build();assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_fixed_mst_contract():
 x=s.build();assert x["policy_id"]=="HVAMST-8";assert x["policy"]["alt_symbols"]==list(s.ALTS);assert x["policy"]["centralization_rank_min"]==.75;assert x["policy"]["decision_hours_utc"]==[5,13,21];assert x["clock"]["hold"]=="8 elapsed hours";assert not x["research_boundary"]["repair_of_prior_candidate"]
