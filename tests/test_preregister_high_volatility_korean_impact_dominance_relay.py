from training import preregister_high_volatility_korean_impact_dominance_relay as s
def test_blind_canonical_registration():
 x=s.build();assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_fixed_impact_contract():
 x=s.build();assert x["policy_id"]=="HVKID-8";assert x["source_plan"]["upbit"]["symbol"]=="KRW-BTC";assert x["source_plan"]["binance"]["columns"][-1]=="volume";assert x["policy"]["dominance_rank_min"]==.8;assert x["clock"]["hold"]=="8 elapsed hours";assert not x["research_boundary"]["repair_of_prior_candidate"]
