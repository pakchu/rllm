from training import preregister_high_volatility_fisher_transform_turn_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVFT-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["range_periods"]==10 and x["policy"]["clip_absolute"]==.999 and x["policy"]["value_new_weight"]==.33 and "volume" not in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
