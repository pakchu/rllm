from training import preregister_high_volatility_larry_williams_range_breakout_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVLWRB-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["range_factor"]==.5 and x["policy"]["hold_hours"]==24 and "quote_asset_volume" not in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
