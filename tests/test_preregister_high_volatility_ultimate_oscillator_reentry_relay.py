from training import preregister_high_volatility_ultimate_oscillator_reentry_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVUO-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["short_periods"]==7 and x["policy"]["medium_periods"]==14 and x["policy"]["long_periods"]==28 and x["policy"]["lower_level"]==30 and x["policy"]["upper_level"]==70 and "quote_asset_volume" not in x["source_plan"]["bars"]["columns"] and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
