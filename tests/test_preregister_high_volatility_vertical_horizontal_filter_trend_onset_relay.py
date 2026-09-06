from training import preregister_high_volatility_vertical_horizontal_filter_trend_onset_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVVHF-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["vhf_periods"]==28 and x["policy"]["trend_threshold"]==.4 and x["research_boundary"]["grid"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
