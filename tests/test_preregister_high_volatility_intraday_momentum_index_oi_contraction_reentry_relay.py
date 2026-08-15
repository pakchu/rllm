from training import preregister_high_volatility_intraday_momentum_index_oi_contraction_reentry_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVIMIOICR-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert x["policy"]["imi_periods"]==14 and x["policy"]["lower_level"]==30 and x["policy"]["upper_level"]==70 and x["research_boundary"]["grid"] is False and x["policy"]["open_interest_lookback_hours"]==24 and x["research_boundary"]["exact_hvimi_oi_contraction_intersection_incidence_or_outcomes_known"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
