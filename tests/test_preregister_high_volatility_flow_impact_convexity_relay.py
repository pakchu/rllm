from training import preregister_high_volatility_flow_impact_convexity_relay as p
def test_boundary():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVFIC-8";assert x["source_incidence_opened"] is False;assert x["outcomes_opened"] is False;assert x["gross9_rows_opened"] is False
def test_operator():
 x=p.build();assert x["policy"]["convexity_rank_min"]==.75;assert x["policy"]["minimum_bars_each_magnitude"]==32;assert x["features"]["decision_grid"].startswith("exact 01:00")
def test_gates():
 x=p.build();assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert x["economic_gates"]["cagr_to_strict_mdd_min"]==3.;assert x["research_boundary"]["repair_of_prior_candidate"] is False
