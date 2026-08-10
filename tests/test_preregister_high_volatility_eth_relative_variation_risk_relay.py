from training import preregister_high_volatility_eth_relative_variation_risk_relay as p
def test_blind_singleton():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVERVR-8" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["research_boundary"]["candidate_count"]==1
def test_frozen():
 x=p.build();assert x["policy"]["relative_variation_rank_min"]==.75 and x["policy"]["btc_variation_rank_min"]==.65 and x["clock"]["side"]=="strict sign(relative_return)" and x["clock"]["hold"]=="8 elapsed hours"
