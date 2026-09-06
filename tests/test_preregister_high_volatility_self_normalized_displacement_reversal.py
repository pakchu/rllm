from training import preregister_high_volatility_self_normalized_displacement_reversal as p
def test_blind_singleton_terminal():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVSNDR-8" and x["singleton"] is True and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["research_boundary"]["candidate_count"]==1
def test_policy_frozen():
 x=p.build();assert x["policy"]["magnitude_rank_min"]==.8 and x["policy"]["variation_rank_min"]==.65 and x["clock"]["side"].startswith("negative") and x["clock"]["hold"]=="8 elapsed hours" and x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8}
