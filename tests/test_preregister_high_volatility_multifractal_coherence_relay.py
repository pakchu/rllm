from training import preregister_high_volatility_multifractal_coherence_relay as p
def test_prereg_is_singleton_blind_terminal():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVMFCR-8" and x["singleton"] is True and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["research_boundary"]["candidate_count"]==1 and x["stopping_rule"].startswith("terminal first failure")
def test_policy_is_frozen():
 x=p.build();assert x["policy"]["scales"]==[8,12,16,24,30,40,60,80,120] and x["policy"]["moments"]==[1,4] and x["policy"]["gap_rank_max"]==.25 and x["policy"]["variation_rank_min"]==.65 and x["features"]["decision_grid"].startswith("exact 04:00") and x["clock"]["hold"]=="8 elapsed hours" and x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8} and x["diagnostic_controls"]["cannot_be_promoted"] is True
