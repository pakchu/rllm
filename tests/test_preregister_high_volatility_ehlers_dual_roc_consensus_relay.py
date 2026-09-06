from training import preregister_high_volatility_ehlers_dual_roc_consensus_relay as p
def test_frozen():
 x=p.build();p.validate(x);assert x["policy_id"]=="HVEDR-24" and x["policy"]["smoother_periods"]==20 and x["policy"]["fast_high_pass_periods"]==55 and x["policy"]["slow_high_pass_periods"]==156 and x["policy"]["roc_lag_periods"]==2
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
