from training import preregister_high_volatility_ehlers_reversion_index_crossover_relay as p
def test_frozen():
 x=p.build();p.validate(x)
 assert x["policy_id"]=="HVERI-24" and x["policy"]["length"]==20 and x["policy"]["smooth_period"]==8 and x["policy"]["trigger_period"]==4
 assert x["diagnostic_controls"]["cannot_be_promoted"] and not x["research_boundary"]["candidate_incidence_opened"]
