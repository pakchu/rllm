from training import preregister_high_volatility_gator_awakening_relay as p
def test_boundary():
 x=p.build();assert x["policy_id"]=="HVGATOR-24" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False and x["gross9_rows_opened"] is False and x["singleton"] is True
 assert (x["policy"]["jaw_period"],x["policy"]["jaw_shift"],x["policy"]["teeth_period"],x["policy"]["teeth_shift"],x["policy"]["lips_period"],x["policy"]["lips_shift"])==(13,8,8,5,5,3)
 assert x["research_boundary"]["grid"] is False and x["research_boundary"]["repair_of_prior_candidate"] is False and x["research_boundary"]["promoted_prior_control"] is False
def test_hash():
 x=p.build();core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core)
