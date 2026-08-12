from training import preregister_high_volatility_spot_perpetual_sign_transition_leadership_relay as subject

def test_manifest_and_blind_boundaries():
 x=subject.build();subject.validate(x);assert x["manifest_hash"]==subject.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"});assert not x["outcomes_opened"] and not x["source_incidence_opened"] and not x["gross9_rows_opened"]
def test_policy_and_gates_are_fixed():
 x=subject.build();assert x["policy"]["minimum_transition_pairs"]==360;assert x["policy"]["leadership_rank_min"]==.75;assert x["source_support_gates"]["minimum_events"]=={"train":8,"test":12,"eval":12,"final":8};assert x["research_boundary"]["candidate_count"]==1 and not x["research_boundary"]["repair_of_prior_candidate"]
