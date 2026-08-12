from training import preregister_high_volatility_cross_venue_flow_disagreement_resolution as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_flow_policy():
 x=s.build();assert x['policy']['divergence_rank_min']==.995;assert x['clock']['side'].startswith('same as spot');assert x['features']['price_direction_not_used'];assert not x['research_boundary']['repair_of_prior_candidate']
