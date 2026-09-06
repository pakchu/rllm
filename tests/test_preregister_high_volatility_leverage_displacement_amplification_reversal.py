from training import preregister_high_volatility_leverage_displacement_amplification_reversal as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_amplification_policy():
 x=s.build();assert x['policy']['excess_rank_min']==.995;assert x['clock']['side'].startswith('opposite');assert not x['research_boundary']['repair_of_prior_candidate']
