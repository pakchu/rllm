from training import preregister_high_volatility_cash_range_sponsorship_relay as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_range_policy():
 x=s.build();assert x['policy']['range_rank_min']==.995 and x['policy']['history_decisions']==8640;assert x['clock']['side'].startswith('common');assert not x['research_boundary']['repair_of_prior_candidate']
