from training import preregister_high_volatility_mexican_peso_risk_relay as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_peso_policy():
 x=s.build();assert x['policy']['shock_rank_min']==.75 and x['policy']['prior_sessions']==90;assert x['clock']['side'].startswith('negative');assert not x['research_boundary']['repair_of_prior_candidate']
