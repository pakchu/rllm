from training import preregister_high_volatility_safehaven_intraday_relative_semivariance_dislocation_relay as s

def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_semivariance_and_local_volatility_policy():
 x=s.build();assert s.FX==('USDJPY','USDCHF');assert x['policy']['imbalance_rank_min']==.65;assert x['policy']['variation_prior_sessions']==20 and x['policy']['variation_minimum_sessions']==15;assert x['clock']['side']=='negative relative_semivariance_dislocation sign';assert not x['research_boundary']['repair_of_prior_candidate']
