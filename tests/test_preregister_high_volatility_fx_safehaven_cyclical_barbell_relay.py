from training import preregister_high_volatility_fx_safehaven_cyclical_barbell_relay as s

def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_cross_sectional_policy():
 x=s.build();assert x['policy']['pair_absolute_z_min']==.5 and x['policy']['minimum_agreeing_pairs']==3;assert x['features']['canonical_risk_returns']['USDMXN'].startswith('negative');assert not x['research_boundary']['repair_of_prior_candidate']
