from training import preregister_high_volatility_cross_alt_three_bar_flip_cascade_relay as s
def test_blind_canonical_registration():
 x=s.build();s.validate(x);assert x['manifest_hash']==s.canonical_hash({k:v for k,v in x.items() if k!='manifest_hash'});assert not x['outcomes_opened'] and not x['source_incidence_opened'] and not x['gross9_rows_opened']
def test_fixed_flip_cascade_policy():
 x=s.build();assert len(s.ALTS)==6;assert x['policy']['minimum_flip_alts']==4;assert x['policy']['decision_hours_utc']==[0,4,8,12,16,20];assert x['clock']['hold']=='6 elapsed hours';assert not x['research_boundary']['repair_of_prior_candidate']
