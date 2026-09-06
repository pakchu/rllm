from training import preregister_deribit_led_shock_deceleration_reversal as p
def test_dlsdr_is_singleton_outcome_blind_and_deribit_led():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False
 assert d['research_boundary']['candidate_count']==1 and d['research_boundary']['grid'] is False
 assert d['research_boundary']['repair_of_prior_candidate'] is False
 assert 'abs(DVOL body)>abs(BVOL body)' in d['clock']['volatility']
 assert d['clock']['side']=='opposite sign of first-half return'
