from training import preregister_deribit_expansion_partial_absorption_relay as p
def test_depar_is_singleton_outcome_blind_and_asymmetric():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False
 assert d['research_boundary']['candidate_count']==1 and d['research_boundary']['grid'] is False and d['research_boundary']['repair_of_prior_candidate'] is False
 assert 'DVOL body strictly positive' in d['clock']['volatility'] and 'q60' in d['clock']['first_half_move']
 assert d['clock']['side']=='sign of second-half return'
