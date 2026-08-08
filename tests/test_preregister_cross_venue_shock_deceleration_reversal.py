from training import preregister_cross_venue_shock_deceleration_reversal as p
def test_cvsdr_is_singleton_outcome_blind_and_distinct():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False
 assert d['research_boundary']['candidate_count']==1 and d['research_boundary']['grid'] is False
 assert d['research_boundary']['repair_of_prior_candidate'] is False
 assert 'same nonzero sign' in d['clock']['deceleration']
 assert d['clock']['side']=='opposite sign of first-half return'
