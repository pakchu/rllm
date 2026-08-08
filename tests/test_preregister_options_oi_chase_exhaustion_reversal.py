from training import preregister_options_oi_chase_exhaustion_reversal as p
def test_oicer_is_singleton_outcome_blind_and_two_sided_by_price():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==p.chash(core)
 assert d['outcomes_opened'] is False and d['research_boundary']['candidate_count']==1 and d['research_boundary']['grid'] is False
 assert d['clock']['side']=='opposite completed-hour return';assert d['economic_gates']['stop_on_first_failure'] is True
