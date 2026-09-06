import json
from training import preregister_realized_over_implied_volatility_shock_continuation_relay as p
def test_rivscr_preregistration_is_outcome_blind_and_hash_bound():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False and d['policy_id']=='RIVSCR-6';assert d['research_boundary']['rivscr_candidate_incidence_opened'] is False and d['research_boundary']['grid'] is False and d['research_boundary']['repair_of_prior_candidate'] is False
def test_written_rivscr_preregistration_matches_builder():assert json.loads(p.DEFAULT_OUTPUT.read_text())==p.build()
