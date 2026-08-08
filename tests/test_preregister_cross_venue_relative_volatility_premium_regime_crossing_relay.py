import json
from training import preregister_cross_venue_relative_volatility_premium_regime_crossing_relay as p

def test_cvrvpr_preregistration_is_outcome_blind_and_hash_bound():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'}
 assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False and d['policy_id']=='CVRVPR-12'
 assert d['research_boundary']['cvrvpr_candidate_incidence_opened'] is False and d['research_boundary']['cvrvpr_post_entry_return_or_pnl_opened'] is False
 assert d['research_boundary']['grid'] is False and d['research_boundary']['repair_of_prior_candidate'] is False

def test_written_cvrvpr_preregistration_matches_builder():
 assert json.loads(p.DEFAULT_OUTPUT.read_text())==p.build()
