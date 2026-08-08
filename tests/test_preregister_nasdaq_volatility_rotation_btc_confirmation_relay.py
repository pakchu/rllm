import json
from training import preregister_nasdaq_volatility_rotation_btc_confirmation_relay as p
def test_nvxcr_preregistration_is_outcome_blind_and_hash_bound():
 d=p.build();core={k:v for k,v in d.items() if k!='manifest_hash'};assert d['manifest_hash']==p.canonical_hash(core) and d['outcomes_opened'] is False and d['policy_id']=='NVXCR-6';assert d['research_boundary']['vxn_data_rows_opened'] is False and d['research_boundary']['nvxcr_candidate_incidence_opened'] is False and d['research_boundary']['grid'] is False and d['research_boundary']['repair_of_prior_candidate'] is False
def test_written_nvxcr_preregistration_matches_builder():assert json.loads(p.DEFAULT_OUTPUT.read_text())==p.build()
