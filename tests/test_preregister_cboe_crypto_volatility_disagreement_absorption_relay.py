import json
from training import preregister_cboe_crypto_volatility_disagreement_absorption_relay as prereg
def test_ccvdar_preregistration_is_outcome_blind_hash_bound_and_singleton():
 p=prereg.build();prereg.validate(p);core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core)
 assert p["outcomes_opened"] is False and p["policy_id"]=="CCVDAR-6" and p["singleton"] is True
 b=p["research_boundary"];assert b["candidate_incidence_opened"] is False and b["post_entry_outcomes_opened"] is False and b["gross9_rows_opened"] is False
 assert b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False
 assert p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
def test_written_ccvdar_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
