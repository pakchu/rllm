import json
from training import preregister_cboe_crypto_vix_transmission_relay as prereg
def test_ccvtr_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();prereg.validate(p);core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core) and p["outcomes_opened"] is False and p["policy_id"]=="CCVTR-6"
 b=p["research_boundary"];assert b["prior_vix_candidate_outcomes_known"] is True and b["selection_not_based_on_prior_control_or_direction_flip"] is True and b["candidate_incidence_opened"] is False and b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False
 assert p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
def test_written_ccvtr_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
