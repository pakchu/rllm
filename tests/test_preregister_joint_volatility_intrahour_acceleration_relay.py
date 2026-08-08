import json
from training import preregister_joint_volatility_intrahour_acceleration_relay as prereg

def test_jviar_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"}
 assert p["manifest_hash"]==prereg.canonical_hash(core)
 assert p["outcomes_opened"] is False and p["policy_id"]=="JVIAR-6"
 b=p["research_boundary"]
 assert b["jviar_candidate_incidence_opened"] is False and b["jviar_post_entry_return_or_pnl_opened"] is False
 assert b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False
 assert p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True

def test_written_jviar_preregistration_matches_builder():
 assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
