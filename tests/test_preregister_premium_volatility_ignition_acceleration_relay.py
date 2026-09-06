import json
from training import preregister_premium_volatility_ignition_acceleration_relay as prereg
def test_pviar_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core) and p["outcomes_opened"] is False and p["policy_id"]=="PVIAR-6"
 b=p["research_boundary"];assert b["premium_snapshot_rows_opened"] is False and b["candidate_incidence_opened"] is False and b["post_entry_outcomes_opened"] is False and b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False
 assert p["source_plan"]["premium_snapshot"]["materialize_after_preregistration"] is True and p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
def test_written_pviar_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
