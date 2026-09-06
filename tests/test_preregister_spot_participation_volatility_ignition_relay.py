import json
from training import preregister_spot_participation_volatility_ignition_relay as prereg
def test_spvir_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core) and p["policy_id"]=="SPVIR-6" and p["outcomes_opened"] is False and p["source_incidence_opened"] is False
 b=p["research_boundary"];assert b["spvir_candidate_incidence_opened"] is False and b["spvir_post_entry_return_or_pnl_opened"] is False and b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False and b["promoted_prior_control"] is False
 assert p["source_plan"]["hourly_spot_perpetual"]["materialize_after_preregistration"] is True and p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
def test_written_spvir_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
