import json
from training import preregister_spot_led_volatility_catchup_relay as prereg
def test_slvcr_preregistration_is_outcome_blind_hash_bound_singleton():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core);assert p["policy_id"]=="SLVCR-6" and p["outcomes_opened"] is False and p["source_incidence_opened"] is False
 b=p["research_boundary"];assert b["slvcr_candidate_incidence_opened"] is False and b["slvcr_post_entry_return_or_pnl_opened"] is False and b["candidate_count"]==1 and b["grid"] is False and b["repair_of_prior_candidate"] is False and b["promoted_prior_control"] is False
 assert p["source_plan"]["spot_hourly"]["materialize_after_preregistration"] is True and p["diagnostic_controls"]["diagnostic_controls_cannot_be_promoted"] is True
def test_written_slvcr_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
