import json
from training import preregister_funding_settlement_volatility_cooling_continuation_relay as prereg
def test_fsvccr_preregistration_is_outcome_blind_and_hash_bound():
 p=prereg.build();core={k:v for k,v in p.items() if k!="manifest_hash"};assert p["manifest_hash"]==prereg.canonical_hash(core);assert p["outcomes_opened"] is False and p["policy_id"]=="FSVCCR-6";b=p["research_boundary"];assert b["fsvccr_candidate_incidence_opened"] is False and b["fsvccr_post_entry_return_or_pnl_opened"] is False and b["grid"] is False and b["repair_of_prior_candidate"] is False and b["promoted_prior_control"] is False
def test_written_fsvccr_preregistration_matches_builder():assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
