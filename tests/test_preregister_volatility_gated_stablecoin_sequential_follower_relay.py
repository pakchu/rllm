import json
from training import preregister_volatility_gated_stablecoin_sequential_follower_relay as prereg

def test_vgsfr_preregistration_is_outcome_blind_and_hash_bound():
 payload=prereg.build();core={key:value for key,value in payload.items() if key!="manifest_hash"}
 assert payload["manifest_hash"]==prereg.canonical_hash(core)
 assert payload["outcomes_opened"] is False and payload["policy_id"]=="VGSFR-6"
 boundary=payload["research_boundary"]
 assert boundary["prior_sqfd_train_outcome_known"] is True
 assert boundary["prior_sqfd_outcome_used_to_define_vgsfr"] is False
 assert boundary["vgsfr_candidate_incidence_opened"] is False
 assert boundary["vgsfr_post_entry_return_or_pnl_opened"] is False
 assert boundary["grid"] is False and boundary["repair_of_prior_candidate"] is False

def test_written_vgsfr_preregistration_matches_builder():
 assert json.loads(prereg.DEFAULT_OUTPUT.read_text())==prereg.build()
