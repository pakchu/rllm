from training import preregister_high_volatility_alt_leadership_rotation_relay as p
def test_manifest_is_blind_canonical_and_not_hvalcr_repair():
 x=p.build();p.validate(x);core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core);assert x["policy_id"]=="HVALRR-8";assert x["outcomes_opened"] is False;assert x["source_incidence_opened"] is False;assert x["research_boundary"]["prior_hvalcr_event_set_reused"] is False;assert x["research_boundary"]["repair_of_prior_candidate"] is False
def test_transition_and_stopping_rule_are_frozen():
 x=p.build();assert "strict opposite signs" in x["features"]["directional_handoff"];assert x["policy"]["variation_rank_min"]==.65;assert x["diagnostic_controls"]["cannot_be_promoted"] is True;assert "no universe" in x["stopping_rule"]
