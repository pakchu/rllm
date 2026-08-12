from training import preregister_high_volatility_cross_alt_quarter_hour_flow_consensus_relay as p
def test_manifest_is_canonical_blind_and_cross_alt():
 x=p.build();p.validate(x);core={k:v for k,v in x.items() if k!="manifest_hash"};assert x["manifest_hash"]==p.canonical_hash(core);assert x["policy_id"]=="HVCAQF-6";assert x["outcomes_opened"] is False;assert x["source_incidence_opened"] is False;assert "never inputs" in x["mechanism"]["side"];assert x["research_boundary"]["prior_btc_quarter_hour_event_sets_reused"] is False
def test_gates_and_no_repair_are_frozen():
 x=p.build();assert x["policy"]["minimum_consensus_breadth"]==4;assert x["policy"]["strength_rank_min"]==.75;assert x["clock"]["hold"]=="6 elapsed hours";assert x["diagnostic_controls"]["cannot_be_promoted"] is True;assert "no universe" in x["stopping_rule"]
