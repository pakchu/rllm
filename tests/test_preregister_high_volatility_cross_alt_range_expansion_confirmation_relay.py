from training import preregister_high_volatility_cross_alt_range_expansion_confirmation_relay as p
def test_frozen_preregistration():
 x=p.build();core=dict(x);h=core.pop("manifest_hash");assert h==p.canonical_hash(core)
 assert x["policy_id"]=="HVCARER-12" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False
 assert x["policy"]["minimum_expansion_breadth"]==4 and x["policy"]["alt_range_rank_min"]==.75
 assert x["clock"]["hold"]=="12 elapsed hours" and x["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
