from training import preregister_high_volatility_cross_alt_cojump_propagation_relay as p
def test_frozen_preregistration():
 x=p.build();core=dict(x);h=core.pop("manifest_hash");assert h==p.canonical_hash(core)
 assert x["policy_id"]=="HVCACJP-6" and x["outcomes_opened"] is False and x["source_incidence_opened"] is False
 assert x["policy"]["jump_quantile"]==.99 and x["policy"]["minimum_cojump_breadth"]==4 and x["clock"]["hold"]=="6 elapsed hours"
 assert x["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
