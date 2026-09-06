from training import preregister_high_volatility_stock_bitcoin_coskewness_relay as p


def test_frozen_preregistration():
    value=p.build(); core=dict(value); digest=core.pop("manifest_hash")
    assert digest==p.canonical_hash(core) and value["policy_id"]=="HVSBCR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["coskewness_sessions"]==63 and value["policy"]["coskewness_abs_z_min"]==0.75
    assert value["clock"]["hold"]=="24 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
