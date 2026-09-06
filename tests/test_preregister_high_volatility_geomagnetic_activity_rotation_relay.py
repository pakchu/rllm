from training import preregister_high_volatility_geomagnetic_activity_rotation_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVGMR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["kp_intervals_per_day"] == 8
    assert value["policy"]["kp_change_midrank_min"] == 0.65
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["decision"] == "D+1 12:00 UTC"
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["bitcoin_return_direction_is_cross_literature_inference"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
