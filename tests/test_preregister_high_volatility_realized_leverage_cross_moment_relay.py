from training import preregister_high_volatility_realized_leverage_cross_moment_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVRLXC-12"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["magnitude_midrank_min"] == 0.80
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_and_estimator_are_fixed():
    value = p.build()
    assert value["policy"]["source_window_hours"] == 24
    assert value["policy"]["source_bar_minutes"] == 5
    assert value["policy"]["decision_interval_hours"] == 4
    assert "r_i*r_(i+1)^2" in value["features"]["leverage_cross_moment"]
    assert value["research_boundary"]["candidate_source_incidence_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
