from training import preregister_high_volatility_lunar_phase_rotation_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVLPR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["eligible_phases"] == ["New Moon", "Full Moon"]
    assert value["policy"]["phase_window_hours"] == 36
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["contrary_lunar_evidence_disclosed"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
