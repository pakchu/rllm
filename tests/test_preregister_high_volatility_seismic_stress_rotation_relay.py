from training import preregister_high_volatility_seismic_stress_rotation_relay as p


def test_frozen_preregistration():
    value = p.build(); core = dict(value); digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVSSR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["minimum_magnitude"] == 5.0
    assert value["policy"]["source_lag_hours_after_day_end"] == 36
    assert value["policy"]["stress_change_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["cross_literature_inference_disclosed"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_contract_requires_causal_product_versions():
    value = p.build()
    assert value["source_plan"]["usgs"]["include_deleted"] is True
    assert value["source_plan"]["usgs"]["include_superseded_products"] is True
    assert "updateTime" in value["features"]["causal_event_version"]
