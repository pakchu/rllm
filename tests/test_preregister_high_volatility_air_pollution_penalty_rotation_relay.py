from training import preregister_high_volatility_air_pollution_penalty_rotation_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVAPPR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["features"]["pollutant"] == "PM2.5 in UG/M3 only"
    assert value["policy"]["air_quality_prior_minimum"] == 60
    assert value["policy"]["air_quality_change_midrank_min"] == 0.65
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_pm25_source_is_causal_hourly_archive():
    value = p.build()
    assert "HourlyData_" in value["source_plan"]["airnow"]["url_template"]
    assert value["policy"]["minimum_monitor_hours"] == 18
    assert value["clock"]["decision"].startswith("01:00 UTC")
