from training import preregister_high_volatility_fan_token_result_rotation_relay as p


def test_frozen_preregistration():
    value = p.build(); core = dict(value); digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core); assert value["policy_id"] == "HVFTRR-12"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["tracked_team_ids"] == ["1068", "243", "83", "89", "94"]
    assert value["policy"]["post_kickoff_decision_hours"] == 3
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "12 elapsed hours"
    assert value["research_boundary"]["cross_asset_inference_disclosed"] is True
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_is_fixed_to_laliga_annual_snapshots():
    value = p.build(); source = value["source_plan"]["match_results"]
    assert "esp.1" in source["url_template"]
    assert source["years"] == [2023, 2024, 2025, 2026]
