from training import preregister_high_volatility_active_address_network_value_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest = core.pop("manifest_hash")
    assert manifest == p.canonical_hash(core)
    assert value["policy_id"] == "HVAANV-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["active_address_average_days"] == 30
    assert value["policy"]["aanv_long_midrank_min"] == 0.80
    assert value["policy"]["aanv_short_midrank_max"] == 0.20
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_contract_and_prior_sample_disclosure_are_fixed():
    value = p.build()
    source = value["source_plan"]["coin_metrics"]
    assert source["metrics"] == list(p.METRICS)
    assert source["current_vintage_not_historical_revision_archive"] is True
    assert value["research_boundary"]["source_contract_sample_rows_opened"] == 5
    assert value["research_boundary"]["sample_rows_used_to_set_formula_side_or_threshold"] is False
    assert value["research_boundary"]["full_historical_source_opened"] is False


def test_direction_and_terminal_no_repair_are_explicit():
    value = p.build()
    assert "rank>=0.80 maps long" in value["mechanism"]["side"]
    assert "no metric, vintage, window" in value["stopping_rule"]
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
