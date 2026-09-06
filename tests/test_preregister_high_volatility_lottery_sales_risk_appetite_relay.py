from training import preregister_high_volatility_lottery_sales_risk_appetite_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVLSRA-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["draw_weekdays"] == ["Monday", "Wednesday", "Saturday"]
    assert value["policy"]["availability_utc_hour"] == 12
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["numeric_sales_or_candidate_incidence_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_is_fixed_to_official_draw_reports():
    value = p.build()
    source = value["source_plan"]["lottery_reports"]
    assert "texaslottery.com" in source["url_template"]
    assert source["first_warmup_draw"] == "2022-01-01"
    assert source["last_draw"] == "2026-07-29"
