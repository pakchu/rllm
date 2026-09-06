from training import preregister_high_volatility_who_outbreak_disclosure_pressure_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert value["policy_id"] == "HVWODP-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["policy"]["recent_window_days"] == 28
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_who_source_and_boundary_are_fixed():
    value = p.build()
    source = value["source_plan"]["who_disease_outbreak_news"]
    assert source["collection_api"] == p.API
    assert source["fields"] == list(p.FIELDS)
    assert source["page_size"] == 50
    assert source["follow_only_same_origin_odata_next_link"] is True
    assert value["research_boundary"]["who_collection_items_or_incidence_opened"] is False
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
