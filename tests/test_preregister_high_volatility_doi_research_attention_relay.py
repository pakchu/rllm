from training import preregister_high_volatility_doi_research_attention_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVDRA-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["query_title_terms"] == ["bitcoin", "cryptocurrency", "cryptoasset"]
    assert value["policy"]["work_types"] == ["journal-article", "posted-content", "proceedings-article"]
    assert value["policy"]["same_weekday_lag_days"] == 7
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_is_fixed_to_crossref_created_clock():
    value = p.build()
    source = value["source_plan"]["crossref"]
    assert source["url"] == "https://api.crossref.org/works"
    assert source["created_start"] == "2022-01-01T00:00:00Z"
    assert source["created_end_exclusive"] == "2026-07-31T00:00:00Z"
    assert source["cursor_to_exhaustion"] is True
