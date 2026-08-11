from training import preregister_high_volatility_kalshi_sp500_range_repricing_relay as p


def test_frozen_preregistration():
    value = p.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == p.canonical_hash(core)
    assert value["policy_id"] == "HVKSRR-24"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["policy"]["query_series_ticker"] == "KXINX"
    assert value["policy"]["anchor_hours_before_strike"] == [3, 2]
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["research_boundary"]["repair_of_prior_candidate"] is False
    assert value["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_source_is_fixed_to_public_kalshi_ladder():
    value = p.build()
    source = value["source_plan"]["kalshi"]
    assert source["api_base"] == "https://external-api.kalshi.com/trade-api/v2"
    assert source["event_start"] == "2022-01-01T00:00:00Z"
    assert source["event_end_exclusive"] == "2026-08-01T00:00:00Z"
    assert source["cursor_to_exhaustion"] is True
    assert value["features"]["quote_state"].endswith("no trade-price fallback")
