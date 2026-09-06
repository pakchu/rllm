from training import record_high_volatility_kalshi_sp500_range_repricing_relay_source_rejection as r


def test_terminal_source_rejection_is_hash_bound():
    value = r.build()
    core = dict(value)
    digest = core.pop("manifest_hash")
    assert digest == r.canonical_hash(core)
    assert value["policy_id"] == "HVKSRR-24"
    assert value["advance"] is False
    assert value["decision"] == "REJECT_NO_REPAIR"
    assert value["first_failure"]["http_status"] == 404
    assert value["gates"]["historical_event_replay"] is False


def test_no_historical_outcome_or_price_was_opened():
    value = r.build()
    assert value["outcomes_opened"] is False
    assert value["historical_candidate_market_prices_opened"] is False
    assert value["funding_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["source_incidence_boundary"]["historical_candidate_repricings_computed"] is False
