from training import preregister_treasury_real_term_spread_relay as prereg


def test_preregistration_is_outcome_blind_and_singleton():
    payload = prereg.payload()
    assert payload["policy_id"] == "TRTSR-24"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["singleton"] is True
    assert payload["research_boundary"]["real_yield_source_rows_opened"] is False
    assert payload["research_boundary"]["candidate_count"] == 1


def test_real_term_spread_rule_and_clock_are_frozen():
    payload = prereg.payload()
    assert payload["features"]["real_term_spread"] == "real_yield_5y minus real_yield_10y"
    assert "rank>=0.70" in payload["features"]["magnitude_rank"]
    assert "rank>=0.65" in payload["features"]["volatility_rank"]
    assert payload["clock"]["entry"] == "exact D+1 00:05 UTC BTCUSDT open"
    assert payload["clock"]["hold"] == "24 elapsed hours"


def test_manifest_hash_and_terminal_gates_are_bound():
    payload = prereg.payload()
    manifest_hash = payload.pop("manifest_hash")
    assert prereg.canonical_hash(payload) == manifest_hash
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
