from training import preregister_cross_alt_funding_acceleration_contradiction_relay as prereg


def test_manifest_and_evidence_boundary_are_frozen():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["btc_price_rows_opened"] is False


def test_cross_alt_majority_clock_and_gates_are_frozen():
    payload = prereg.build()
    assert payload["features"]["universe"] == ["BTCUSDT", *prereg.ALTS]
    assert payload["policy"]["alt_majority_minimum"] == 4
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0


def test_rv20_is_not_opened_or_used_at_source_stage():
    payload = prereg.build()
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["rv20_stress_slice"]["source_stage_opened"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
