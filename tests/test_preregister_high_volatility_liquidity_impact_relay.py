from training import preregister_high_volatility_liquidity_impact_relay as prereg


def test_preregistration_is_outcome_blind_and_hash_bound():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_fixed_gate_contract():
    payload = prereg.build()
    assert payload["features"]["eligibility"].startswith("impact_rank>=0.80")
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }
    assert payload["novelty_gates"]["absolute_signed_exposure_pearson_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
