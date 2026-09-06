from training import preregister_high_volatility_knn_analog_relay as prereg


def test_preregistration_is_hash_bound_and_outcome_blind():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_model_and_rv20_contract_are_fixed():
    payload = prereg.build()
    assert payload["model"]["neighbors"] == 21
    assert payload["model"]["fit_decisions"][1] == "2023-07-01T00:00:00Z"
    assert payload["rv20"]["feature"].startswith("sqrt(365*")
    assert payload["clock"]["hold"] == "12 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }


def test_strict_sequential_gates_are_unchanged():
    payload = prereg.build()
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5
    assert payload["research_boundary"]["candidate_count"] == 1
