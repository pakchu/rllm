from training import preregister_monotone_funding_price_divergence_handoff as prereg


def test_manifest_is_hash_bound_and_outcome_blind():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_funding_path_clock_and_strict_gates_are_frozen():
    payload = prereg.build()
    assert payload["policy"]["funding_path_settlements"] == 3
    assert payload["policy"]["return_rank_minimum"] == 0.60
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_rv20_is_later_stress_slice_and_alternatives_are_not_fallbacks():
    payload = prereg.build()
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["research_boundary"]["ranked_design_alternatives_are_not_fallback_candidates"] is True
