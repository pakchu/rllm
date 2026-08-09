from training import preregister_scheduled_trend_concordance_relay as prereg


def test_manifest_is_hash_bound_and_outcome_blind():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_schedule_and_strict_gates_are_frozen():
    payload = prereg.build()
    assert payload["features"]["decision_days"] == ["Monday", "Thursday"]
    assert payload["clock"]["hold"] == "72 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert payload["economic_gates"]["stress_cagr_to_strict_mdd_min"] == 2.5


def test_rv20_is_a_stress_slice_not_entry_filter():
    payload = prereg.build()
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["post_stage_volatility_audit"]["candidate_specific_q90_residual_positive"] is True
