from training import preregister_temporal_sign_persistence_imbalance as prereg


def test_manifest_and_evidence_boundary():
    payload = prereg.build(); core = {k: v for k, v in payload.items() if k != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_run_geometry_clock_and_gates_are_frozen():
    payload = prereg.build()
    assert "run_length^2" in payload["mechanism"]["side"]
    assert payload["clock"]["entry"] == "exact BTCUSDT Wednesday 00:05 UTC open"
    assert payload["clock"]["hold"] == "12 elapsed hours"
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }


def test_rv20_is_later_audit_only():
    payload = prereg.build()
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
