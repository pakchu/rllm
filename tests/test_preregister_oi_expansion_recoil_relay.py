from training import preregister_oi_expansion_recoil_relay as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_symmetric_oi_recoil():
    payload = prereg.build(); features = payload["features"]
    assert "oi_ret_4h_z >= 0.8954018630586817" in features["oi_expansion"]
    assert "abs(px_ret_4h_z) >= 0.7389570664259131" in features["price_shock"]
    assert "side*rsi_norm <= -0.04507656773717145" in features["rsi_confirmation"]
    assert "no rank refit, grid, or directional threshold asymmetry" in features["eligible_state"]
    assert payload["clock"]["hold"] == "8 elapsed hours"

def test_unchanged_gate_contract():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
