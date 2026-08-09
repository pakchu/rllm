from training import preregister_confirmation_ladder_impulse_reversal_relay as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_confirmation_ladder():
    payload = prereg.build()
    features = payload["features"]
    assert "divisible by 36" in features["anchor"]
    assert "R_1,R_2,R_3" in features["early_return"]
    assert "R_4,R_5,R_6" in features["late_return"]
    assert "abs(L)/abs(E)" in features["acceleration_ratio"]
    assert ">=0.80" in features["eligible_state"]
    assert payload["clock"]["hold"] == "6 elapsed hours"


def test_unchanged_gate_contract():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
