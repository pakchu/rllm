from training import preregister_premium_polarity_cross_settlement_relay as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_premium_cross():
    payload = prereg.build()
    features = payload["features"]
    assert "exact five-minute" in features["decision_schedule"]
    assert "60 exact" in features["persistence_window"]
    assert "sign(P0)=-S" in features["polarity_cross"]
    assert "no magnitude threshold or fitted rank" in features["polarity_cross"]
    assert "D+10m" in payload["clock"]["entry"]
    assert payload["clock"]["hold"] == "6 elapsed hours"

def test_unchanged_gate_contract():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
