from training import preregister_header_backdating_disorder_correction as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_block_clock_and_side():
    payload = prereg.build()
    features = payload["features"]
    assert ">= 300" in features["anchor"]
    assert "mediantime_(i-1)" in features["consensus_valid"]
    assert "plus 7200" in features["raw_availability"]
    assert features["side"] == "negative strict sign of side_return"
    assert payload["clock"]["hold"] == "6 elapsed hours"


def test_unchanged_gate_contract():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
