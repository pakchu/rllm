from training import preregister_confirmation_ladder_coinbase_routing_diversity_relay as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["policy_id"] == "CLCBRD-6"
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_routing_diversity_ladder():
    payload = prereg.build()
    features = payload["features"]
    assert "divisible by 36" in features["anchor"]
    assert "R_4,R_5,R_6" in features["late_unanimity"]
    assert "SHA256(raw bytes" in features["script_fingerprint"]
    assert features["routing_diversity_broadening"].startswith("D_L>D_E")
    assert "threshold-free" in features["routing_diversity_broadening"]
    assert payload["clock"]["hold"] == "6 elapsed hours"


def test_new_transaction_transport_and_no_repair_contract():
    payload = prereg.build()
    source = payload["source_plan"]["coinbase_transactions"]
    assert "Blockstream Esplora" in source["primary"]
    assert "Mempool Esplora" in source["secondary"]
    assert source["old_summary_transport_reused"] is False
    assert payload["research_boundary"]["prior_coinbase_candidate_frozen"] is False
    assert payload["research_boundary"]["exact_clcbrd_source_incidence_known"] is False
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True


def test_unchanged_gates():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
