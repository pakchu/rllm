from training import preregister_confirmation_ladder_witness_migration_sponsorship_relay as prereg


def test_manifest_and_seals():
    payload = prereg.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["policy_id"] == "CLWMSR-6"
    assert not payload["outcomes_opened"]
    assert not payload["source_incidence_opened"]
    assert not payload["gross9_rows_opened"]


def test_frozen_witness_ladder():
    payload = prereg.build()
    features = payload["features"]
    assert "divisible by 36" in features["anchor"]
    assert "R_4,R_5,R_6" in features["late_unanimity"]
    assert "Q_k=(4*S_k-W_k)/(3*S_k)" in features["witness_share"]
    assert "mean(Q_4,Q_5,Q_6) > mean(Q_1,Q_2,Q_3)" in features["witness_migration"]
    assert "threshold-free" in features["witness_migration"]
    assert payload["clock"]["hold"] == "6 elapsed hours"


def test_unchanged_gate_and_no_repair_contract():
    payload = prereg.build()
    assert payload["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert payload["novelty_gates"]["candidate_near_6h_share_max"] == 0.35
    assert payload["rv20_stress_slice"]["entry_filter"] is False
    assert payload["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True
    assert payload["research_boundary"]["prior_aggregate_witness_source_incidence_known"] is True
    assert payload["research_boundary"]["exact_clwmsr_incidence_known"] is False
