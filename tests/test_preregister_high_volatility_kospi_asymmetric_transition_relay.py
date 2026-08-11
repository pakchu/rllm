import json

from training import preregister_high_volatility_kospi_asymmetric_transition_relay as p


def test_build_is_singleton_outcome_blind_and_canonical():
    value = p.build()
    p.validate(value)
    assert value["policy_id"] == "HVKATR-24"
    assert value["outcomes_opened"] is False
    assert value["source_incidence_opened"] is False
    assert value["gross9_rows_opened"] is False
    assert value["manifest_hash"] == p.canonical_hash(
        {key: item for key, item in value.items() if key != "manifest_hash"}
    )


def test_frozen_mechanism_clock_gates_and_controls():
    value = p.build()
    assert value["policy"]["asset_symbol"] == "^KS11"
    assert value["policy"]["cash_close_local"] == "15:30"
    assert value["policy"]["shock_midrank_min"] == 0.60
    assert value["policy"]["variation_midrank_min"] == 0.65
    assert value["clock"]["gross_exposure"] == 0.5
    assert value["clock"]["hold"] == "24 elapsed hours"
    assert value["source_support_gates"]["minimum_events"] == {
        "train": 8, "test": 12, "eval": 12, "final": 8,
    }
    assert value["novelty_gates"] == {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    assert len(value["diagnostic_controls"]["names"]) == 6


def test_committed_artifact_matches_builder():
    value = json.loads(p.DEFAULT_OUTPUT.read_text())
    p.validate(value)
    assert value == p.build()
