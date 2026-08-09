from training import preregister_volume_stratified_price_control_relay as prereg


def test_manifest_boundary():
    manifest = prereg.build()
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == prereg.canonical_hash(payload)
    assert manifest["policy_id"] == "VSPCR-8"
    assert manifest["singleton"] is True
    assert not manifest["outcomes_opened"]
    assert not manifest["source_incidence_opened"]
    assert not manifest["gross9_rows_opened"]


def test_feature_formula_clock_and_gates_are_frozen():
    manifest = prereg.build()
    features = manifest["features"]

    assert features["decision_grid"] == "every exact 4-hour UTC boundary"
    assert "72 completed coherent five-minute bars" in features["window"]
    assert features["volume_rank"].endswith("H is the top 18 and L is the bottom 18")
    assert features["high_volume_return"] == "R_H=sum r_i over H"
    assert features["low_volume_return"] == "R_L=sum r_i over L"
    assert "V=sqrt(sum r_i^2 over all 72 bars)" in features["normalizer"]
    assert features["stratified_disagreement"] == "S=abs(R_H-R_L)/V, finite"
    assert "D_H=abs(R_H)/(abs(R_H)+abs(R_L))" in features["high_volume_dominance"]
    assert features["stratified_disagreement_rank"].endswith("current excluded")
    assert features["eligible_state"] == (
        "R_H*R_L<0, D_H>=2/3, and stratified disagreement rank>=0.80"
    )
    assert "immediately preceding exact four-hour decision" in features["onset"]
    assert manifest["clock"]["entry"] == "exact BTCUSDT decision+5m open"
    assert manifest["clock"]["hold"] == "8 elapsed hours"
    assert manifest["clock"]["reservation"] == "global half-open; exit first on equal open"
    assert manifest["source_support_gates"]["minimum_events"] == {
        "train": 8,
        "test": 12,
        "eval": 12,
        "final": 8,
    }


def test_novelty_economics_rv_audit_and_controls_are_frozen():
    manifest = prereg.build()

    assert manifest["novelty_gates"] == {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_bar_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }
    economics = manifest["economic_gates"]
    assert economics["absolute_return_positive"] is True
    assert economics["cagr_to_strict_mdd_min"] == 3.0
    assert economics["strict_mdd_max_pct"] == 15.0
    assert economics["mean_gross_underlying_min_bp"] == 20.0
    assert economics["weekly_signflip_one_sided_p_max"] == 0.1
    assert economics["stress_absolute_return_positive"] is True
    assert economics["stress_cagr_to_strict_mdd_min"] == 2.5
    assert economics["each_calendar_half_positive"] is True
    assert manifest["rv20_stress_slice"]["entry_filter"] is False
    assert manifest["post_stage_volatility_audit"]["minimum_q90_trades"] == 8
    assert set(manifest["diagnostic_controls"]["definitions"]) == {
        "unweighted_6h_return",
        "high_volume_without_disagreement",
        "single_dominant_volume_bar",
        "temporal_half_partition",
        "one_decision_stale_strata",
        "direction_flip",
    }
    assert manifest["diagnostic_controls"]["cannot_be_promoted"] is True
