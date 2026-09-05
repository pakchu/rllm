from training import distill_oi_divergence_fresh_candidate as d


def test_distilled_oi_candidate_is_disabled_and_frozen():
    candidate = d.build()
    evidence = candidate["evidence"]
    assert not candidate["enabled"]
    assert not candidate["live_authorized"]
    assert candidate["position_overlap_allowed"] is False
    assert candidate["execution"]["hold_5m_bars"] == 96
    assert candidate["execution"]["decision_stride_5m_bars"] == 6
    assert [gate["threshold"] for gate in candidate["signal"]["gates"]] == [
        0.8954018630586817,
        -0.7389570664259131,
        0.04008415457867338,
        -0.04507656773717145,
    ]
    assert evidence["scheduled_nonoverlap_trades"] == 8
    assert evidence["fresh_base"]["absolute_return_pct"] > 0
    assert evidence["fresh_stress"]["absolute_return_pct"] > 0
    assert evidence["historical_validation_from_source"]["train_pre2024"]["cagr_to_strict_mdd"] < 1
