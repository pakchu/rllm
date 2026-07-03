from training.sweep_single_feature_event_gate import to_preds, score


def test_to_preds_keeps_prediction_shape():
    rows = [{"date":"2025-01-01", "signal_pos":1, "prediction":{"gate":"TRADE", "side":"LONG", "hold_bars":144}}]
    assert to_preds(rows)[0]["prediction"]["side"] == "LONG"


def test_score_prefers_positive_ratio_with_trade_count():
    sim = {"sim":{"cagr_to_strict_mdd":2.0,"cagr_pct":10.0,"trade_entries":50}, "trade_stats":{"p_value_mean_ret_approx":0.05}}
    assert score(sim) > 2.0
