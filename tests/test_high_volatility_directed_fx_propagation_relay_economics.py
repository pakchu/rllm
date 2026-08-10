from training import evaluate_high_volatility_directed_fx_propagation_relay_economics as economics


def test_frozen_accounting_stage_order_and_controls():
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert tuple(economics.STAGES) == ("train", "test", "eval", "final")
    assert economics.PREDECESSOR == {"test": "train", "eval": "test", "final": "eval"}
    assert economics.CONTROLS == (
        "no_variation_gate", "no_source_strength_tail", "no_breadth_gate",
        "one_session_stale_network", "direction_flip", "forced_long",
    )


def test_public_metrics_removes_trade_rows_only():
    assert economics.public_metrics({"x": 1, "trade_rows": [1]}) == {"x": 1}
