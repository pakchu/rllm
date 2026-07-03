from training.backtest_decision_label_predictions import decision_rows_to_actions


def test_decision_rows_to_actions_uses_metadata_action_for_trade_only():
    rows = [
        {"date": "2025-01-01", "signal_pos": 1, "metadata": {"action": {"family": "rex", "side": "LONG", "hold_bars": 144}}},
        {"date": "2025-01-02", "signal_pos": 2, "metadata": {"action": {"family": "rex", "side": "SHORT", "hold_bars": 144}}},
    ]
    decisions = {("2025-01-01", 1): "TRADE", ("2025-01-02", 2): "ABSTAIN"}
    out = decision_rows_to_actions(rows, decisions)
    assert len(out) == 1
    assert out[0]["prediction"]["side"] == "LONG"
