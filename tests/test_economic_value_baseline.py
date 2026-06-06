from training.economic_value_baseline import choose_best_per_signal


def test_choose_best_per_signal_applies_threshold_to_skip():
    rows = [
        {"date": "d", "signal_pos": 1, "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 36}, "predicted_utility": -0.1, "actual_utility": 0.0},
        {"date": "d", "signal_pos": 1, "prediction": {"gate": "TRADE", "side": "SHORT", "hold_bars": 36}, "predicted_utility": -0.2, "actual_utility": 0.0},
    ]
    out = choose_best_per_signal(rows, threshold=0.0)
    assert out[0]["prediction"] == {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
