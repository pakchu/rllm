from training.economic_opportunity_baseline import best_rows_by_signal


def test_best_rows_by_signal_selects_highest_utility():
    rows = [
        {"date": "d", "signal_pos": 1, "utility": -1, "action": "A"},
        {"date": "d", "signal_pos": 1, "utility": 2, "action": "B"},
    ]
    assert best_rows_by_signal(rows)[0]["action"] == "B"
