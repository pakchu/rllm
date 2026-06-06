import json

from training.economic_drift_diagnostic import action_distribution, rows_by_signal_action, utility_summary


def test_utility_summary_reports_mean_and_ci_in_pct():
    s = utility_summary([0.01, -0.005, 0.0])
    assert s["n"] == 3
    assert round(s["mean_pct"], 4) == 0.1667
    assert round(s["win_rate"], 4) == 0.3333
    assert len(s["ci95_mean_pct"]) == 2


def test_rows_by_signal_action_indexes_normalized_action_text():
    row = {"date": "d", "signal_pos": 7, "action": json.dumps({"side": "LONG", "gate": "TRADE", "hold_bars": 72}), "utility": 0.01}
    idx = rows_by_signal_action([row])
    assert idx[("d", 7)][json.dumps({"gate": "TRADE", "hold_bars": 72, "side": "LONG"}, sort_keys=True, separators=(",", ":"))] is row


def test_action_distribution_counts_normalized_actions():
    rows = [
        {"action": json.dumps({"side": "LONG", "gate": "TRADE", "hold_bars": 72})},
        {"action": json.dumps({"gate": "TRADE", "hold_bars": 72, "side": "LONG"})},
    ]
    counts = action_distribution(rows)
    assert counts == {json.dumps({"gate": "TRADE", "hold_bars": 72, "side": "LONG"}, sort_keys=True, separators=(",", ":")): 2}
