import json

from training.economic_value_data import EconomicValueConfig, summarize_value_rows


def test_summarize_value_rows_counts_signals_and_actions():
    rows = [
        {"date": "d", "signal_pos": 1, "action": json.dumps({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}), "utility": 0.0},
        {"date": "d", "signal_pos": 1, "action": json.dumps({"gate": "TRADE", "side": "LONG", "hold_bars": 36}), "utility": 0.01},
    ]
    summary = summarize_value_rows(rows, cfg=EconomicValueConfig())
    assert summary["rows"] == 2
    assert summary["signals"] == 1
    assert summary["utility_pct"]["max"] == 1.0
