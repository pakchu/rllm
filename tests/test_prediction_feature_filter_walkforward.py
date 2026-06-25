from pathlib import Path

import numpy as np

from training.prediction_feature_filter_walkforward import _apply, _is_trade, _side, _months, _slice


def test_apply_blocks_scoped_trades_only():
    rows = [
        {"date": "2026-01-01", "signal_pos": 0, "prediction": {"gate": "TRADE", "side": "LONG"}},
        {"date": "2026-01-01", "signal_pos": 1, "prediction": {"gate": "TRADE", "side": "SHORT"}},
    ]
    out = _apply(rows, np.array([2.0, 2.0]), "f", "le", 1.0, "LONG")
    assert out[0]["prediction"]["gate"] == "NO_TRADE"
    assert out[1]["prediction"]["gate"] == "TRADE"


def test_months_and_slice_are_half_open():
    assert [str(m.date()) for m in _months("2026-01-15", "2026-03-01")] == ["2026-01-01", "2026-02-01"]
    rows = [{"date": "2026-01-01"}, {"date": "2026-02-01"}]
    assert _slice(rows, "2026-01-01", "2026-02-01") == [rows[0]]
