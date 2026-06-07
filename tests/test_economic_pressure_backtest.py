import pandas as pd

from training.economic_pressure_backtest import PressureBacktestConfig, pressure_to_side, strict_pressure_backtest


def test_pressure_to_side_maps_only_directional_labels():
    assert pressure_to_side({"prediction": {"direction_pressure": "LONG_FAVORED"}}) == "LONG"
    assert pressure_to_side({"prediction": {"direction_pressure": "SHORT_FAVORED"}}) == "SHORT"
    assert pressure_to_side({"prediction": {"direction_pressure": "NO_TRADE_FAVORED"}}) == "NONE"


def test_strict_pressure_backtest_exits_target():
    market = pd.DataFrame([
        {"open": 100.0, "high": 100.0, "low": 100.0},
        {"open": 100.0, "high": 100.7, "low": 99.9},
        {"open": 100.5, "high": 100.8, "low": 100.2},
    ])
    rows = [{"date": "2025-01-01 00:00:00", "signal_pos": 0, "prediction": {"direction_pressure": "LONG_FAVORED"}}]
    bt = strict_pressure_backtest(rows, market, PressureBacktestConfig(horizon_bars=1, target_pct=0.5, stop_pct=0.6, entry_delay_bars=1))
    assert bt["sim"]["trade_entries"] == 1
    assert bt["sim"]["target_exits"] == 1
