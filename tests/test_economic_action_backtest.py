import json
from pathlib import Path

import pandas as pd

from training.economic_action_backtest import (
    EconomicActionBacktestConfig,
    dedupe_signal_predictions,
    run_economic_action_backtest,
    strict_backtest_actions,
)


def test_dedupe_signal_predictions_keeps_first():
    rows = [
        {"date": "2025-01-01 00:00:00", "signal_pos": 1, "prediction": {"gate": "TRADE"}},
        {"date": "2025-01-01 00:00:00", "signal_pos": 1, "prediction": {"gate": "NO_TRADE"}},
        {"date": "2025-01-01 00:05:00", "signal_pos": 2, "prediction": {"gate": "NO_TRADE"}},
    ]
    out = dedupe_signal_predictions(rows)
    assert len(out) == 2
    assert out[0]["prediction"]["gate"] == "TRADE"


def test_run_economic_action_backtest_smoke(tmp_path: Path):
    market = tmp_path / "market.csv"
    market.write_text(
        "date,open,high,low,close\n"
        "2025-01-01 00:00:00,100,101,99,100\n"
        "2025-01-01 00:05:00,100,103,99,102\n"
        "2025-01-01 00:10:00,102,104,101,103\n"
        "2025-01-01 00:15:00,103,104,102,103\n"
    )
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps(
            {
                "date": "2025-01-01 00:00:00",
                "signal_pos": 0,
                "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 2},
            }
        )
        + "\n"
    )
    out = run_economic_action_backtest(
        predictions_jsonl=str(preds),
        market_csv=str(market),
        output=str(tmp_path / "out.json"),
        leverage=1.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        entry_delay_bars=1,
    )
    assert out["backtest"]["sim"]["trade_entries"] == 1
    assert out["backtest"]["sim"]["ret_pct"] > 0


def test_strict_action_mdd_uses_intratrade_favorable_high_water():
    market = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="5min"),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 200.0, 100.0, 100.0],
            "low": [100.0, 90.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )
    rows = [
        {
            "date": "2025-01-01 00:00:00",
            "signal_pos": 0,
            "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 1},
        }
    ]
    out = strict_backtest_actions(
        rows,
        market,
        EconomicActionBacktestConfig(
            annualization_start="2025-01-01",
            annualization_end="2026-01-01",
            leverage=1.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            entry_delay_bars=1,
        ),
    )
    assert abs(out["sim"]["ret_pct"]) < 1e-12
    assert abs(out["sim"]["strict_mdd_pct"] - 55.0) < 1e-12
