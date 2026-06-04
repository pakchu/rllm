import json
from pathlib import Path

from training.economic_action_backtest import dedupe_signal_predictions, run_economic_action_backtest


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
