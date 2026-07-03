"""Backtest TRADE/ABSTAIN label predictions against action metadata rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars
from training.train_text_sft import load_jsonl


def _load_predictions(path: str) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[(str(row.get("date")), int(row.get("signal_pos", 0) or 0))] = str(row.get("prediction", "ABSTAIN")).upper()
    return out


def _action_from_row(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    action = meta.get("action", {}) if isinstance(meta.get("action"), dict) else row.get("action", {})
    return {"family": action.get("family"), "side": str(action.get("side", "")).upper(), "hold_bars": int(action.get("hold_bars", 0) or 0)}


def decision_rows_to_actions(rows: list[dict[str, Any]], decisions: dict[tuple[str, int], str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", 0) or 0))
        decision = decisions.get(key, "ABSTAIN")
        if decision != "TRADE":
            continue
        action = _action_from_row(row)
        if action["side"] not in {"LONG", "SHORT"} or not action["hold_bars"]:
            continue
        actions.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "prediction": {"gate": "TRADE", **action}})
    return actions


def backtest_decision_predictions(*, rows_jsonl: str, predictions_jsonl: str, market_csv: str, output: str, leverage_grid: str = "0.5,1.0,1.25,1.5,2.0") -> dict[str, Any]:
    rows = load_jsonl(rows_jsonl)
    decisions = _load_predictions(predictions_jsonl)
    action_rows = decision_rows_to_actions(rows, decisions)
    market = load_market_bars(market_csv)
    results = []
    for raw in str(leverage_grid).split(","):
        if not raw.strip():
            continue
        lev = float(raw)
        sim = strict_backtest_actions(
            action_rows,
            market,
            EconomicActionBacktestConfig(leverage=lev, fee_rate=0.0004, slippage_rate=0.0001, entry_delay_bars=1, max_hold_bars=144),
        )
        results.append({"leverage": lev, "n_action_rows": len(action_rows), "sim": sim["sim"], "trade_stats": sim["trade_stats"]})
    report = {
        "rows_jsonl": rows_jsonl,
        "predictions_jsonl": predictions_jsonl,
        "market_csv": market_csv,
        "predicted_counts": {"TRADE": sum(1 for v in decisions.values() if v == "TRADE"), "ABSTAIN": sum(1 for v in decisions.values() if v != "TRADE")},
        "results": results,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest TRADE/ABSTAIN predictions against stored action metadata")
    p.add_argument("--rows-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--leverage-grid", default="0.5,1.0,1.25,1.5,2.0")
    return p.parse_args()


def main() -> None:
    print(json.dumps(backtest_decision_predictions(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
