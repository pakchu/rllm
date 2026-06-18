"""Compose explicit gate predictions with event-action predictions.

If the first-stage gate classifier says NO_TRADE, force a no-trade action.
Otherwise keep the action scorer's prediction.  Rows are joined by
(date, signal_pos), preserving one live-style action per signal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "NONE", "confidence": "HIGH"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def compose_predictions(*, gate_predictions_jsonl: str, action_predictions_jsonl: str, output: str, summary_output: str = "") -> dict[str, Any]:
    gate_rows = _read_jsonl(gate_predictions_jsonl)
    action_rows = _read_jsonl(action_predictions_jsonl)
    gate_by_key = {_key(r): str(r.get("prediction", "NO_TRADE")).upper() for r in gate_rows}
    out: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    missing_gate = 0
    for row in action_rows:
        key = _key(row)
        gate = gate_by_key.get(key)
        if gate is None:
            missing_gate += 1
            gate = "NO_TRADE"
        pred = dict(row.get("prediction", {})) if isinstance(row.get("prediction"), dict) else {}
        if gate != "TRADE":
            pred = dict(NO_TRADE)
            counts["forced_no_trade"] += 1
        else:
            pred["gate"] = "TRADE"
            counts[f"trade_{str(pred.get('side', 'NONE')).upper()}"] += 1
        out.append({**row, "prediction": pred, "gate_prediction": gate})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    summary = {
        "gate_predictions_jsonl": str(Path(gate_predictions_jsonl).resolve()),
        "action_predictions_jsonl": str(Path(action_predictions_jsonl).resolve()),
        "output": output,
        "rows": len(out),
        "missing_gate": missing_gate,
        "counts": dict(sorted(counts.items())),
        "leakage_guard": {"joins_on_signal_identity_only": True, "gate_forces_only_no_trade_or_trade": True},
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose two-stage gate + action event predictions")
    p.add_argument("--gate-predictions-jsonl", required=True)
    p.add_argument("--action-predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(compose_predictions(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
