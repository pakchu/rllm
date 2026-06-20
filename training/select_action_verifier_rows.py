"""Extract verifier rows matching actually selected trader actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(date: Any, signal_pos: Any, action: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (str(date), int(signal_pos or -1), str(action.get("family", "")), str(action.get("side", "")).upper(), int(action.get("hold_bars", 0) or 0))


def _selected_keys(prediction_files: str) -> set[tuple[str, int, str, str, int]]:
    keys = set()
    for path in [x for x in str(prediction_files).split(",") if x.strip()]:
        for row in _read(path):
            pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
            if pred.get("gate") != "TRADE":
                continue
            keys.add(_key(row.get("date"), row.get("signal_pos"), pred))
    return keys


def select_rows(*, verifier_jsonl: str, trader_predictions: str, output: str) -> dict[str, Any]:
    keys = _selected_keys(trader_predictions)
    selected = []
    for row in _read(verifier_jsonl):
        action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
        if _key(row.get("date"), row.get("signal_pos"), action) in keys:
            selected.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in selected) + "\n")
    return {"verifier_jsonl": verifier_jsonl, "trader_predictions": trader_predictions, "output": output, "selected_keys": len(keys), "rows": len(selected), "missing": len(keys) - len(selected)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select verifier rows for actual trader actions")
    p.add_argument("--verifier-jsonl", required=True)
    p.add_argument("--trader-predictions", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(select_rows(**vars(parse_args())), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
