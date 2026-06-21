"""Apply auditable action-level gates to prediction streams.

Unlike position scaling, this converts selected matching trades to NO_TRADE so
cooldown and execution reflect actually skipping the signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rule(name: str) -> Callable[[dict[str, Any]], bool]:
    if name == "short_144":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" and int(p.get("hold_bars", 0) or 0) == 144
    if name == "short_72_144":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" and int(p.get("hold_bars", 0) or 0) in {72, 144}
    if name == "short_72":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" and int(p.get("hold_bars", 0) or 0) == 72
    if name == "short_432":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" and int(p.get("hold_bars", 0) or 0) == 432
    if name == "long_432":
        return lambda p: str(p.get("side", "")).upper() == "LONG" and int(p.get("hold_bars", 0) or 0) == 432
    if name == "drawdown_reversal":
        return lambda p: str(p.get("family")) == "drawdown_reversal"
    if name == "long_432_or_drawdown_reversal":
        return lambda p: (str(p.get("side", "")).upper() == "LONG" and int(p.get("hold_bars", 0) or 0) == 432) or str(p.get("family")) == "drawdown_reversal"
    raise ValueError(f"unknown action gate rule: {name}")


def run(*, predictions_jsonl: str, output: str, rule: str) -> dict[str, Any]:
    pred = _rule(rule)
    rows = _read(predictions_jsonl)
    out: list[dict[str, Any]] = []
    blocked = 0
    for row in rows:
        p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if p.get("gate") == "TRADE" and pred(p):
            row = {
                **row,
                "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "ACTION_GATE", "confidence": "HIGH"},
                "blocked_prediction": p,
            }
            blocked += 1
        out.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"predictions_jsonl": predictions_jsonl, "output": output, "rule": rule, "rows": len(rows), "blocked_trade_rows": blocked}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply action-level NO_TRADE gates")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rule", required=True, choices=["short_144", "short_72_144", "short_72", "short_432", "long_432", "drawdown_reversal", "long_432_or_drawdown_reversal"])
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
