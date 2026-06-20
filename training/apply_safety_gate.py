"""Apply pre-entry safety predictions to action-value trader predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "SAFETY_GATE", "confidence": "HIGH"}


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def apply_gate(*, trader_predictions: str, safety_predictions: str, output: str, safe_label: str = "SAFE_TRADE", min_safe_margin: float | None = None) -> dict[str, Any]:
    trader_rows = _read(trader_predictions)
    safety_by_key = {_key(r): r for r in _read(safety_predictions)}
    out_rows: list[dict[str, Any]] = []
    missing = allowed = blocked = 0
    for row in trader_rows:
        safety = safety_by_key.get(_key(row))
        allow = False
        if safety is None:
            missing += 1
        else:
            allow = str(safety.get("prediction", "")).upper() == safe_label
            if min_safe_margin is not None:
                allow = allow and float(safety.get("safe_margin", -1e9)) >= float(min_safe_margin)
        if allow:
            pred = row.get("prediction", {})
            allowed += 1
        else:
            pred = dict(NO_TRADE)
            blocked += 1
        out_rows.append({
            **row,
            "prediction": pred,
            "safety_prediction": None if safety is None else safety.get("prediction"),
            "safety_target": None if safety is None else safety.get("target"),
            "safety_safe_margin": None if safety is None else safety.get("safe_margin"),
        })
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + "\n")
    return {"trader_predictions": trader_predictions, "safety_predictions": safety_predictions, "output": output, "rows": len(out_rows), "allowed": allowed, "blocked": blocked, "missing_safety": missing, "safe_label": safe_label, "min_safe_margin": min_safe_margin}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter trader predictions through safety classifier")
    p.add_argument("--trader-predictions", required=True)
    p.add_argument("--safety-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--safe-label", default="SAFE_TRADE")
    p.add_argument("--min-safe-margin", type=float, default=None)
    return p.parse_args()


def main() -> None:
    print(json.dumps(apply_gate(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
