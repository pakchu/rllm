"""Filter trader predictions by exact-action verifier outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "ACTION_VERIFIER", "confidence": "HIGH"}


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key_from_action(row: dict[str, Any], action: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1), str(action.get("family", "")), str(action.get("side", "")).upper(), int(action.get("hold_bars", 0) or 0))


def _pred_action(row: dict[str, Any]) -> dict[str, Any]:
    p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    return {"family": p.get("family"), "side": str(p.get("side", "")).upper(), "hold_bars": int(p.get("hold_bars", 0) or 0)}


def apply_gate(*, trader_predictions: str, verifier_predictions: str, output: str, min_allow_margin: float | None = None) -> dict[str, Any]:
    verifier_by_key = {_key_from_action(r, r.get("action", {}) if isinstance(r.get("action"), dict) else {}): r for r in _read(verifier_predictions)}
    out = []
    allowed = blocked = missing = no_trade = 0
    for row in _read(trader_predictions):
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if pred.get("gate") != "TRADE":
            out.append(row); no_trade += 1; continue
        action = _pred_action(row)
        v = verifier_by_key.get(_key_from_action(row, action))
        allow = False
        if v is None:
            missing += 1
        else:
            allow = str(v.get("prediction", "")).upper() == "ALLOW"
            if min_allow_margin is not None:
                allow = allow and float(v.get("allow_margin", -1e9)) >= float(min_allow_margin)
        if allow:
            allowed += 1
            out.append({**row, "verifier_prediction": v.get("prediction"), "verifier_target": v.get("target"), "verifier_allow_margin": v.get("allow_margin")})
        else:
            blocked += 1
            out.append({**row, "prediction": dict(NO_TRADE), "verifier_prediction": None if v is None else v.get("prediction"), "verifier_target": None if v is None else v.get("target"), "verifier_allow_margin": None if v is None else v.get("allow_margin")})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"trader_predictions": trader_predictions, "verifier_predictions": verifier_predictions, "output": output, "rows": len(out), "allowed": allowed, "blocked": blocked, "missing_verifier": missing, "input_no_trade": no_trade, "min_allow_margin": min_allow_margin}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply exact-action verifier gate")
    p.add_argument("--trader-predictions", required=True)
    p.add_argument("--verifier-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-allow-margin", type=float, default=None)
    return p.parse_args()


def main() -> None:
    print(json.dumps(apply_gate(**vars(parse_args())), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
