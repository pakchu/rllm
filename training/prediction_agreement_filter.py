"""Filter primary predictions by agreement with a secondary prediction stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _pred(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}


def run(*, primary_jsonl: str, secondary_jsonl: str, output: str, agreement: str = "side") -> dict[str, Any]:
    secondary = {_key(r): r for r in _read(secondary_jsonl)}
    rows = _read(primary_jsonl)
    out = []
    kept = blocked = missing = primary_no_trade = secondary_no_trade = 0
    for row in rows:
        p = _pred(row)
        if p.get("gate") != "TRADE":
            primary_no_trade += 1
            out.append(row)
            continue
        srow = secondary.get(_key(row))
        if srow is None:
            missing += 1
            ok = False
            sp = {}
        else:
            sp = _pred(srow)
            if sp.get("gate") != "TRADE":
                secondary_no_trade += 1
                ok = False
            elif agreement == "side":
                ok = str(p.get("side")).upper() == str(sp.get("side")).upper()
            elif agreement == "family_side":
                ok = str(p.get("side")).upper() == str(sp.get("side")).upper() and str(p.get("family")) == str(sp.get("family"))
            elif agreement == "trade":
                ok = True
            else:
                raise ValueError("agreement must be side, family_side, or trade")
        if ok:
            kept += 1
            out.append({**row, "secondary_prediction": sp})
        else:
            blocked += 1
            out.append({**row, "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "AGREEMENT_FILTER", "confidence": "HIGH"}, "secondary_prediction": sp})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"primary_jsonl": primary_jsonl, "secondary_jsonl": secondary_jsonl, "output": output, "agreement": agreement, "rows": len(rows), "kept_trade_rows": kept, "blocked_trade_rows": blocked, "missing_secondary": missing, "primary_no_trade": primary_no_trade, "secondary_no_trade_on_primary_trade": secondary_no_trade}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediction agreement filter")
    p.add_argument("--primary-jsonl", required=True)
    p.add_argument("--secondary-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--agreement", choices=["side", "family_side", "trade"], default="side")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
