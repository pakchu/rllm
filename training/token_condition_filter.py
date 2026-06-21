"""Filter predictions when selected candidate prompt contains blocked tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _action_key(date: Any, signal_pos: Any, action: dict[str, Any]) -> tuple[str, int, str, str, int]:
    return (str(date), int(signal_pos or -1), str(action.get("family", "")), str(action.get("side", "")).upper(), int(action.get("hold_bars", 0) or 0))


def _pred_action(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") == "TRADE":
        return {"family": pred.get("family"), "side": str(pred.get("side", "")).upper(), "hold_bars": int(pred.get("hold_bars", 0) or 0)}
    act = row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}
    return {"family": act.get("family"), "side": str(act.get("side", "")).upper(), "hold_bars": int(act.get("hold_bars", 0) or 0)}


def _tokens(prompt: str) -> set[str]:
    out: set[str] = set()
    for line in str(prompt).splitlines():
        if line.startswith("Regime tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                t = part.strip()
                if t:
                    out.add(t)
        elif line.startswith("Selected action tokens:"):
            for part in line.split(":", 1)[1].split(";"):
                t = part.strip()
                if t:
                    out.add("action_" + t)
    return out


def run(*, candidate_jsonl: str, predictions_jsonl: str, output: str, family: str, blocked_tokens: str) -> dict[str, Any]:
    block = {x.strip() for x in blocked_tokens.split(",") if x.strip()}
    cand_by_key = {_action_key(r.get("date"), r.get("signal_pos"), r.get("action", {}) if isinstance(r.get("action"), dict) else {}): r for r in _read(candidate_jsonl)}
    rows = _read(predictions_jsonl)
    out = []
    blocked = missing = allowed = no_trade = 0
    for row in rows:
        pred = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if pred.get("gate") != "TRADE":
            no_trade += 1
            out.append(row)
            continue
        action = _pred_action(row)
        if str(action.get("family")) != str(family):
            allowed += 1
            out.append(row)
            continue
        cand = cand_by_key.get(_action_key(row.get("date"), row.get("signal_pos"), action))
        toks = set() if cand is None else _tokens(str(cand.get("prompt", "")))
        if cand is None:
            missing += 1
        if toks & block:
            blocked += 1
            out.append({**row, "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "TOKEN_FILTER", "confidence": "HIGH"}, "blocked_tokens": sorted(toks & block)})
        else:
            allowed += 1
            out.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"candidate_jsonl": candidate_jsonl, "predictions_jsonl": predictions_jsonl, "output": output, "family": family, "blocked_tokens": sorted(block), "rows": len(rows), "blocked": blocked, "allowed_trade_rows": allowed, "input_no_trade_rows": no_trade, "missing_candidate": missing}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token-conditioned prediction filter")
    p.add_argument("--candidate-jsonl", required=True)
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--blocked-tokens", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
