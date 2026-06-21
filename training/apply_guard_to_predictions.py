"""Apply auxiliary guard score streams to sparse action predictions.

Supports:
- exact match guard rows, e.g. another sparse symbolic prediction stream.
- asof guard rows, e.g. dense 5m feature-signal predictions; the latest guard
  at or before the base signal is used within a max bar tolerance.

The guard can block trades or apply position_scale when guard conditions fail.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def _trade(row: dict[str, Any]) -> bool:
    p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    return p.get("gate") == "TRADE"


def _side(row: dict[str, Any]) -> str:
    p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
    return str(p.get("side", "NONE")).upper()


def _selected(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("selected_action", {}) if isinstance(row.get("selected_action"), dict) else {}


def _guard_ok(base: dict[str, Any], guard: dict[str, Any] | None, *, rule: str, min_score: float, require_same_side: bool, require_trade: bool) -> bool:
    if guard is None:
        return False
    if require_trade and not _trade(guard):
        return False
    if require_same_side and _trade(guard) and _side(base) != _side(guard):
        return False
    if rule == "score_min":
        return float(guard.get("predicted_utility", guard.get("feature_value", -1e9)) or -1e9) >= float(min_score)
    if rule == "trade":
        return _trade(guard)
    if rule == "same_action":
        if not _trade(guard):
            return False
        g = _selected(guard)
        bp = base.get("prediction", {}) if isinstance(base.get("prediction"), dict) else {}
        return str(g.get("side", "")).upper() == str(bp.get("side", "")).upper() and int(g.get("hold_bars", 0) or 0) == int(bp.get("hold_bars", 0) or 0) and str(g.get("family", "")) == str(bp.get("family", ""))
    if rule == "side_agree_or_score":
        if _trade(guard) and _side(base) == _side(guard):
            return True
        return float(guard.get("predicted_utility", guard.get("feature_value", -1e9)) or -1e9) >= float(min_score)
    raise ValueError(f"unknown rule: {rule}")


def run(*, base_predictions: str, guard_predictions: str, output: str, mode: str, rule: str, min_score: float, require_same_side: bool, require_trade: bool, max_pos_lag: int, fail_action: str, fail_scale: float) -> dict[str, Any]:
    base = read_jsonl(base_predictions)
    guards = sorted(read_jsonl(guard_predictions), key=lambda r: int(r.get("signal_pos", -1) or -1))
    exact = {(str(r.get("date")), int(r.get("signal_pos", -1) or -1)): r for r in guards}
    positions = [int(r.get("signal_pos", -1) or -1) for r in guards]
    out: list[dict[str, Any]] = []
    stats = {"rows": 0, "trade_rows": 0, "guard_missing": 0, "guard_pass": 0, "guard_fail": 0, "blocked": 0, "scaled": 0}
    for row in base:
        stats["rows"] += 1
        if not _trade(row):
            out.append(row)
            continue
        stats["trade_rows"] += 1
        guard: dict[str, Any] | None = None
        if mode == "exact":
            guard = exact.get((str(row.get("date")), int(row.get("signal_pos", -1) or -1)))
        elif mode == "asof_pos":
            pos = int(row.get("signal_pos", -1) or -1)
            idx = bisect_right(positions, pos) - 1
            if idx >= 0 and 0 <= pos - positions[idx] <= int(max_pos_lag):
                guard = guards[idx]
        else:
            raise ValueError(f"unknown mode: {mode}")
        if guard is None:
            stats["guard_missing"] += 1
        ok = _guard_ok(row, guard, rule=rule, min_score=min_score, require_same_side=require_same_side, require_trade=require_trade)
        if ok:
            stats["guard_pass"] += 1
            row = {**row, "guard": guard}
        else:
            stats["guard_fail"] += 1
            p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
            if fail_action == "block":
                row = {**row, "blocked_prediction": p, "guard": guard, "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "GUARD_BLOCK", "confidence": "HIGH"}}
                stats["blocked"] += 1
            elif fail_action == "scale":
                row = {**row, "guard": guard, "position_scale": float(fail_scale)}
                stats["scaled"] += 1
            else:
                raise ValueError(f"unknown fail_action: {fail_action}")
        out.append(row)
    write_jsonl(output, out)
    return {"base_predictions": base_predictions, "guard_predictions": guard_predictions, "output": output, "config": {"mode": mode, "rule": rule, "min_score": min_score, "require_same_side": require_same_side, "require_trade": require_trade, "max_pos_lag": max_pos_lag, "fail_action": fail_action, "fail_scale": fail_scale}, **stats}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply auxiliary guard predictions to base action stream")
    p.add_argument("--base-predictions", required=True)
    p.add_argument("--guard-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["exact", "asof_pos"], default="exact")
    p.add_argument("--rule", choices=["score_min", "trade", "same_action", "side_agree_or_score"], default="score_min")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--require-same-side", action="store_true")
    p.add_argument("--require-trade", action="store_true")
    p.add_argument("--max-pos-lag", type=int, default=72)
    p.add_argument("--fail-action", choices=["block", "scale"], default="block")
    p.add_argument("--fail-scale", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
