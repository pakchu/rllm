"""Aggregate executed trades across walk-forward fold outputs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import _trade_stats


def _years(start: str, end: str) -> float:
    a = datetime.fromisoformat(str(start))
    b = datetime.fromisoformat(str(end))
    return max(1.0 / 365.25, (b - a).days / 365.25)


def aggregate(path: str) -> dict[str, Any]:
    wf = json.loads(Path(path).read_text())
    trades: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    for fold in wf.get("folds", []):
        if fold.get("skipped"):
            continue
        sim = fold.get("eval_fixed", {}).get("sim", {})
        period = sim.get("period") or {}
        if not period:
            f = fold.get("fold", {})
            period = {"start": f.get("eval_start"), "end": f.get("eval_end")}
        periods.append({"name": fold.get("fold", {}).get("name"), "gate_passed": fold.get("validation_gate_passed"), **period})
        for tr in fold.get("eval_fixed", {}).get("executed", []):
            trades.append({**tr, "fold": fold.get("fold", {}).get("name")})
    trades.sort(key=lambda r: str(r.get("signal_date", "")))
    seen: dict[tuple[str, str], int] = {}
    for tr in trades:
        key = (str(tr.get("signal_date", "")), str(tr.get("side", "")))
        seen[key] = seen.get(key, 0) + 1
    duplicate_keys = {"|".join(k): v for k, v in seen.items() if v > 1}

    eq = peak = 1.0
    max_dd = 0.0
    returns = []
    equity_points = []
    for tr in trades:
        ret = float(tr.get("executed_ret_pct", 0.0)) / 100.0
        eq *= max(0.0, 1.0 + ret)
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)
        returns.append(ret)
        equity_points.append({"date": tr.get("signal_date"), "fold": tr.get("fold"), "equity": eq, "ret_pct": ret * 100.0})

    if periods:
        start = min(str(p.get("start")) for p in periods if p.get("start"))
        end = max(str(p.get("end")) for p in periods if p.get("end"))
    elif trades:
        start = str(trades[0].get("signal_date"))
        end = str(trades[-1].get("signal_date"))
    else:
        start = end = datetime.now(timezone.utc).isoformat()
    yrs = _years(start, end)
    ret_pct = (eq - 1.0) * 100.0
    cagr = ((eq ** (1.0 / yrs) - 1.0) * 100.0) if eq > 0 else -100.0
    by_fold: dict[str, dict[str, Any]] = {}
    for tr in trades:
        name = str(tr.get("fold"))
        bucket = by_fold.setdefault(name, {"n": 0, "simple_sum_ret_pct": 0.0, "long": 0, "short": 0})
        bucket["n"] += 1
        bucket["simple_sum_ret_pct"] += float(tr.get("executed_ret_pct", 0.0))
        side = str(tr.get("side", "")).upper()
        if side == "LONG":
            bucket["long"] += 1
        elif side == "SHORT":
            bucket["short"] += 1
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input": path,
        "period": {"start": start, "end": end, "years": yrs},
        "fold_periods": periods,
        "trade_count": len(trades),
        "duplicate_trade_keys": duplicate_keys,
        "overlap_warning": bool(duplicate_keys),
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr,
            "strict_mdd_pct": max_dd * 100.0,
            "cagr_to_strict_mdd": cagr / (max_dd * 100.0) if max_dd > 1e-12 else 0.0,
            "return_application": "walk_forward_executed_trade_compound",
        },
        "trade_stats": _trade_stats(returns),
        "by_fold": by_fold,
        "equity_points": equity_points,
        "executed": trades,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate walk-forward executed trades")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = aggregate(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"period": out["period"], "trade_count": out["trade_count"], "overlap_warning": out["overlap_warning"], "duplicate_count": len(out["duplicate_trade_keys"]), "sim": out["sim"], "trade_stats": out["trade_stats"], "by_fold": out["by_fold"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
