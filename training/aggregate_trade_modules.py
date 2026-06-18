"""Aggregate multiple already-executed trade modules.

This is a portfolio-composition diagnostic, not a new alpha selector.  It reads
trade lists emitted by no-leak module backtests, sorts them chronologically, and
applies a simple priority rule: when trade windows overlap, keep the higher
priority module and skip the lower priority one.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import _trade_stats


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _extract_period(obj: dict[str, Any]) -> dict[str, Any]:
    if "period" in obj:
        return dict(obj["period"])
    if "result" in obj and "period" in obj["result"]:
        return dict(obj["result"]["period"])
    return {}


def _extract_executed(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if "executed" in obj:
        return list(obj.get("executed") or [])
    if "result" in obj and "executed" in obj["result"]:
        return list(obj["result"].get("executed") or [])
    return []


def _years(start: datetime, end: datetime) -> float:
    return max(1.0 / 365.25, float((end - start).days) / 365.25)


def aggregate_modules(module_specs: list[str], *, output: str) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for ordinal, spec in enumerate(module_specs):
        parts = spec.split("=", 2)
        if len(parts) == 3:
            priority = int(parts[0])
            name = parts[1]
            path = parts[2]
        elif len(parts) == 2:
            priority = ordinal
            name = parts[0]
            path = parts[1]
        else:
            raise ValueError(f"module spec must be [priority=]name=path: {spec}")
        obj = json.loads(Path(path).read_text())
        period = _extract_period(obj)
        if period:
            periods.append({"module": name, **period})
        extracted = []
        for tr in _extract_executed(obj):
            if "signal_date" not in tr or "executed_ret_pct" not in tr:
                continue
            row = {
                **tr,
                "module": name,
                "priority": int(priority),
                "_signal_dt": _parse_dt(tr["signal_date"]),
                "_entry_dt": _parse_dt(tr.get("entry_date", tr["signal_date"])),
                "_exit_dt": _parse_dt(tr.get("exit_date", tr.get("signal_date"))),
            }
            extracted.append(row)
        modules.append({"name": name, "priority": priority, "path": path, "period": period, "executed": len(extracted)})
        trades.extend(extracted)

    accepted: list[dict[str, Any]] = []
    skipped_overlap: list[dict[str, Any]] = []
    accepted_windows: list[tuple[datetime, datetime, str]] = []
    # Process higher-priority modules first.  Inside one priority tier, keep the
    # module's own chronological non-overlap behavior.
    for tr in sorted(trades, key=lambda r: (r["priority"], r["_signal_dt"])):
        overlaps = [
            (start, end, module)
            for start, end, module in accepted_windows
            if tr["_signal_dt"] < end and tr["_exit_dt"] > start
        ]
        if overlaps:
            first = min(overlaps, key=lambda r: r[0])
            skipped_overlap.append(
                {
                    "module": tr["module"],
                    "signal_date": tr["signal_date"],
                    "side": tr.get("side"),
                    "executed_ret_pct": tr.get("executed_ret_pct"),
                    "overlapped_module": first[2],
                    "overlapped_window": [first[0].isoformat(sep=" "), first[1].isoformat(sep=" ")],
                }
            )
            continue
        accepted.append(tr)
        accepted_windows.append((tr["_signal_dt"], tr["_exit_dt"], str(tr["module"])))
    accepted.sort(key=lambda r: r["_signal_dt"])

    eq = peak = 1.0
    max_dd = 0.0
    returns: list[float] = []
    equity_points: list[dict[str, Any]] = []
    by_module: dict[str, dict[str, Any]] = {}
    for tr in accepted:
        ret = float(tr.get("executed_ret_pct", 0.0)) / 100.0
        eq *= max(0.0, 1.0 + ret)
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)
        returns.append(ret)
        bucket = by_module.setdefault(str(tr["module"]), {"n": 0, "simple_sum_ret_pct": 0.0, "long": 0, "short": 0})
        bucket["n"] += 1
        bucket["simple_sum_ret_pct"] += ret * 100.0
        side = str(tr.get("side", "")).upper()
        if side == "LONG":
            bucket["long"] += 1
        elif side == "SHORT":
            bucket["short"] += 1
        equity_points.append(
            {
                "date": tr["signal_date"],
                "module": tr["module"],
                "side": tr.get("side"),
                "ret_pct": ret * 100.0,
                "equity": eq,
            }
        )

    period_starts = [_parse_dt(p["start"]) for p in periods if p.get("start")]
    period_ends = [_parse_dt(p["end"]) for p in periods if p.get("end")]
    if period_starts and period_ends:
        start = min(period_starts)
        end = max(period_ends)
    elif accepted:
        start = accepted[0]["_signal_dt"]
        end = accepted[-1]["_signal_dt"]
    else:
        start = end = datetime.now()
    yrs = _years(start, end)
    cagr = ((eq ** (1.0 / yrs) - 1.0) * 100.0) if eq > 0 else -100.0
    mdd = max_dd * 100.0
    public_trades = [
        {k: v for k, v in tr.items() if not str(k).startswith("_")}
        for tr in accepted
    ]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "modules": modules,
        "period": {"start": start.isoformat(sep=" "), "end": end.isoformat(sep=" "), "years": yrs},
        "trade_count": len(accepted),
        "skipped_overlap_count": len(skipped_overlap),
        "sim": {
            "ret_pct": (eq - 1.0) * 100.0,
            "cagr_pct": cagr,
            "strict_mdd_pct": mdd,
            "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
            "return_application": "accepted_module_executed_trade_compound",
        },
        "trade_stats": _trade_stats(returns),
        "by_module": by_module,
        "equity_points": equity_points,
        "executed": public_trades,
        "skipped_overlap": skipped_overlap,
        "leakage_guard": {
            "does_not_select_parameters": True,
            "combines_precomputed_no_leak_module_outputs": True,
            "overlap_resolution": "chronological_priority_skip_lower_or_later_overlaps",
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate precomputed executed trades from multiple modules")
    p.add_argument("--module", action="append", required=True, help="[priority=]name=path")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = aggregate_modules(args.module, output=args.output)
    print(json.dumps({k: out[k] for k in ("modules", "period", "trade_count", "skipped_overlap_count", "sim", "trade_stats", "by_module")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
