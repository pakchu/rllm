"""Select trading modules by validation evidence before eval execution.

This script is deliberately boring: it does not tune, rescore, or inspect eval
performance while selecting.  A manifest lists precomputed module validation
and eval result files.  For each fold, only modules whose validation result
passes a fixed gate are eligible; the best eligible validation module contributes
its already-executed eval trades.  If no module passes, the fold is no-trade.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import _trade_stats


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _sim(obj: dict[str, Any]) -> dict[str, Any]:
    if "sim" in obj:
        return dict(obj["sim"])
    if "result" in obj and "sim" in obj["result"]:
        return dict(obj["result"]["sim"])
    raise ValueError("result JSON has no sim block")


def _stats(obj: dict[str, Any]) -> dict[str, Any]:
    if "trade_stats" in obj:
        return dict(obj["trade_stats"])
    if "result" in obj and "trade_stats" in obj["result"]:
        return dict(obj["result"]["trade_stats"])
    return {}


def _executed(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if "executed" in obj:
        return list(obj.get("executed") or [])
    if "result" in obj and "executed" in obj["result"]:
        return list(obj["result"].get("executed") or [])
    return []


def _period_from_eval(eval_obj: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    sim = _sim(eval_obj)
    if "period" in sim:
        return dict(sim["period"])
    if "period" in eval_obj:
        return dict(eval_obj["period"])
    return {"start": fold.get("eval_start", ""), "end": fold.get("eval_end", ""), "years": 0.0}


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _years(start: str, end: str) -> float:
    if not start or not end:
        return 1.0 / 365.25
    return max(1.0 / 365.25, (_parse_dt(end) - _parse_dt(start)).days / 365.25)


def _passes(sim: dict[str, Any], stats: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        int(sim.get("trade_entries", 0)) >= int(gate.get("min_trades", 0))
        and float(sim.get("cagr_to_strict_mdd", -1e9)) >= float(gate.get("min_ratio", -1e9))
        and float(sim.get("strict_mdd_pct", 1e9)) <= float(gate.get("max_mdd", 1e9))
        and float(stats.get("p_value_mean_ret_approx", 1.0)) <= float(gate.get("max_p", 1.0))
    )


def _rank_key(candidate: dict[str, Any]) -> tuple[float, float, int, float]:
    sim = candidate["validation"]["sim"]
    stats = candidate["validation"]["trade_stats"]
    return (
        float(sim.get("cagr_to_strict_mdd", -1e9)),
        -float(stats.get("p_value_mean_ret_approx", 1.0)),
        int(sim.get("trade_entries", 0)),
        float(sim.get("cagr_pct", -1e9)),
    )


def _compound(trades: list[dict[str, Any]], periods: list[dict[str, Any]], component_mdds: list[float]) -> dict[str, Any]:
    trades = sorted(trades, key=lambda r: str(r.get("signal_date", "")))
    eq = peak = 1.0
    trade_to_trade_mdd = 0.0
    returns: list[float] = []
    equity_points: list[dict[str, Any]] = []
    by_module: dict[str, dict[str, Any]] = {}
    for tr in trades:
        ret = float(tr.get("executed_ret_pct", 0.0)) / 100.0
        eq *= max(0.0, 1.0 + ret)
        peak = max(peak, eq)
        if peak > 0:
            trade_to_trade_mdd = max(trade_to_trade_mdd, 1.0 - eq / peak)
        returns.append(ret)
        module = str(tr.get("module", ""))
        bucket = by_module.setdefault(module, {"n": 0, "simple_sum_ret_pct": 0.0, "long": 0, "short": 0})
        bucket["n"] += 1
        bucket["simple_sum_ret_pct"] += ret * 100.0
        side = str(tr.get("side", "")).upper()
        if side == "LONG":
            bucket["long"] += 1
        elif side == "SHORT":
            bucket["short"] += 1
        equity_points.append({"date": tr.get("signal_date"), "module": module, "ret_pct": ret * 100.0, "equity": eq})
    starts = [str(p.get("start")) for p in periods if p.get("start")]
    ends = [str(p.get("end")) for p in periods if p.get("end")]
    start = min(starts) if starts else (str(trades[0].get("signal_date")) if trades else "")
    end = max(ends) if ends else (str(trades[-1].get("signal_date")) if trades else "")
    yrs = _years(start, end)
    cagr = ((eq ** (1.0 / yrs) - 1.0) * 100.0) if eq > 0 else -100.0
    mdd = max([trade_to_trade_mdd * 100.0, *component_mdds], default=0.0)
    return {
        "period": {"start": start, "end": end, "years": yrs},
        "trade_count": len(trades),
        "sim": {
            "ret_pct": (eq - 1.0) * 100.0,
            "cagr_pct": cagr,
            "strict_mdd_pct": mdd,
            "trade_to_trade_mdd_pct": trade_to_trade_mdd * 100.0,
            "component_strict_mdd_floor_pct": max(component_mdds, default=0.0),
            "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
            "return_application": "validated_module_eval_trades_compound_with_component_strict_mdd_floor",
        },
        "trade_stats": _trade_stats(returns),
        "by_module": by_module,
        "equity_points": equity_points,
    }


def select_validated_modules(manifest_path: str, output: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    gate = manifest.get("validation_gate", {})
    selected_folds: list[dict[str, Any]] = []
    accepted_trades: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    component_mdds: list[float] = []
    for fold in manifest.get("folds", []):
        candidates = []
        for mod in fold.get("modules", []):
            val_obj = _read_json(mod["validation_result"])
            eval_obj = _read_json(mod["eval_result"])
            val_sim = _sim(val_obj)
            val_stats = _stats(val_obj)
            eval_sim = _sim(eval_obj)
            eval_stats = _stats(eval_obj)
            passes = _passes(val_sim, val_stats, gate)
            candidates.append(
                {
                    "name": mod["name"],
                    "validation_result": mod["validation_result"],
                    "eval_result": mod["eval_result"],
                    "passes_gate": passes,
                    "validation": {"sim": val_sim, "trade_stats": val_stats},
                    "eval": {"sim": eval_sim, "trade_stats": eval_stats, "executed_count": len(_executed(eval_obj))},
                }
            )
        eligible = [c for c in candidates if c["passes_gate"]]
        period = None
        if candidates:
            period = _period_from_eval(_read_json(candidates[0]["eval_result"]), fold)
            periods.append(period)
        if eligible:
            chosen = max(eligible, key=_rank_key)
            eval_obj = _read_json(chosen["eval_result"])
            trades = [{**tr, "module": chosen["name"], "fold": fold.get("name")} for tr in _executed(eval_obj)]
            accepted_trades.extend(trades)
            component_mdds.append(float(chosen["eval"]["sim"].get("strict_mdd_pct", 0.0) or 0.0))
            selected_folds.append(
                {
                    "name": fold.get("name"),
                    "selected": chosen["name"],
                    "no_trade": False,
                    "period": period,
                    "candidates": candidates,
                }
            )
        else:
            selected_folds.append(
                {
                    "name": fold.get("name"),
                    "selected": None,
                    "no_trade": True,
                    "period": period,
                    "candidates": candidates,
                }
            )
    aggregate = _compound(accepted_trades, periods, component_mdds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path,
        "validation_gate": gate,
        "folds": selected_folds,
        "aggregate": aggregate,
        "executed": accepted_trades,
        "leakage_guard": {
            "selection_uses_validation_results_only": True,
            "eval_results_loaded_for_reporting_but_not_for_selection": True,
            "no_module_selected_from_eval_performance": True,
            "component_strict_mdd_floor_applied": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select modules by validation gate and aggregate eval trades")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = select_validated_modules(args.manifest, args.output)
    aggregate = out["aggregate"]
    print(
        json.dumps(
            {
                "validation_gate": out["validation_gate"],
                "folds": [
                    {"name": f["name"], "selected": f["selected"], "no_trade": f["no_trade"]}
                    for f in out["folds"]
                ],
                "aggregate": {
                    "period": aggregate["period"],
                    "trade_count": aggregate["trade_count"],
                    "sim": aggregate["sim"],
                    "trade_stats": aggregate["trade_stats"],
                    "by_module": aggregate["by_module"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
