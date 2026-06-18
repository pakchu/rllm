"""Backtest the intersection of selected candidates from multiple modules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, load_market, simulate_candidates


def _rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if "guarded_candidates" in obj:
        return list(obj.get("guarded_candidates") or [])
    if "scored_candidates" in obj:
        return list(obj.get("scored_candidates") or [])
    if "executed" in obj:
        return list(obj.get("executed") or [])
    raise ValueError("module result lacks guarded_candidates/scored_candidates/executed")


def _threshold(obj: dict[str, Any]) -> float:
    cfg = obj.get("config", {}) or {}
    return float(cfg.get("score_threshold", obj.get("score_threshold", 0.0)))


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("signal_date", row.get("date", ""))), str(row.get("side", "")).upper())


def _selected_map(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    obj = json.loads(Path(path).read_text())
    threshold = _threshold(obj)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _rows(obj):
        side = str(row.get("side", "")).upper()
        if side not in {"LONG", "SHORT"}:
            continue
        if float(row.get("score_mean", 1.0)) <= threshold:
            continue
        selected[_key(row)] = row
    return selected


def intersect_and_backtest(*, module_results: list[str], market_csv: str, output: str, hold_bars: int = 288, entry_delay_bars: int = 1, leverage: float = 0.5, fee_rate: float = 0.0004, slippage_rate: float = 0.0001) -> dict[str, Any]:
    if len(module_results) < 2:
        raise ValueError("need at least two module results")
    maps = [_selected_map(path) for path in module_results]
    keys = set(maps[0])
    for mp in maps[1:]:
        keys &= set(mp)
    candidates = []
    for key in sorted(keys):
        base = dict(maps[0][key])
        base["score_mean"] = 1.0
        base["intersection_modules"] = len(module_results)
        candidates.append(base)
    cfg = CandidateBacktestConfig(
        market_csv=market_csv,
        pairwise_jsonl="",
        predictions_jsonl="",
        output="",
        score_threshold=0.0,
        hold_bars=int(hold_bars),
        entry_delay_bars=int(entry_delay_bars),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
    )
    market = load_market(market_csv)
    result = simulate_candidates(candidates, market, cfg)
    result["as_of"] = datetime.now(timezone.utc).isoformat()
    result["module_results"] = module_results
    result["intersection_summary"] = {
        "input_selected_counts": [len(mp) for mp in maps],
        "intersection_count": len(candidates),
    }
    result["leakage_guard"] = {
        "module_results_must_be_precomputed_without_eval_selection": True,
        "intersection_uses_signal_date_and_side_only": True,
        "future_path_used_only_by_simulation_audit": True,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intersect selected candidates from module result files")
    p.add_argument("--module-result", action="append", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = intersect_and_backtest(
        module_results=args.module_result,
        market_csv=args.market_csv,
        output=args.output,
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    print(json.dumps({"intersection_summary": out["intersection_summary"], "sim": out["sim"], "trade_stats": out["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
