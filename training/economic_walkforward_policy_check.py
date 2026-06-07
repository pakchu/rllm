"""Walk-forward stability check for a fixed calibrated pressure-trader policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from training.economic_pressure_value_calibration import attach_context_cache, evaluate_config, fit_tables, load_jsonl
from training.strict_bar_backtest import load_market_bars


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def in_range(row: dict[str, Any], start: datetime | None, end: datetime | None) -> bool:
    d = parse_dt(row["date"])
    return (start is None or d >= start) and (end is None or d <= end)


def load_all_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(load_jsonl(p))
    rows.sort(key=lambda r: parse_dt(r["date"]))
    return attach_context_cache(rows)


def run_walkforward_check(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    output: str,
    prefix: str,
    horizon_bars: int,
    target_pct: float,
    stop_pct: float,
    level: str,
    min_n: int,
    min_score: float,
    score_mode: str,
    side_gate: str,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
) -> dict[str, Any]:
    all_rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    folds = [
        {"name": "2024_h1", "train_end": "2023-12-31 23:59:59", "test_start": "2024-01-01 00:00:00", "test_end": "2024-06-30 23:59:59"},
        {"name": "2024_h2_to_2025_feb", "train_end": "2024-06-30 23:59:59", "test_start": "2024-07-01 00:00:00", "test_end": "2025-02-28 23:59:59"},
        {"name": "2025_h1_val", "train_end": "2025-02-28 23:59:59", "test_start": "2025-03-01 00:00:00", "test_end": "2025-08-31 23:59:59"},
        {"name": "2025_h2_oos", "train_end": "2025-08-31 23:59:59", "test_start": "2025-09-01 00:00:00", "test_end": "2026-02-28 23:59:59"},
    ]
    results = []
    for fold in folds:
        train_end = parse_dt(fold["train_end"])
        test_start = parse_dt(fold["test_start"])
        test_end = parse_dt(fold["test_end"])
        train_rows = [r for r in all_rows if parse_dt(r["date"]) <= train_end]
        test_rows = [r for r in all_rows if in_range(r, test_start, test_end)]
        tables = fit_tables(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
        result = evaluate_config(
            test_rows,
            tables,
            market=market,
            market_csv=market_csv,
            prefix=prefix,
            split=fold["name"],
            level=level,
            min_n=min_n,
            min_score=min_score,
            score_mode=score_mode,
            side_gate=side_gate,
            horizon_bars=horizon_bars,
            target_pct=target_pct,
            stop_pct=stop_pct,
            leverage=leverage,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            entry_delay_bars=entry_delay_bars,
            write_artifacts=False,
        )
        results.append({"fold": fold, "train_rows": len(train_rows), "test_rows": len(test_rows), "table_count": len(tables), "result": result})
    positives = sum(1 for r in results if r["result"]["sim"]["cagr_to_strict_mdd"] > 0)
    strong = sum(1 for r in results if r["result"]["sim"]["cagr_to_strict_mdd"] >= 3.0 and r["result"]["sim"]["trade_entries"] >= 50)
    report = {
        "fixed_policy": {"horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct, "level": level, "min_n": min_n, "min_score": min_score, "score_mode": score_mode, "side_gate": side_gate},
        "folds": results,
        "summary": {"positive_ratio_folds": positives, "strong_ratio_folds": strong, "fold_count": len(results)},
        "leakage_guard": {"each_fold_fits_only_rows_at_or_before_train_end": True, "test_rows_are_after_train_end": True, "fixed_policy_not_reselected_per_fold": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-paths", required=True, help="Comma-separated SFT/prediction JSONL paths to combine by date")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--horizon-bars", type=int, default=144)
    p.add_argument("--target-pct", type=float, default=1.8)
    p.add_argument("--stop-pct", type=float, default=1.5)
    p.add_argument("--level", default="teacher_only")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.0005)
    p.add_argument("--score-mode", default="mean")
    p.add_argument("--side-gate", default="free")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_walkforward_check(jsonl_paths=[x for x in args.jsonl_paths.split(",") if x], market_csv=args.market_csv, output=args.output, prefix=args.prefix, horizon_bars=args.horizon_bars, target_pct=args.target_pct, stop_pct=args.stop_pct, level=args.level, min_n=args.min_n, min_score=args.min_score, score_mode=args.score_mode, side_gate=args.side_gate)
    print(json.dumps(report["summary"], indent=2))
    for fold in report["folds"]:
        sim = fold["result"]["sim"]
        print(fold["fold"]["name"], "trades", sim["trade_entries"], "cagr", round(sim["cagr_pct"], 2), "mdd", round(sim["strict_mdd_pct"], 2), "ratio", round(sim["cagr_to_strict_mdd"], 3))


if __name__ == "__main__":
    main()
