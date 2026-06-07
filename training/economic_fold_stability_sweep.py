"""Select calibrated trader configs by multi-fold stability, not one validation period."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from training.economic_pressure_value_calibration import KEY_LEVELS, attach_context_cache, evaluate_config, fit_tables, load_jsonl
from training.strict_bar_backtest import load_market_bars


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def load_all_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(load_jsonl(p))
    rows.sort(key=lambda r: parse_dt(r["date"]))
    return attach_context_cache(rows)


def split_rows(rows: list[dict[str, Any]], *, train_end: str, test_start: str, test_end: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tr_end = parse_dt(train_end)
    te_start = parse_dt(test_start)
    te_end = parse_dt(test_end)
    train = [r for r in rows if parse_dt(r["date"]) <= tr_end]
    test = [r for r in rows if te_start <= parse_dt(r["date"]) <= te_end]
    return train, test


def default_folds() -> list[dict[str, str]]:
    return [
        {"name": "2024_h1", "train_end": "2023-12-31 23:59:59", "test_start": "2024-01-01 00:00:00", "test_end": "2024-06-30 23:59:59"},
        {"name": "2024_h2_to_2025_feb", "train_end": "2024-06-30 23:59:59", "test_start": "2024-07-01 00:00:00", "test_end": "2025-02-28 23:59:59"},
        {"name": "2025_h1_val", "train_end": "2025-02-28 23:59:59", "test_start": "2025-03-01 00:00:00", "test_end": "2025-08-31 23:59:59"},
        {"name": "2025_h2_oos", "train_end": "2025-08-31 23:59:59", "test_start": "2025-09-01 00:00:00", "test_end": "2026-02-28 23:59:59"},
    ]


def score_fold_result(result: dict[str, Any], *, min_trades: int) -> dict[str, float | int]:
    sim = result["sim"]
    stats = result["trade_stats"]
    trades = int(sim["trade_entries"])
    ratio = float(sim["cagr_to_strict_mdd"])
    if not (ratio < 1e100 and ratio > -1e100):
        ratio = 0.0
    cagr = float(sim["cagr_pct"])
    mean_ret = float(stats.get("mean_trade_ret_pct", 0.0))
    ok_trades = trades >= min_trades
    positive = ok_trades and cagr > 0.0 and ratio > 0.0 and mean_ret > 0.0
    strong = ok_trades and ratio >= 3.0 and cagr > 0.0 and mean_ret > 0.0
    return {"trades": trades, "ratio": ratio, "cagr": cagr, "mean_trade_ret_pct": mean_ret, "positive": int(positive), "strong": int(strong)}


def aggregate_score(fold_scores: list[dict[str, float | int]], *, min_trades: int = 50) -> dict[str, float | int]:
    ratios = [float(s["ratio"]) if int(s["trades"]) > 0 else -10.0 for s in fold_scores]
    cagrs = [float(s["cagr"]) for s in fold_scores]
    trades = [int(s["trades"]) for s in fold_scores]
    positives = sum(int(s["positive"]) for s in fold_scores)
    strong = sum(int(s["strong"]) for s in fold_scores)
    min_ratio = min(ratios) if ratios else -999.0
    avg_ratio = sum(ratios) / len(ratios) if ratios else -999.0
    avg_cagr = sum(cagrs) / len(cagrs) if cagrs else -999.0
    min_trades_seen = min(trades) if trades else 0
    adequate_folds = sum(1 for t in trades if t >= min_trades)
    # Conservative objective: first maximize number of positive folds, then coverage,
    # then worst fold and average ratio/CAGR. No-trade folds are not allowed to win via inf ratio.
    low_coverage_penalty = sum(1 for t in trades if t <= 0) * 200.0 + sum(1 for t in trades if 0 < t < min_trades) * 25.0
    stability_score = positives * 100.0 + strong * 25.0 + min_ratio * 5.0 + avg_ratio + avg_cagr * 0.1 - low_coverage_penalty
    return {"positive_folds": positives, "strong_folds": strong, "min_ratio": min_ratio, "avg_ratio": avg_ratio, "avg_cagr": avg_cagr, "min_trades": min_trades_seen, "adequate_folds": adequate_folds, "stability_score": stability_score}


def run_fold_stability_sweep(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    output: str,
    prefix: str,
    horizon_bars: int = 144,
    target_pct: float = 1.8,
    stop_pct: float = 1.5,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    min_trades: int = 50,
) -> dict[str, Any]:
    rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    folds = default_folds()
    configs = []
    for level in ["teacher_only", "coarse", "risk", "macro", "state", "micro"]:
        assert level in KEY_LEVELS
        for min_n in [10, 20, 35, 50, 80, 120]:
            for min_score in [-0.0005, 0.0, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020]:
                for score_mode in ["mean", "lower95"]:
                    for side_gate in ["free", "teacher", "model"]:
                        configs.append({"level": level, "min_n": min_n, "min_score": min_score, "score_mode": score_mode, "side_gate": side_gate})

    # Pre-fit tables once per fold, then evaluate every config on that fold.
    fold_material = []
    for fold in folds:
        train_rows, test_rows = split_rows(rows, train_end=fold["train_end"], test_start=fold["test_start"], test_end=fold["test_end"])
        tables = fit_tables(train_rows, market, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
        fold_material.append({"fold": fold, "train_rows": len(train_rows), "test_rows": len(test_rows), "test": test_rows, "tables": tables})

    summaries = []
    for cfg in configs:
        fold_results = []
        fold_scores = []
        for material in fold_material:
            result = evaluate_config(
                material["test"],
                material["tables"],
                market=market,
                market_csv=market_csv,
                prefix=prefix,
                split="tmp",
                horizon_bars=horizon_bars,
                target_pct=target_pct,
                stop_pct=stop_pct,
                leverage=leverage,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                entry_delay_bars=entry_delay_bars,
                write_artifacts=False,
                **cfg,
            )
            fs = score_fold_result(result, min_trades=min_trades)
            fold_scores.append(fs)
            fold_results.append({"fold": material["fold"], "train_rows": material["train_rows"], "test_rows": material["test_rows"], "result": result, "score": fs})
        agg = aggregate_score(fold_scores, min_trades=min_trades)
        summaries.append({"config": cfg, "aggregate": agg, "folds": fold_results})

    ranked = sorted(summaries, key=lambda x: (x["aggregate"]["stability_score"], x["aggregate"]["positive_folds"], x["aggregate"]["min_ratio"], x["aggregate"]["avg_ratio"]), reverse=True)
    selected = ranked[0]
    report = {
        "fixed_economics": {"horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct},
        "selection_rule": "rank configs by positive fold count, strong fold count, worst-fold ratio, average ratio/CAGR",
        "selected": selected,
        "top": ranked[:25],
        "sweep_space": {"config_count": len(configs), "fold_count": len(folds), "min_trades_per_fold": min_trades},
        "leakage_guard": {"each_fold_fits_only_prior_rows": True, "no_single_eval_fold_selection": True, "future_rows_not_used_in_fold_tables": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl-paths", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--horizon-bars", type=int, default=144)
    p.add_argument("--target-pct", type=float, default=1.8)
    p.add_argument("--stop-pct", type=float, default=1.5)
    p.add_argument("--min-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fold_stability_sweep(
        jsonl_paths=[x for x in args.jsonl_paths.split(",") if x],
        market_csv=args.market_csv,
        output=args.output,
        prefix=args.prefix,
        horizon_bars=args.horizon_bars,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        min_trades=args.min_trades,
    )
    sel = report["selected"]
    print(json.dumps({"fixed_economics": report["fixed_economics"], "selected_config": sel["config"], "aggregate": sel["aggregate"]}, indent=2, ensure_ascii=False))
    for f in sel["folds"]:
        sim = f["result"]["sim"]
        print(f["fold"]["name"], "trades", sim["trade_entries"], "cagr", round(sim["cagr_pct"], 2), "mdd", round(sim["strict_mdd_pct"], 2), "ratio", round(sim["cagr_to_strict_mdd"], 3))


if __name__ == "__main__":
    main()
