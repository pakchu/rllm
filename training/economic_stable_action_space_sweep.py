"""Sweep economic action templates by multi-fold stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.economic_fold_stability_sweep import aggregate_score, default_folds, load_all_rows, score_fold_result, split_rows
from training.economic_pressure_value_calibration import KEY_LEVELS, evaluate_config, fit_tables
from training.strict_bar_backtest import load_market_bars


def parse_floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def parse_ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def config_grid(*, quick: bool = False) -> list[dict[str, Any]]:
    configs = []
    levels = ["teacher_only"] if quick else ["teacher_only", "coarse", "risk", "macro"]
    min_ns = [20, 35, 50] if quick else [10, 20, 35, 50, 80]
    min_scores = [0.0002, 0.0005, 0.0010] if quick else [0.0, 0.0002, 0.0005, 0.0010, 0.0015]
    score_modes = ["mean"] if quick else ["mean", "lower95"]
    side_gates = ["free"] if quick else ["free", "teacher", "model"]
    for level in levels:
        assert level in KEY_LEVELS
        for min_n in min_ns:
            for min_score in min_scores:
                for score_mode in score_modes:
                    for side_gate in side_gates:
                        configs.append({"level": level, "min_n": min_n, "min_score": min_score, "score_mode": score_mode, "side_gate": side_gate})
    return configs


def evaluate_economics(
    *,
    rows: list[dict[str, Any]],
    market,
    market_csv: str,
    prefix: str,
    horizon_bars: int,
    target_pct: float,
    stop_pct: float,
    configs: list[dict[str, Any]],
    min_trades: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    entry_delay_bars: int,
) -> dict[str, Any]:
    folds = default_folds()
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
        summaries.append({"config": cfg, "aggregate": aggregate_score(fold_scores, min_trades=min_trades), "folds": fold_results})
    ranked = sorted(summaries, key=lambda x: (x["aggregate"]["stability_score"], x["aggregate"]["positive_folds"], x["aggregate"]["min_ratio"], x["aggregate"]["avg_ratio"]), reverse=True)
    return {"economics": {"horizon_bars": horizon_bars, "target_pct": target_pct, "stop_pct": stop_pct}, "best": ranked[0], "top_configs": ranked[:5]}


def run_stable_action_space_sweep(
    *,
    jsonl_paths: list[str],
    market_csv: str,
    output: str,
    prefix: str,
    horizons: list[int],
    targets: list[float],
    stops: list[float],
    min_trades: int = 50,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    quick: bool = False,
) -> dict[str, Any]:
    rows = load_all_rows(jsonl_paths)
    market = load_market_bars(market_csv)
    configs = config_grid(quick=quick)
    combo_reports = []
    for horizon in horizons:
        for target in targets:
            for stop in stops:
                combo_reports.append(evaluate_economics(rows=rows, market=market, market_csv=market_csv, prefix=prefix, horizon_bars=horizon, target_pct=target, stop_pct=stop, configs=configs, min_trades=min_trades, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars))
    ranked = sorted(combo_reports, key=lambda x: (x["best"]["aggregate"]["stability_score"], x["best"]["aggregate"]["positive_folds"], x["best"]["aggregate"]["min_ratio"], x["best"]["aggregate"]["avg_ratio"]), reverse=True)
    report = {
        "selection_rule": "rank economic templates and configs by four-fold stability, penalizing low/no-trade folds",
        "selected": ranked[0],
        "top": ranked[:20],
        "sweep_space": {"horizons": horizons, "targets": targets, "stops": stops, "config_count_per_combo": len(configs), "min_trades_per_fold": min_trades, "quick": quick},
        "leakage_guard": {"each_fold_fits_only_prior_rows": True, "no_single_val_or_oos_selection": True},
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
    p.add_argument("--horizons", default="36,72,144,288")
    p.add_argument("--targets", default="0.8,1.2,1.8,2.5,3.5")
    p.add_argument("--stops", default="0.6,1.0,1.5,2.0")
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_stable_action_space_sweep(
        jsonl_paths=[x for x in args.jsonl_paths.split(",") if x],
        market_csv=args.market_csv,
        output=args.output,
        prefix=args.prefix,
        horizons=parse_ints(args.horizons),
        targets=parse_floats(args.targets),
        stops=parse_floats(args.stops),
        min_trades=args.min_trades,
        quick=args.quick,
    )
    sel = report["selected"]
    print(json.dumps({"selected_economics": sel["economics"], "selected_config": sel["best"]["config"], "aggregate": sel["best"]["aggregate"]}, indent=2, ensure_ascii=False))
    for f in sel["best"]["folds"]:
        sim = f["result"]["sim"]
        print(f["fold"]["name"], "trades", sim["trade_entries"], "cagr", round(sim["cagr_pct"], 2), "mdd", round(sim["strict_mdd_pct"], 2), "ratio", round(sim["cagr_to_strict_mdd"], 3))


if __name__ == "__main__":
    main()
