"""Leakage-safe sweep over target/stop/horizon economic action spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.economic_pressure_value_calibration import KEY_LEVELS, attach_context_cache, evaluate_config, fit_tables, load_jsonl
from training.strict_bar_backtest import load_market_bars


def parse_floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def parse_ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def run_action_space_sweep(
    *,
    train_jsonl: str,
    val_predictions_jsonl: str,
    oos_predictions_jsonl: str,
    market_csv: str,
    output: str,
    prefix: str,
    horizons: list[int],
    targets: list[float],
    stops: list[float],
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    min_trades: int = 50,
) -> dict[str, Any]:
    market = load_market_bars(market_csv)
    train_rows = attach_context_cache(load_jsonl(train_jsonl))
    val_rows = attach_context_cache(load_jsonl(val_predictions_jsonl))
    oos_rows = attach_context_cache(load_jsonl(oos_predictions_jsonl))
    configs = []
    for level in ["teacher_only", "coarse", "risk", "macro"]:
        assert level in KEY_LEVELS
        for min_n in [20, 35, 50, 80]:
            for min_score in [0.0, 0.0005, 0.0010, 0.0015, 0.0020]:
                for score_mode in ["mean", "lower95"]:
                    for side_gate in ["free", "teacher", "model"]:
                        configs.append((level, min_n, min_score, score_mode, side_gate))
    combo_reports = []
    for horizon in horizons:
        for target in targets:
            for stop in stops:
                tables = fit_tables(train_rows, market, horizon_bars=horizon, target_pct=target, stop_pct=stop, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
                val_results = []
                for level, min_n, min_score, score_mode, side_gate in configs:
                    val_results.append(evaluate_config(
                        val_rows,
                        tables,
                        market=market,
                        market_csv=market_csv,
                        prefix=prefix,
                        split="val_tmp",
                        level=level,
                        min_n=min_n,
                        min_score=min_score,
                        score_mode=score_mode,
                        side_gate=side_gate,
                        horizon_bars=horizon,
                        target_pct=target,
                        stop_pct=stop,
                        leverage=leverage,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        entry_delay_bars=entry_delay_bars,
                    ))
                eligible = [r for r in val_results if r["sim"]["trade_entries"] >= min_trades]
                ranked = sorted(eligible or val_results, key=lambda r: (r["sim"]["cagr_to_strict_mdd"], r["sim"]["cagr_pct"], r["sim"]["trade_entries"]), reverse=True)
                best = ranked[0]
                combo_reports.append({"economics": {"horizon_bars": horizon, "target_pct": target, "stop_pct": stop}, "best_val": best, "table_count": len(tables)})
    ranked_combos = sorted(combo_reports, key=lambda r: (r["best_val"]["sim"]["cagr_to_strict_mdd"], r["best_val"]["sim"]["cagr_pct"], r["best_val"]["sim"]["trade_entries"]), reverse=True)
    selected_combo = ranked_combos[0]
    econ = selected_combo["economics"]
    cfg = selected_combo["best_val"]["config"]
    tables = fit_tables(train_rows, market, horizon_bars=econ["horizon_bars"], target_pct=econ["target_pct"], stop_pct=econ["stop_pct"], leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars)
    selected_val = evaluate_config(val_rows, tables, market=market, market_csv=market_csv, prefix=prefix, split="selected_val", horizon_bars=econ["horizon_bars"], target_pct=econ["target_pct"], stop_pct=econ["stop_pct"], leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, write_artifacts=True, **cfg)
    selected_oos = evaluate_config(oos_rows, tables, market=market, market_csv=market_csv, prefix=prefix, split="oos", horizon_bars=econ["horizon_bars"], target_pct=econ["target_pct"], stop_pct=econ["stop_pct"], leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, write_artifacts=True, **cfg)
    report = {
        "selection_rule": f"fit train per economic combo; rank val configs with >= {min_trades} trades; evaluate selected combo/config once on OOS",
        "selected_economics": econ,
        "selected_config": cfg,
        "selected_val": selected_val,
        "selected_oos": selected_oos,
        "top_combos": ranked_combos[:30],
        "sweep_space": {"horizons": horizons, "targets": targets, "stops": stops, "config_count_per_combo": len(configs)},
        "leakage_guard": {"fit_split": "train only", "selection_split": "validation only", "oos_used_for_selection": False, "strict_costs_and_mdd": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-predictions-jsonl", required=True)
    p.add_argument("--oos-predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--horizons", default="36,72,144")
    p.add_argument("--targets", default="0.8,1.2,1.8,2.5")
    p.add_argument("--stops", default="0.6,1.0,1.5")
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--min-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_action_space_sweep(
        train_jsonl=args.train_jsonl,
        val_predictions_jsonl=args.val_predictions_jsonl,
        oos_predictions_jsonl=args.oos_predictions_jsonl,
        market_csv=args.market_csv,
        output=args.output,
        prefix=args.prefix,
        horizons=parse_ints(args.horizons),
        targets=parse_floats(args.targets),
        stops=parse_floats(args.stops),
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        entry_delay_bars=args.entry_delay_bars,
        min_trades=args.min_trades,
    )
    print(json.dumps({"selected_economics": report["selected_economics"], "selected_config": report["selected_config"], "selected_val": report["selected_val"], "selected_oos": report["selected_oos"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
