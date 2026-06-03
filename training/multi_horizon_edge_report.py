"""Multi-horizon edge viability report for analyzer/trader redesign.

The current 432-bar decision target is not learnable from the coarse analyzer
summary.  Before changing the LLM target again, this diagnostic asks whether the
past trend side has stable TREND/FADE edge at any shorter horizon.

This is an offline label/target design report, not a deployable strategy.  It
uses past-only trend side for action orientation and future OHLC paths only for
split-wise outcome summaries and oracle upper bounds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.analyzer_state_edge_report import _opposite, _stats_to_json, _trend_side, summarize_values
from training.decision_feature_learnability import load_jsonl
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome
from training.strict_bar_backtest import load_market_bars

ACTIONS = ("TREND", "FADE", "ORACLE", "SKIP")


def parse_horizons(raw: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("at least one hold horizon is required")
    return vals


def _path_for_action(market, row: dict[str, Any], action: str, cfg: PathOutcomeConfig):
    trend_side = _trend_side(row)
    if trend_side not in {"LONG", "SHORT"}:
        return None
    if action == "TREND":
        side = trend_side
    elif action == "FADE":
        side = _opposite(trend_side)
    else:
        return None
    if side not in {"LONG", "SHORT"}:
        return None
    return compute_trade_path_outcome(market, int(row.get("signal_pos", 0)), side, cfg)


def summarize_action(rows: list[dict[str, Any]], market, hold_bars: int, action: str, cfg_base: dict[str, Any]) -> dict[str, Any]:
    cfg = PathOutcomeConfig(hold_bars=int(hold_bars), **cfg_base)
    returns: list[float] = []
    maes: list[float] = []
    action_counts: Counter[str] = Counter({"TREND": 0, "FADE": 0, "ORACLE": 0, "SKIP": 0})
    for row in rows:
        if action in {"TREND", "FADE"}:
            out = _path_for_action(market, row, action, cfg)
            if out is None:
                action_counts["SKIP"] += 1
                continue
            action_counts[action] += 1
            returns.append(float(out.net_return))
            maes.append(float(out.mae))
        elif action == "ORACLE":
            trend = _path_for_action(market, row, "TREND", cfg)
            fade = _path_for_action(market, row, "FADE", cfg)
            candidates = [x for x in [trend, fade] if x is not None]
            if not candidates:
                action_counts["SKIP"] += 1
                continue
            best = max(candidates, key=lambda x: float(x.net_return))
            if float(best.net_return) <= 0.0:
                action_counts["SKIP"] += 1
                continue
            best_action = "TREND" if trend is not None and best.side == trend.side else "FADE"
            action_counts[best_action] += 1
            action_counts["ORACLE"] += 1
            returns.append(float(best.net_return))
            maes.append(float(best.mae))
        else:
            action_counts["SKIP"] += 1
    stats = summarize_values(returns, maes)
    return {
        "samples": len(rows),
        "trades": stats.n,
        "action_counts": dict(action_counts),
        "trade_stats": _stats_to_json(stats),
    }


def run_report(
    splits: dict[str, list[dict[str, Any]]],
    market,
    horizons: tuple[int, ...],
    cfg_base: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon in horizons:
        hkey = str(int(horizon))
        out[hkey] = {}
        for split, rows in splits.items():
            out[hkey][split] = {action: summarize_action(rows, market, horizon, action, cfg_base) for action in ("TREND", "FADE", "ORACLE")}
    return out


def best_horizon_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for horizon, by_split in report.items():
        row: dict[str, Any] = {"hold_bars": int(horizon)}
        for split in ("train", "val", "oos"):
            if split not in by_split:
                continue
            best_action = None
            best_mean = float("-inf")
            for action in ("TREND", "FADE"):
                mean = by_split[split][action]["trade_stats"]["mean_return_pct"]
                if mean > best_mean:
                    best_mean = mean
                    best_action = action
            row[f"{split}_best_static_action"] = best_action
            row[f"{split}_best_static_mean_return_pct"] = best_mean
            row[f"{split}_oracle_mean_return_pct"] = by_split[split]["ORACLE"]["trade_stats"]["mean_return_pct"]
            row[f"{split}_oracle_trades"] = by_split[split]["ORACLE"]["trades"]
        rows.append(row)
    return sorted(rows, key=lambda r: int(r["hold_bars"]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-horizon TREND/FADE/ORACLE edge viability report")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", default="")
    p.add_argument("--output", default="results/multi_horizon_edge_report.json")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    market = load_market_bars(args.market_csv)
    splits = {"train": load_jsonl(args.train_jsonl), "val": load_jsonl(args.val_jsonl)}
    if args.oos_jsonl:
        splits["oos"] = load_jsonl(args.oos_jsonl)
    cfg_base = {
        "entry_delay_bars": int(args.entry_delay_bars),
        "fee_rate": float(args.fee_rate),
        "slippage_rate": float(args.slippage_rate),
        "leverage": float(args.leverage),
    }
    horizons = parse_horizons(args.hold_bars_list)
    by_horizon = run_report(splits, market, horizons, cfg_base)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "market_csv": args.market_csv,
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "oos_jsonl": args.oos_jsonl,
        },
        "config": {"hold_bars_list": list(horizons), **cfg_base},
        "summary_table": best_horizon_table(by_horizon),
        "by_horizon": by_horizon,
        "leakage_guard": {
            "action_orientation_uses_past_trend_side_only": True,
            "future_ohlc_used_for_offline_target_design_only": True,
            "oracle_is_upper_bound_not_deployable": True,
            "no_eval_parameter_selection_performed": True,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
