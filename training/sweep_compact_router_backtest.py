"""Fast val-selection/OOS evaluation sweep for compact router backtests."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.compact_router_backtest import CompactRouterExecutionConfig, load_jsonl, simulate_compact_router_records
from training.strict_bar_backtest import load_market_bars


def _csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _csv_strs(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _score_candidate(sim: dict[str, Any], *, min_trades: int, max_mdd: float) -> tuple[Any, ...]:
    s = sim["sim"]
    t = sim["trade_stats"]
    enough = int(s["trade_entries"] >= int(min_trades))
    mdd_ok = int(float(s["strict_mdd_pct"]) <= float(max_mdd))
    positive_ci = int(float(t["ci95_mean_trade_ret_pct"][0]) > 0.0)
    return (
        enough,
        mdd_ok,
        positive_ci,
        float(s["cagr_to_strict_mdd"]),
        float(s["cagr_pct"]),
        -float(s["strict_mdd_pct"]),
        int(s["trade_entries"]),
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    val_records = load_jsonl(args.val_records)
    oos_records = load_jsonl(args.oos_records)
    market = load_market_bars(args.market_csv)
    rows: list[dict[str, Any]] = []
    for mode, edge, cool, short, mid, long in itertools.product(
        _csv_strs(args.routing_modes),
        _csv_strs(args.min_edge_qualities),
        _csv_ints(args.cooldown_bars_list),
        _csv_ints(args.short_hold_bars_list),
        _csv_ints(args.mid_hold_bars_list),
        _csv_ints(args.long_hold_bars_list),
    ):
        cfg = CompactRouterExecutionConfig(
            cooldown_bars=int(cool),
            entry_delay_bars=int(args.entry_delay_bars),
            leverage=float(args.leverage),
            fee_rate=float(args.fee_rate),
            slippage_rate=float(args.slippage_rate),
            short_hold_bars=int(short),
            mid_hold_bars=int(mid),
            long_hold_bars=int(long),
            min_edge_quality=str(edge),
            routing_mode=str(mode),
            use_target=False,
        )
        val = simulate_compact_router_records(val_records, market, cfg)
        rows.append({"config": asdict(cfg), "val": val, "selection_score": _score_candidate(val, min_trades=args.min_trades, max_mdd=args.max_mdd)})
    rows.sort(key=lambda r: r["selection_score"], reverse=True)
    selected_cfg = CompactRouterExecutionConfig(**rows[0]["config"])
    oos = simulate_compact_router_records(oos_records, market, selected_cfg)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "val_records": args.val_records,
        "oos_records": args.oos_records,
        "market_csv": args.market_csv,
        "selection_rule": {
            "selected_on": "val_only",
            "sort": "min_trades_pass, max_mdd_pass, positive_ci_pass, cagr_to_strict_mdd, cagr, -mdd, trades",
            "min_trades": int(args.min_trades),
            "max_mdd": float(args.max_mdd),
        },
        "selected_config": rows[0]["config"],
        "selected_val": rows[0]["val"],
        "selected_oos": oos,
        "top_val": [{"rank": i + 1, "config": r["config"], "val": r["val"]} for i, r in enumerate(rows[: int(args.top_k)])],
        "num_candidates": len(rows),
        "leakage_guard": {
            "selected_on_val_only": True,
            "oos_not_used_for_selection": True,
            "records_are_model_predictions": True,
            "strict_bar_by_bar": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast compact router backtest sweep with val selection and OOS report")
    p.add_argument("--val-records", required=True)
    p.add_argument("--oos-records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/compact_router_sweep.json")
    p.add_argument("--routing-modes", default="learned_fields,action_path")
    p.add_argument("--min-edge-qualities", default="MODERATE,STRONG")
    p.add_argument("--cooldown-bars-list", default="0,12,36")
    p.add_argument("--short-hold-bars-list", default="36,72")
    p.add_argument("--mid-hold-bars-list", default="72,144")
    p.add_argument("--long-hold-bars-list", default="144,288,432")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--max-mdd", type=float, default=25.0)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    report = run_sweep(parse_args())
    print(json.dumps({"selected_config": report["selected_config"], "val": report["selected_val"]["sim"], "oos": report["selected_oos"]["sim"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
