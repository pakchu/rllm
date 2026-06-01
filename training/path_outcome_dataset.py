"""Build executable path-outcome labels for LLM trading training.

The previous LLM labels optimized a scalar forward return.  This module builds
labels from the actual OHLC path that an order would experience: delayed entry,
scheduled exit, fees/slippage, maximum adverse excursion (MAE), and maximum
favorable excursion (MFE).  The resulting records are intended to train an
analyzer/trader pair on executable trade outcomes rather than numeric
next-return shortcuts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from training.strict_bar_backtest import load_market_bars

Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class PathOutcomeConfig:
    hold_bars: int = 144
    entry_delay_bars: int = 1
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    leverage: float = 1.0
    mae_penalty: float = 1.0
    mfe_bonus: float = 0.0
    hold_margin: float = 0.0
    min_net_return: float = 0.0
    max_mae: float = 1.0


@dataclass(frozen=True)
class TradePathOutcome:
    side: Side
    entry_pos: int
    exit_pos: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    mae: float
    mfe: float
    utility: float


def compute_trade_path_outcome(
    market: pd.DataFrame,
    signal_pos: int,
    side: Side,
    cfg: PathOutcomeConfig,
) -> TradePathOutcome | None:
    """Compute executable outcome for one side from a signal timestamp.

    A signal at bar ``signal_pos`` is filled at ``signal_pos + entry_delay``
    open and closed at ``entry + hold_bars`` open.  MAE/MFE use highs/lows from
    every held bar and are measured from entry price, not from future labels.
    """

    hold_bars = max(1, int(cfg.hold_bars))
    entry_pos = int(signal_pos) + max(0, int(cfg.entry_delay_bars))
    exit_pos = entry_pos + hold_bars
    if entry_pos < 0 or exit_pos >= len(market):
        return None
    entry_price = float(market.iloc[entry_pos]["open"])
    exit_price = float(market.iloc[exit_pos]["open"])
    if entry_price <= 0.0 or exit_price <= 0.0:
        return None

    held = market.iloc[entry_pos:exit_pos]
    if len(held) == 0:
        return None
    max_high = float(held["high"].max())
    min_low = float(held["low"].min())

    if side == "LONG":
        gross_return = (exit_price - entry_price) / entry_price
        mae = max(0.0, (entry_price - min_low) / entry_price)
        mfe = max(0.0, (max_high - entry_price) / entry_price)
    elif side == "SHORT":
        gross_return = (entry_price - exit_price) / entry_price
        mae = max(0.0, (max_high - entry_price) / entry_price)
        mfe = max(0.0, (entry_price - min_low) / entry_price)
    else:
        raise ValueError(f"unsupported side: {side}")

    lev = max(0.0, float(cfg.leverage))
    round_trip_cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate)) * lev
    net_return = lev * gross_return - round_trip_cost
    utility = net_return - lev * float(cfg.mae_penalty) * mae + lev * float(cfg.mfe_bonus) * mfe
    return TradePathOutcome(
        side=side,
        entry_pos=entry_pos,
        exit_pos=exit_pos,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=float(gross_return),
        net_return=float(net_return),
        mae=float(mae),
        mfe=float(mfe),
        utility=float(utility),
    )


def _bucket_pct(value: float, *, good: float, strong: float) -> str:
    v = float(value)
    if v >= strong:
        return "strong"
    if v >= good:
        return "positive"
    if v > 0.0:
        return "weak"
    return "negative"


def _risk_bucket(mae: float) -> str:
    m = float(mae)
    if m <= 0.005:
        return "low"
    if m <= 0.015:
        return "medium"
    if m <= 0.03:
        return "high"
    return "extreme"


def make_path_outcome_record(market: pd.DataFrame, signal_pos: int, cfg: PathOutcomeConfig) -> dict[str, Any] | None:
    """Build one analyzer/trader label record from executable path outcomes."""

    long = compute_trade_path_outcome(market, signal_pos, "LONG", cfg)
    short = compute_trade_path_outcome(market, signal_pos, "SHORT", cfg)
    if long is None or short is None:
        return None
    best = long if long.utility >= short.utility else short
    directional_best_utility = float(best.utility)
    gate_label = "TRADE"
    if directional_best_utility <= float(cfg.hold_margin):
        gate_label = "NO_TRADE"
    if float(best.net_return) <= float(cfg.min_net_return):
        gate_label = "NO_TRADE"
    if float(best.mae) > float(cfg.max_mae):
        gate_label = "NO_TRADE"

    row = market.iloc[signal_pos]
    date = str(pd.to_datetime(row["date"]))
    quality_label = _bucket_pct(best.net_return, good=float(cfg.min_net_return), strong=max(float(cfg.min_net_return) * 3.0, 0.003))
    risk_label = _risk_bucket(best.mae)
    outcome_summary = (
        f"{gate_label} {best.side}: net={best.net_return * 100:.3f}%, "
        f"mae={best.mae * 100:.3f}%, mfe={best.mfe * 100:.3f}%, "
        f"utility={best.utility * 100:.3f}%"
    )
    return {
        "date": date,
        "signal_pos": int(signal_pos),
        "entry_delay_bars": int(cfg.entry_delay_bars),
        "hold_bars": int(cfg.hold_bars),
        "trade_gate_label": gate_label,
        "trade_side_label": best.side,
        "best_side": best.side,
        "quality_label": quality_label,
        "risk_label": risk_label,
        "outcome_summary": outcome_summary,
        "best_utility": directional_best_utility,
        "best_net_return": float(best.net_return),
        "best_mae": float(best.mae),
        "best_mfe": float(best.mfe),
        "long": asdict(long),
        "short": asdict(short),
        "config": asdict(cfg),
    }


def build_path_outcome_records(
    market: pd.DataFrame,
    cfg: PathOutcomeConfig,
    *,
    window_size: int = 96,
    start_date: str | None = None,
    end_date: str | None = None,
    stride_bars: int = 1,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Build chronological path outcome records without using future data in inputs.

    The output labels necessarily use future OHLC paths.  Callers must use only
    past bars up to ``signal_pos`` when constructing prompts/features.
    """

    stride = max(1, int(stride_bars))
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    last_signal_pos = len(market) - max(1, int(cfg.entry_delay_bars)) - max(1, int(cfg.hold_bars)) - 1
    records: list[dict[str, Any]] = []
    for pos in range(max(0, int(window_size) - 1), max(0, last_signal_pos) + 1, stride):
        ts = pd.to_datetime(market.iloc[pos]["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        record = make_path_outcome_record(market, pos, cfg)
        if record is None:
            continue
        records.append(record)
        if max_records is not None and len(records) >= int(max_records):
            break
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}
    gate_counts: dict[str, int] = {}
    side_counts: dict[str, int] = {}
    net_returns = []
    maes = []
    utilities = []
    for rec in records:
        gate_counts[str(rec["trade_gate_label"])] = gate_counts.get(str(rec["trade_gate_label"]), 0) + 1
        side_counts[str(rec["trade_side_label"])] = side_counts.get(str(rec["trade_side_label"]), 0) + 1
        net_returns.append(float(rec["best_net_return"]))
        maes.append(float(rec["best_mae"]))
        utilities.append(float(rec["best_utility"]))
    return {
        "num_records": len(records),
        "period": {"start": records[0]["date"], "end": records[-1]["date"]},
        "gate_counts": gate_counts,
        "side_counts": side_counts,
        "mean_best_net_return_pct": 100.0 * sum(net_returns) / len(net_returns),
        "mean_best_mae_pct": 100.0 * sum(maes) / len(maes),
        "mean_best_utility_pct": 100.0 * sum(utilities) / len(utilities),
        "positive_best_net_return_rate": sum(1 for x in net_returns if x > 0.0) / len(net_returns),
    }


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build executable path-outcome labels for analyzer/trader training")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="data/path_outcomes.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--window-size", type=int, default=96)
    p.add_argument("--hold-bars", type=int, default=144)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--mfe-bonus", type=float, default=0.0)
    p.add_argument("--hold-margin", type=float, default=0.0)
    p.add_argument("--min-net-return", type=float, default=0.0)
    p.add_argument("--max-mae", type=float, default=1.0)
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--stride-bars", type=int, default=1)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PathOutcomeConfig(
        hold_bars=args.hold_bars,
        entry_delay_bars=args.entry_delay_bars,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        leverage=args.leverage,
        mae_penalty=args.mae_penalty,
        mfe_bonus=args.mfe_bonus,
        hold_margin=args.hold_margin,
        min_net_return=args.min_net_return,
        max_mae=args.max_mae,
    )
    market = load_market_bars(args.market_csv)
    records = build_path_outcome_records(
        market,
        cfg,
        window_size=args.window_size,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        stride_bars=args.stride_bars,
        max_records=args.max_records or None,
    )
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "market_csv": args.market_csv,
        "output": args.output,
        "config": asdict(cfg),
        "leakage_guard": {
            "labels_use_future_path": True,
            "prompt_features_must_use_bars_at_or_before_signal_pos": True,
            "entry_after_signal_by_bars": int(args.entry_delay_bars),
            "next_return_scalar_not_used": True,
        },
        "summary": summarize_records(records),
    }
    summary_output = args.summary_output or f"{args.output}.summary.json"
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
