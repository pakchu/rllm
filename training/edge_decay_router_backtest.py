"""Strict diagnostic backtest for edge-decay analyzer router labels.

This module is intentionally an *oracle-label* diagnostic when fed teacher
records from ``edge_decay_analyzer_data``: the labels use future path outcomes,
so passing results here are not deployable.  The purpose is to verify whether
these labels define a useful routing target before spending GPU time on LLM
fine-tuning.  A deployable run must replace teacher targets with model
predictions generated from past-only prompts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from training.hierarchical_direct_split_search import _norm_cdf
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class EdgeRouterExecutionConfig:
    hold_bars: int = 432
    cooldown_bars: int = 12
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    allow_hints: tuple[str, ...] = ("ALLOW_TREND_SPECIALIST",)
    reversal_hints: tuple[str, ...] = ("CONSIDER_REVERSAL_SPECIALIST",)
    skip_hints: tuple[str, ...] = (
        "REDUCE_OR_SKIP_TREND_SPECIALIST",
        "RANGE_ROUTER_ONLY",
        "LOW_CONFIDENCE_ROUTER",
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: str(r.get("date", "")))


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def route_record(record: dict[str, Any], cfg: EdgeRouterExecutionConfig) -> str:
    """Map an edge-decay target/prediction record to LONG/SHORT/NONE."""
    target_raw = record.get("prediction") or record.get("target") or "{}"
    target = json.loads(target_raw) if isinstance(target_raw, str) else dict(target_raw)
    hint = str(target.get("recommended_router_hint", ""))
    trend_side = str(target.get("trend_side", "NONE"))
    if hint in set(cfg.allow_hints):
        return trend_side if trend_side in {"LONG", "SHORT"} else "NONE"
    if hint in set(cfg.reversal_hints):
        side = _opposite(trend_side)
        return side if side in {"LONG", "SHORT"} else "NONE"
    return "NONE"


def _trade_stats(trade_returns: list[float]) -> dict[str, Any]:
    n = len(trade_returns)
    mean = sum(trade_returns) / n if n else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in trade_returns) / (n - 1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    t_like = mean / se if se > 0 else 0.0
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t_like))) if se > 0 else 1.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    effect_d = mean / std if std > 1e-12 else 0.0
    n_required = int(math.ceil(((1.959963984540054 + 0.8416212335729143) / abs(effect_d)) ** 2)) if abs(effect_d) > 1e-12 else None
    return {
        "n_trades": n,
        "mean_trade_ret_pct": mean * 100.0,
        "std_trade_ret_pct": std * 100.0,
        "t_stat_like": t_like,
        "p_value_mean_ret_approx": p_two,
        "ci95_mean_trade_ret_pct": [ci_low * 100.0, ci_high * 100.0],
        "effect_size_d": effect_d,
        "n_required_for_80pct_power_alpha5pct": n_required,
        "n_gap_to_power_rule": max(0, n_required - n) if n_required is not None else None,
    }


def _drawdown(peak: float, eq: float) -> float:
    if peak <= 0.0:
        return 0.0
    return max(0.0, 1.0 - max(0.0, eq) / peak)


def simulate_router_records(
    records: list[dict[str, Any]],
    market: pd.DataFrame,
    cfg: EdgeRouterExecutionConfig,
) -> dict[str, Any]:
    if not records:
        return {"period": {}, "sim": {"trade_entries": 0}, "trade_stats": _trade_stats([])}
    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)

    eq = peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    skipped_missing_bars = 0
    next_allowed_market_pos = 0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    hold = max(1, int(cfg.hold_bars))
    cooldown = max(0, int(cfg.cooldown_bars))
    delay = max(0, int(cfg.entry_delay_bars))

    hint_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {"LONG": 0, "SHORT": 0, "NONE": 0}

    for record in sorted(records, key=lambda r: str(r.get("date", ""))):
        target = json.loads(record.get("prediction") or record.get("target") or "{}")
        hint = str(target.get("recommended_router_hint", ""))
        hint_counts[hint] = hint_counts.get(hint, 0) + 1
        action = route_record(record, cfg)
        action_counts[action] = action_counts.get(action, 0) + 1
        if action == "NONE":
            continue
        dt = datetime.fromisoformat(str(record["date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped_missing_bars += 1
            continue
        if pos < next_allowed_market_pos:
            continue
        entry_pos = pos + delay
        exit_pos = entry_pos + hold
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped_missing_bars += 1
            continue

        signal = 1 if action == "LONG" else -1
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown(peak, eq))
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                close_ret = (open_j - float(opens[j + 1])) / open_j
            adverse_eq = eq * (1.0 + float(cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_market_pos = exit_pos + cooldown
        if eq <= 0.0:
            break

    start_dt = datetime.fromisoformat(str(records[0]["date"]))
    end_dt = datetime.fromisoformat(str(records[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(records[0]["date"]), "end": str(records[-1]["date"]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(records),
            "skipped_missing_bars": skipped_missing_bars,
            "entry_delay_bars": delay,
            "return_application": "actual_ohlc_bar_by_bar_strict_mdd_edge_router",
        },
        "trade_stats": _trade_stats(trade_returns),
        "router_counts": {"hints": hint_counts, "actions": action_counts},
    }


def _filter_records(records: Iterable[dict[str, Any]], start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    out = []
    for rec in records:
        ts = pd.to_datetime(rec["date"])
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append(rec)
    return out


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    records = load_jsonl(args.records)
    market = load_market_bars(args.market_csv)
    cfg = EdgeRouterExecutionConfig(
        hold_bars=args.hold_bars,
        cooldown_bars=args.cooldown_bars,
        entry_delay_bars=args.entry_delay_bars,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        allow_hints=tuple(x for x in args.allow_hints.split(",") if x),
        reversal_hints=tuple(x for x in args.reversal_hints.split(",") if x),
        skip_hints=tuple(x for x in args.skip_hints.split(",") if x),
    )
    split_specs = {
        "all": (args.start_date or None, args.end_date or None),
    }
    if args.train_start or args.val_start or args.oos_start:
        split_specs = {
            "train": (args.train_start or None, args.train_end or None),
            "val": (args.val_start or None, args.val_end or None),
            "oos": (args.oos_start or None, args.oos_end or None),
        }
    splits = {}
    for name, (start, end) in split_specs.items():
        rows = _filter_records(records, start, end)
        splits[name] = simulate_router_records(rows, market, cfg)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "records": args.records,
        "market_csv": args.market_csv,
        "execution": asdict(cfg),
        "splits": splits,
        "leakage_guard": {
            "oracle_targets_may_use_future_path": True,
            "not_deployable_unless_prediction_field_is_model_output": True,
            "strict_bar_by_bar": True,
            "uses_forward_return_column": False,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict diagnostic backtest for edge-decay router labels")
    p.add_argument("--records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/edge_decay_router_backtest.json")
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--train-start", default="")
    p.add_argument("--train-end", default="")
    p.add_argument("--val-start", default="")
    p.add_argument("--val-end", default="")
    p.add_argument("--oos-start", default="")
    p.add_argument("--oos-end", default="")
    p.add_argument("--hold-bars", type=int, default=432)
    p.add_argument("--cooldown-bars", type=int, default=12)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--allow-hints", default="ALLOW_TREND_SPECIALIST")
    p.add_argument("--reversal-hints", default="CONSIDER_REVERSAL_SPECIALIST")
    p.add_argument("--skip-hints", default="REDUCE_OR_SKIP_TREND_SPECIALIST,RANGE_ROUTER_ONLY,LOW_CONFIDENCE_ROUTER")
    return p.parse_args()


def main() -> None:
    out = run_backtest(parse_args())
    print(json.dumps({name: payload["sim"] for name, payload in out["splits"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
