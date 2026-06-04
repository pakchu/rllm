"""Strict backtest for compact router-state analyzer predictions.

This evaluates deployable *model prediction* JSONL files produced by
``eval_compact_path_shape_analyzer``.  It intentionally supports a conservative
``learned_fields`` mode that ignores weak/collapsed fields from the first
compact run and trades only when the learned trend/horizon/edge fields agree.
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

from training.eval_compact_path_shape_analyzer import parse_compact_path_shape_json
from training.hierarchical_direct_split_search import _norm_cdf
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class CompactRouterExecutionConfig:
    cooldown_bars: int = 12
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    short_hold_bars: int = 72
    mid_hold_bars: int = 144
    long_hold_bars: int = 432
    min_edge_quality: str = "STRONG"
    routing_mode: str = "learned_fields"  # learned_fields | action_path
    use_target: bool = False


EDGE_RANK = {"NO_EDGE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}


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


def _payload(record: dict[str, Any], cfg: CompactRouterExecutionConfig) -> dict[str, str]:
    raw = record.get("target") if cfg.use_target else record.get("prediction") or record.get("target")
    return parse_compact_path_shape_json(str(raw or "{}"))


def hold_for_policy(policy: str, cfg: CompactRouterExecutionConfig) -> int:
    if policy == "SHORT_STEP":
        return max(1, int(cfg.short_hold_bars))
    if policy == "MID_STEP":
        return max(1, int(cfg.mid_hold_bars))
    if policy == "LONG_STEP":
        return max(1, int(cfg.long_hold_bars))
    return 0


def route_compact_record(record: dict[str, Any], cfg: CompactRouterExecutionConfig) -> tuple[str, int, str]:
    """Return (LONG/SHORT/NONE, hold_bars, reason)."""
    pred = _payload(record, cfg)
    min_rank = EDGE_RANK.get(str(cfg.min_edge_quality), EDGE_RANK["STRONG"])
    if EDGE_RANK.get(pred["edge_quality"], 0) < min_rank:
        return "NONE", 0, "edge_quality_below_min"
    hold = hold_for_policy(pred["horizon_policy"], cfg)
    if hold <= 0:
        return "NONE", 0, "skip_horizon_policy"
    trend_side = pred["trend_side"]
    if trend_side not in {"LONG", "SHORT"}:
        return "NONE", 0, "no_trend_side"
    if cfg.routing_mode == "learned_fields":
        return trend_side, hold, "trend_side_horizon_edge"
    if cfg.routing_mode == "action_path":
        action_path = pred["action_path"]
        if action_path == "TREND":
            return trend_side, hold, "action_path_trend"
        if action_path == "FADE":
            return _opposite(trend_side), hold, "action_path_fade"
        return "NONE", 0, "action_path_none"
    raise ValueError("routing_mode must be one of {'learned_fields','action_path'}")


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


def simulate_compact_router_records(
    records: list[dict[str, Any]],
    market: pd.DataFrame,
    cfg: CompactRouterExecutionConfig,
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
    cooldown = max(0, int(cfg.cooldown_bars))
    delay = max(0, int(cfg.entry_delay_bars))
    action_counts: dict[str, int] = {"LONG": 0, "SHORT": 0, "NONE": 0}
    hold_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    payload_counts: dict[str, dict[str, int]] = {"edge_quality": {}, "horizon_policy": {}, "action_path": {}, "trend_side": {}}

    for record in sorted(records, key=lambda r: str(r.get("date", ""))):
        payload = _payload(record, cfg)
        for key in payload_counts:
            val = str(payload.get(key, ""))
            payload_counts[key][val] = payload_counts[key].get(val, 0) + 1
        action, hold, reason = route_compact_record(record, cfg)
        action_counts[action] = action_counts.get(action, 0) + 1
        hold_counts[str(hold)] = hold_counts.get(str(hold), 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
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
        exit_pos = entry_pos + int(hold)
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
            "return_application": "actual_ohlc_bar_by_bar_strict_mdd_compact_router",
        },
        "trade_stats": _trade_stats(trade_returns),
        "router_counts": {"actions": action_counts, "holds": hold_counts, "reasons": reason_counts, "payload": payload_counts},
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
    cfg = CompactRouterExecutionConfig(
        cooldown_bars=int(args.cooldown_bars),
        entry_delay_bars=int(args.entry_delay_bars),
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        short_hold_bars=int(args.short_hold_bars),
        mid_hold_bars=int(args.mid_hold_bars),
        long_hold_bars=int(args.long_hold_bars),
        min_edge_quality=str(args.min_edge_quality),
        routing_mode=str(args.routing_mode),
        use_target=bool(args.use_target),
    )
    split_specs = {"all": (args.start_date or None, args.end_date or None)}
    if args.train_start or args.val_start or args.oos_start:
        split_specs = {
            "train": (args.train_start or None, args.train_end or None),
            "val": (args.val_start or None, args.val_end or None),
            "oos": (args.oos_start or None, args.oos_end or None),
        }
    splits = {}
    for name, (start, end) in split_specs.items():
        rows = _filter_records(records, start, end)
        splits[name] = simulate_compact_router_records(rows, market, cfg)
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "records": args.records,
        "market_csv": args.market_csv,
        "execution": asdict(cfg),
        "splits": splits,
        "leakage_guard": {
            "uses_model_predictions_when_use_target_false": not bool(args.use_target),
            "target_mode_is_oracle_only": bool(args.use_target),
            "strict_bar_by_bar": True,
            "uses_forward_return_column": False,
            "learned_fields_mode_ignores_action_path_and_risk_budget": str(args.routing_mode) == "learned_fields",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict backtest for compact router-state analyzer predictions")
    p.add_argument("--records", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", default="results/compact_router_backtest.json")
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--train-start", default="")
    p.add_argument("--train-end", default="")
    p.add_argument("--val-start", default="")
    p.add_argument("--val-end", default="")
    p.add_argument("--oos-start", default="")
    p.add_argument("--oos-end", default="")
    p.add_argument("--cooldown-bars", type=int, default=12)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--short-hold-bars", type=int, default=72)
    p.add_argument("--mid-hold-bars", type=int, default=144)
    p.add_argument("--long-hold-bars", type=int, default=432)
    p.add_argument("--min-edge-quality", choices=["WEAK", "MODERATE", "STRONG"], default="STRONG")
    p.add_argument("--routing-mode", choices=["learned_fields", "action_path"], default="learned_fields")
    p.add_argument("--use-target", action="store_true")
    return p.parse_args()


def main() -> None:
    out = run_backtest(parse_args())
    print(json.dumps({name: payload["sim"] for name, payload in out["splits"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
