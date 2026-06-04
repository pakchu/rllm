"""Strict backtest for using fade-warning model predictions as trend-route filters.

The fade-warning analyzer is not an entry policy.  This module tests it as a
veto/filter on the model's trend_side: trade with trend_side unless the predicted
fade_warning is in a selected skip set.  Parameter selection can be done on val
only and then frozen for OOS.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from training.eval_fade_warning_analyzer import parse_fade_warning_json
from training.hierarchical_direct_split_search import _norm_cdf
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class FadeWarningFilterConfig:
    hold_bars: int = 432
    cooldown_bars: int = 12
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    skip_fade_warnings: tuple[str, ...] = ("FADE_STRONG",)
    use_target: bool = False
    flip_fade_strong: bool = False


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: str(r.get("date", "")))


def _payload(record: dict[str, Any], cfg: FadeWarningFilterConfig) -> dict[str, str]:
    raw = record.get("target") if cfg.use_target else record.get("prediction") or record.get("target")
    return parse_fade_warning_json(str(raw or "{}"))


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def route_fade_warning_record(record: dict[str, Any], cfg: FadeWarningFilterConfig) -> tuple[str, str]:
    pred = _payload(record, cfg)
    side = pred.get("trend_side", "NONE")
    fade = pred.get("fade_warning", "NO_FADE_WARNING")
    if side not in {"LONG", "SHORT"}:
        return "NONE", "no_trend_side"
    if bool(cfg.flip_fade_strong) and fade == "FADE_STRONG":
        return _opposite(side), "flip_fade_strong_diagnostic"
    if fade in set(cfg.skip_fade_warnings):
        return "NONE", f"skip_{fade}"
    return side, "trend_allowed"


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


def simulate_fade_warning_filter_records(records: list[dict[str, Any]], market: pd.DataFrame, cfg: FadeWarningFilterConfig) -> dict[str, Any]:
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
    action_counts: dict[str, int] = {"LONG": 0, "SHORT": 0, "NONE": 0}
    reason_counts: dict[str, int] = {}
    payload_counts: dict[str, dict[str, int]] = {"fade_warning": {}, "trend_side": {}, "skip_reason": {}}

    for record in sorted(records, key=lambda r: str(r.get("date", ""))):
        payload = _payload(record, cfg)
        for key in payload_counts:
            val = str(payload.get(key, ""))
            payload_counts[key][val] = payload_counts[key].get(val, 0) + 1
        action, reason = route_fade_warning_record(record, cfg)
        action_counts[action] = action_counts.get(action, 0) + 1
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
            "return_application": "actual_ohlc_bar_by_bar_strict_mdd_fade_warning_filter",
        },
        "trade_stats": _trade_stats(trade_returns),
        "router_counts": {"actions": action_counts, "reasons": reason_counts, "payload": payload_counts},
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
    skip = tuple(x.strip() for x in str(args.skip_fade_warnings).split(",") if x.strip())
    cfg = FadeWarningFilterConfig(
        hold_bars=int(args.hold_bars),
        cooldown_bars=int(args.cooldown_bars),
        entry_delay_bars=int(args.entry_delay_bars),
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        skip_fade_warnings=skip,
        use_target=bool(args.use_target),
        flip_fade_strong=bool(args.flip_fade_strong),
    )
    split_specs = {"all": (args.start_date or None, args.end_date or None)}
    if args.train_start or args.val_start or args.oos_start:
        split_specs = {"train": (args.train_start or None, args.train_end or None), "val": (args.val_start or None, args.val_end or None), "oos": (args.oos_start or None, args.oos_end or None)}
    splits = {name: simulate_fade_warning_filter_records(_filter_records(records, start, end), market, cfg) for name, (start, end) in split_specs.items()}
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
            "fade_warning_is_filter_not_entry_policy": True,
            "flip_fade_strong_is_diagnostic_only": bool(args.flip_fade_strong),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _skip_sets(raw: str) -> list[tuple[str, ...]]:
    sets = []
    for part in str(raw).split(";"):
        vals = tuple(x.strip() for x in part.split(",") if x.strip())
        sets.append(vals)
    return sets


def _score_candidate(sim: dict[str, Any], *, min_trades: int, max_mdd: float) -> tuple[Any, ...]:
    s = sim["sim"]
    t = sim["trade_stats"]
    return (
        int(s["trade_entries"] >= int(min_trades)),
        int(float(s["strict_mdd_pct"]) <= float(max_mdd)),
        int(float(t["ci95_mean_trade_ret_pct"][0]) > 0.0),
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
    for hold, cool, skip, flip in itertools.product(_csv_ints(args.hold_bars_list), _csv_ints(args.cooldown_bars_list), _skip_sets(args.skip_fade_warning_sets), [False, True] if args.include_flip_diagnostic else [False]):
        cfg = FadeWarningFilterConfig(
            hold_bars=int(hold),
            cooldown_bars=int(cool),
            entry_delay_bars=int(args.entry_delay_bars),
            leverage=float(args.leverage),
            fee_rate=float(args.fee_rate),
            slippage_rate=float(args.slippage_rate),
            skip_fade_warnings=tuple(skip),
            use_target=False,
            flip_fade_strong=bool(flip),
        )
        val = simulate_fade_warning_filter_records(val_records, market, cfg)
        rows.append({"config": asdict(cfg), "val": val, "selection_score": _score_candidate(val, min_trades=args.min_trades, max_mdd=args.max_mdd)})
    rows.sort(key=lambda r: r["selection_score"], reverse=True)
    selected = FadeWarningFilterConfig(**rows[0]["config"])
    oos = simulate_fade_warning_filter_records(oos_records, market, selected)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "val_records": args.val_records,
        "oos_records": args.oos_records,
        "market_csv": args.market_csv,
        "selection_rule": {"selected_on": "val_only", "sort": "min_trades_pass,max_mdd_pass,positive_ci_pass,cagr_to_strict_mdd,cagr,-mdd,trades", "min_trades": int(args.min_trades), "max_mdd": float(args.max_mdd)},
        "selected_config": rows[0]["config"],
        "selected_val": rows[0]["val"],
        "selected_oos": oos,
        "top_val": [{"rank": i + 1, "config": r["config"], "val": r["val"]} for i, r in enumerate(rows[: int(args.top_k)])],
        "num_candidates": len(rows),
        "leakage_guard": {"selected_on_val_only": True, "oos_not_used_for_selection": True, "records_are_model_predictions": True, "strict_bar_by_bar": True, "flip_diagnostic_not_promotable": bool(rows[0]["config"].get("flip_fade_strong"))},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict backtest/sweep for fade-warning filter predictions")
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest")
    b.add_argument("--records", required=True)
    b.add_argument("--market-csv", required=True)
    b.add_argument("--output", default="results/fade_warning_filter_backtest.json")
    b.add_argument("--start-date", default="")
    b.add_argument("--end-date", default="")
    b.add_argument("--train-start", default="")
    b.add_argument("--train-end", default="")
    b.add_argument("--val-start", default="")
    b.add_argument("--val-end", default="")
    b.add_argument("--oos-start", default="")
    b.add_argument("--oos-end", default="")
    b.add_argument("--hold-bars", type=int, default=432)
    b.add_argument("--cooldown-bars", type=int, default=12)
    b.add_argument("--entry-delay-bars", type=int, default=1)
    b.add_argument("--leverage", type=float, default=0.5)
    b.add_argument("--fee-rate", type=float, default=0.0004)
    b.add_argument("--slippage-rate", type=float, default=0.0001)
    b.add_argument("--skip-fade-warnings", default="FADE_STRONG")
    b.add_argument("--use-target", action="store_true")
    b.add_argument("--flip-fade-strong", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("--val-records", required=True)
    s.add_argument("--oos-records", required=True)
    s.add_argument("--market-csv", required=True)
    s.add_argument("--output", default="results/fade_warning_filter_sweep.json")
    s.add_argument("--hold-bars-list", default="36,72,144,288,432")
    s.add_argument("--cooldown-bars-list", default="0,12,36")
    s.add_argument("--skip-fade-warning-sets", default="FADE_STRONG;FADE_STRONG,FADE_WATCH;")
    s.add_argument("--include-flip-diagnostic", action="store_true")
    s.add_argument("--entry-delay-bars", type=int, default=1)
    s.add_argument("--leverage", type=float, default=0.5)
    s.add_argument("--fee-rate", type=float, default=0.0004)
    s.add_argument("--slippage-rate", type=float, default=0.0001)
    s.add_argument("--min-trades", type=int, default=30)
    s.add_argument("--max-mdd", type=float, default=25.0)
    s.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "sweep":
        report = run_sweep(args)
        print(json.dumps({"selected_config": report["selected_config"], "val": report["selected_val"]["sim"], "oos": report["selected_oos"]["sim"]}, indent=2, ensure_ascii=False))
    else:
        report = run_backtest(args)
        print(json.dumps({name: payload["sim"] for name, payload in report["splits"].items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
