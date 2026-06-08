"""Strict variable-hold backtest for single-policy prediction rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.eval_single_policy import parse_policy_json, policy_to_action
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats, load_market_bars


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _policy_from_row(row: dict[str, Any]) -> dict[str, str]:
    if isinstance(row.get("policy_prediction"), dict):
        return parse_policy_json(json.dumps(row["policy_prediction"]))
    return parse_policy_json(str(row.get("target", "{}")))


def simulate(rows: list[dict[str, Any]], market_csv: str, exec_cfg: BarExecutionConfig, *, cooldown_bars: int = 0) -> dict[str, Any]:
    market = load_market_bars(market_csv)
    date_to_pos = {ts.to_pydatetime().replace(tzinfo=None): int(i) for i, ts in enumerate(market["date"])}
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    rows = sorted(rows, key=lambda r: str(r.get("date", "")))

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    entries = 0
    skipped_missing_bars = 0
    next_allowed_market_pos = 0
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)

    for row in rows:
        dt = datetime.fromisoformat(str(row["date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped_missing_bars += 1
            continue
        if pos < next_allowed_market_pos:
            continue
        action = policy_to_action(_policy_from_row(row))
        side = str(action.get("side", "NONE"))
        hold_bars = int(action.get("hold_bars", 0) or 0)
        signal = 1 if side == "LONG" else -1 if side == "SHORT" else 0
        if signal == 0 or hold_bars <= 0:
            continue
        entry_pos = pos + int(exec_cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped_missing_bars += 1
            continue

        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
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
            adverse_eq = eq * (1.0 + float(exec_cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(exec_cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_market_pos = exit_pos + max(0, int(cooldown_bars))
        if eq <= 0.0:
            break

    start_dt = datetime.fromisoformat(str(rows[0]["date"])) if rows else datetime.now()
    end_dt = datetime.fromisoformat(str(rows[-1]["date"])) if rows else start_dt
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(rows[0].get("date")) if rows else None, "end": str(rows[-1].get("date")) if rows else None, "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "skipped_missing_bars": skipped_missing_bars,
            "entry_delay_bars": int(exec_cfg.entry_delay_bars),
            "return_application": "actual_ohlc_bar_by_bar_variable_hold_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.predictions_jsonl)
    exec_cfg = BarExecutionConfig(
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(args.entry_delay_bars),
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"predictions_jsonl": args.predictions_jsonl, "market_csv": args.market_csv},
        "execution": exec_cfg.__dict__ | {"cooldown_bars": int(args.cooldown_bars)},
        "result": simulate(rows, args.market_csv, exec_cfg, cooldown_bars=int(args.cooldown_bars)),
        "leakage_guard": {
            "uses_forward_return_column": False,
            "uses_prediction_target_or_policy_prediction_only": True,
            "entry_after_signal_by_bars": int(args.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict backtest single-policy prediction rows")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
