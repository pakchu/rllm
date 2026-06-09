"""Strict bar-by-bar backtest for VLM action-score evaluation reports.

The evaluator consumes ``training.eval_vlm_policy`` JSON reports with
``action_scores``. It uses only the model's predicted action at the signal
timestamp, enters after a configurable bar delay, and marks equity through
actual OHLC bars so strict MDD includes intrabar adverse excursions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats, load_market_bars


def _load_action_rows(report_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(Path(report_path).read_text())
    rows = report.get("action_scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError("VLM eval report must contain a non-empty action_scores list")
    return rows, report


def _action_to_signal(action: str) -> int:
    key = str(action).upper().strip()
    if key.startswith("LONG_"):
        return 1
    if key.startswith("SHORT_"):
        return -1
    if key in {"BUY", "LONG"}:
        return 1
    if key in {"SELL", "SHORT"}:
        return -1
    return 0


def _action_to_hold_bars(action: str, default_hold_bars: int) -> int:
    """Return action-specific hold bars for labels like LONG_36."""
    key = str(action).upper().strip()
    if "_" not in key:
        return max(1, int(default_hold_bars))
    suffix = key.rsplit("_", 1)[-1]
    try:
        hold = int(suffix)
    except ValueError:
        return max(1, int(default_hold_bars))
    return max(1, hold)


def simulate(
    rows: list[dict[str, Any]],
    market_csv: str,
    exec_cfg: BarExecutionConfig,
    *,
    hold_bars: int,
    cooldown_bars: int = 0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
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
    hold = max(1, int(hold_bars))
    entry_delay = max(0, int(exec_cfg.entry_delay_bars))

    for row in rows:
        dt = datetime.fromisoformat(str(row["date"]))
        pos = date_to_pos.get(dt.replace(tzinfo=None))
        if pos is None:
            skipped_missing_bars += 1
            continue
        if pos < next_allowed_market_pos:
            continue
        pred_action = str(row.get("pred", "HOLD"))
        signal = _action_to_signal(pred_action)
        if signal == 0:
            continue
        row_hold = _action_to_hold_bars(pred_action, hold)
        entry_pos = pos + entry_delay
        exit_pos = entry_pos + row_hold
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

    start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(rows[0].get("date")), "end": str(rows[-1].get("date")), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "skipped_missing_bars": skipped_missing_bars,
            "hold_bars": hold,
            "entry_delay_bars": entry_delay,
            "return_application": "actual_ohlc_bar_by_bar_fixed_hold_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, report = _load_action_rows(args.eval_report)
    hold_bars = int(args.hold_bars) if int(args.hold_bars) > 0 else int(report.get("target_horizon", 1))
    exec_cfg = BarExecutionConfig(
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        drawdown_stop=1.0,
        pause_bars=0,
        monthly_loss_stop=1.0,
        entry_delay_bars=int(args.entry_delay_bars),
    )
    result = simulate(
        rows,
        args.market_csv,
        exec_cfg,
        hold_bars=hold_bars,
        cooldown_bars=int(args.cooldown_bars),
    )
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"eval_report": args.eval_report, "market_csv": args.market_csv},
        "source_eval": {
            "model": report.get("model"),
            "adapter_dir": report.get("adapter_dir"),
            "num_samples": report.get("num_samples"),
            "metrics": report.get("metrics"),
        },
        "execution": exec_cfg.__dict__ | {"cooldown_bars": int(args.cooldown_bars), "hold_bars": hold_bars},
        "result": result,
        "leakage_guard": {
            "uses_forward_return_column": False,
            "uses_model_pred_only": True,
            "entry_after_signal_by_bars": int(args.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
            "hold_bars_source": "--hold-bars" if int(args.hold_bars) > 0 else "eval_report.target_horizon",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict backtest VLM eval action_scores")
    p.add_argument("--eval-report", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hold-bars", type=int, default=0, help="<=0 uses eval_report.target_horizon")
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
