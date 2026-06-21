"""Strict backtest with online loss/pause overlays for generated actions.

This evaluates live-usable regime-break protection: pause future entries after
realized losses or local drawdown using only trades that have already completed.
It does not inspect future labels, future score distributions, or current-month
outcomes when deciding whether the next signal may trade.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.eval_text_trader import parse_trader_json
from training.strict_bar_backtest import _drawdown_from_trough, _trade_stats, load_market_bars


@dataclass(frozen=True)
class OnlineRiskOverlayConfig:
    predictions_jsonl: str
    market_csv: str
    output: str
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_hold_bars: int = 432
    pause_after_losses: int = 0
    pause_bars: int = 864
    rolling_window_trades: int = 0
    rolling_loss_stop_pct: float = 0.0
    rolling_drawdown_stop_pct: float = 0.0
    monthly_loss_stop_pct: float = 0.0


def _read_prediction_files(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [x for x in str(raw).split(",") if x.strip()]:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1)))
    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    if not out:
        raise ValueError(f"no prediction rows loaded from {raw}")
    return out


def _month_key(date: str) -> str:
    dt = datetime.fromisoformat(str(date))
    return f"{dt.year:04d}-{dt.month:02d}"


def _rolling_loss(trade_returns: list[float], n: int) -> float:
    if n <= 0 or not trade_returns:
        return 0.0
    xs = trade_returns[-int(n) :]
    eq = 1.0
    for r in xs:
        eq *= max(0.0, 1.0 + float(r))
    return min(0.0, eq - 1.0)


def _rolling_dd(trade_returns: list[float], n: int) -> float:
    if n <= 0 or not trade_returns:
        return 0.0
    eq = peak = 1.0
    dd = 0.0
    for r in trade_returns[-int(n) :]:
        eq *= max(0.0, 1.0 + float(r))
        peak = max(peak, eq)
        dd = max(dd, 1.0 - eq / peak if peak > 0 else 0.0)
    return dd


def run_overlay(cfg: OnlineRiskOverlayConfig) -> dict[str, Any]:
    rows = _read_prediction_files(cfg.predictions_jsonl)
    market = load_market_bars(cfg.market_csv)
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = peak = 1.0
    max_dd = 0.0
    entries = 0
    skipped_no_trade = 0
    skipped_cooldown = 0
    skipped_overlay = 0
    skipped_missing_bars = 0
    invalid_actions = 0
    forced_liquidations = 0
    next_allowed_pos = 0
    overlay_paused_until = -1
    consecutive_losses = 0
    trade_returns: list[float] = []
    executed: list[dict[str, Any]] = []
    month = ""
    month_start_eq = 1.0
    month_paused = False
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)

    for row in rows:
        signal_pos = int(row.get("signal_pos", -1))
        if signal_pos < next_allowed_pos:
            skipped_cooldown += 1
            continue
        if signal_pos < overlay_paused_until:
            skipped_overlay += 1
            continue
        mk = _month_key(str(row.get("date")))
        if mk != month:
            month = mk
            month_start_eq = eq
            month_paused = False
        if month_paused:
            skipped_overlay += 1
            continue

        if cfg.rolling_window_trades > 0 and len(trade_returns) >= int(cfg.rolling_window_trades):
            roll_loss = -_rolling_loss(trade_returns, int(cfg.rolling_window_trades)) * 100.0
            roll_dd = _rolling_dd(trade_returns, int(cfg.rolling_window_trades)) * 100.0
            if float(cfg.rolling_loss_stop_pct) > 0.0 and roll_loss >= float(cfg.rolling_loss_stop_pct):
                overlay_paused_until = signal_pos + max(1, int(cfg.pause_bars))
                skipped_overlay += 1
                continue
            if float(cfg.rolling_drawdown_stop_pct) > 0.0 and roll_dd >= float(cfg.rolling_drawdown_stop_pct):
                overlay_paused_until = signal_pos + max(1, int(cfg.pause_bars))
                skipped_overlay += 1
                continue

        action = parse_trader_json(json.dumps(row.get("prediction", {})))
        if action["gate"] != "TRADE":
            skipped_no_trade += 1
            continue
        side = str(action.get("side", "NONE"))
        if side not in {"LONG", "SHORT"}:
            invalid_actions += 1
            continue
        hold_bars = min(max(1, int(action.get("hold_bars", 0) or 0)), int(cfg.max_hold_bars))
        entry_pos = signal_pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped_missing_bars += 1
            continue

        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        signal = 1 if side == "LONG" else -1
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
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                forced_liquidations += 1
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        trade_returns.append(trade_ret)
        executed.append({"date": row.get("date"), "signal_pos": signal_pos, "side": side, "hold_bars": hold_bars, "trade_ret_pct": trade_ret * 100.0, "equity": eq})
        consecutive_losses = consecutive_losses + 1 if trade_ret < 0.0 else 0
        if int(cfg.pause_after_losses) > 0 and consecutive_losses >= int(cfg.pause_after_losses):
            overlay_paused_until = exit_pos + max(1, int(cfg.pause_bars))
            consecutive_losses = 0
        if float(cfg.monthly_loss_stop_pct) > 0.0 and (1.0 - eq / month_start_eq) * 100.0 >= float(cfg.monthly_loss_stop_pct):
            month_paused = True
        next_allowed_pos = exit_pos + max(0, int(cfg.cooldown_bars))
        if eq <= 0.0:
            break

    start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    years = max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else 0.0,
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "skipped_no_trade": skipped_no_trade,
            "skipped_cooldown": skipped_cooldown,
            "skipped_overlay": skipped_overlay,
            "skipped_missing_bars": skipped_missing_bars,
            "invalid_actions": invalid_actions,
            "forced_liquidations": forced_liquidations,
            "return_application": "generated_action_actual_ohlc_bar_by_bar_online_overlay",
        },
        "trade_stats": _trade_stats(trade_returns),
        "executed": executed,
        "leakage_guard": {
            "overlay_uses_only_completed_prior_trades": True,
            "does_not_use_future_prices_for_gate_decision": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest generated actions with online risk pause overlay")
    p.add_argument("--predictions-jsonl", required=True, help="Comma-separated prediction jsonl files")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--max-hold-bars", type=int, default=432)
    p.add_argument("--pause-after-losses", type=int, default=0)
    p.add_argument("--pause-bars", type=int, default=864)
    p.add_argument("--rolling-window-trades", type=int, default=0)
    p.add_argument("--rolling-loss-stop-pct", type=float, default=0.0)
    p.add_argument("--rolling-drawdown-stop-pct", type=float, default=0.0)
    p.add_argument("--monthly-loss-stop-pct", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    out = run_overlay(OnlineRiskOverlayConfig(**vars(parse_args())))
    print(json.dumps({"period": out["period"], "sim": out["sim"], "trade_stats": out["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
