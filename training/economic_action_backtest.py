"""Strict OHLC backtest for generated economic-preference trader actions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.strict_bar_backtest import _drawdown_from_trough, _mark_worst_order_bar_path, _trade_stats, load_market_bars


VALID_GATES = {"TRADE", "NO_TRADE"}
VALID_SIDES = {"LONG", "SHORT", "NONE"}


def parse_trader_json(text: str) -> dict[str, Any]:
    """Parse a trader action without importing model/inference dependencies.

    The original parser lives in ``training.eval_text_trader`` but that module
    imports VLM/model helpers and can require torch even for pure JSON
    backtests.  Backtest tooling only needs this small validation routine, so
    keep it local and dependency-light.
    """

    raw = str(text).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        import re

        obj = {}
        for match in re.finditer(r"\{[^{}]*\}", raw, flags=re.DOTALL):
            try:
                candidate = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(candidate, dict):
                obj = candidate
                break
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    side = str(obj.get("side", "NONE")).upper()
    if gate not in VALID_GATES:
        gate = "NO_TRADE"
    if side not in VALID_SIDES:
        side = "NONE"
    try:
        hold_bars = int(obj.get("hold_bars", 0) or 0)
    except Exception:
        hold_bars = 0
    if gate == "NO_TRADE":
        side = "NONE"
        hold_bars = 0
    elif hold_bars <= 0:
        hold_bars = 0
    return {"gate": gate, "side": side, "hold_bars": hold_bars}


@dataclass(frozen=True)
class EconomicActionBacktestConfig:
    annualization_start: str = ""
    annualization_end: str = ""
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    max_hold_bars: int = 432


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no prediction rows loaded from {path}")
    return rows


def dedupe_signal_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first model action per signal timestamp.

    Economic preference JSONL can contain multiple rejected alternatives for the
    same prompt.  A live system emits one action per prompt, so strict backtest
    must not count duplicate preference pairs as duplicate opportunities.
    """

    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1))):
        key = (str(row.get("date")), int(row.get("signal_pos", -1)))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def strict_backtest_actions(rows: list[dict[str, Any]], market, cfg: EconomicActionBacktestConfig) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    entry_delay = max(0, int(cfg.entry_delay_bars))
    cooldown = max(0, int(cfg.cooldown_bars))
    max_hold = max(1, int(cfg.max_hold_bars))
    next_allowed_pos = 0
    entries = 0
    skipped_no_trade = 0
    skipped_missing_bars = 0
    skipped_cooldown = 0
    invalid_actions = 0
    forced_liquidations = 0
    trade_returns: list[float] = []

    for row in rows:
        signal_pos = int(row.get("signal_pos", -1))
        if signal_pos < next_allowed_pos:
            skipped_cooldown += 1
            continue
        action = parse_trader_json(json.dumps(row.get("prediction", {})))
        if action["gate"] != "TRADE":
            skipped_no_trade += 1
            continue
        side = str(action.get("side", "NONE"))
        if side not in {"LONG", "SHORT"}:
            invalid_actions += 1
            continue
        hold_bars = int(action.get("hold_bars", 0) or 0)
        if hold_bars <= 0:
            invalid_actions += 1
            continue
        hold_bars = min(max_hold, hold_bars)
        entry_pos = signal_pos + entry_delay
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
            peak, bar_dd = _mark_worst_order_bar_path(
                equity_at_open=eq,
                peak=peak,
                open_price=open_j,
                high_price=float(highs[j]),
                low_price=float(lows[j]),
                signal=signal,
                leverage=float(cfg.leverage),
            )
            max_dd = max(max_dd, bar_dd)
            if signal > 0:
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                close_ret = (open_j - float(opens[j + 1])) / open_j
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                forced_liquidations += 1
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_pos = exit_pos + cooldown
        if eq <= 0.0:
            break

    trade_start_dt = datetime.fromisoformat(str(rows[0]["date"]))
    trade_end_dt = datetime.fromisoformat(str(rows[-1]["date"]))
    start_dt = datetime.fromisoformat(str(cfg.annualization_start)) if str(cfg.annualization_start).strip() else trade_start_dt
    end_dt = datetime.fromisoformat(str(cfg.annualization_end)) if str(cfg.annualization_end).strip() else trade_end_dt
    years = max(1.0 / 365.25, (end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    ratio = cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf")
    return {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "trade_period": {"start": str(trade_start_dt), "end": str(trade_end_dt)},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": ratio,
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "skipped_no_trade": skipped_no_trade,
            "skipped_cooldown": skipped_cooldown,
            "skipped_missing_bars": skipped_missing_bars,
            "invalid_actions": invalid_actions,
            "forced_liquidations": forced_liquidations,
            "entry_delay_bars": entry_delay,
            "return_application": "generated_action_actual_ohlc_bar_by_bar_strict_mdd",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run_economic_action_backtest(
    *,
    predictions_jsonl: str,
    market_csv: str,
    output: str,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    cooldown_bars: int = 0,
    max_hold_bars: int = 432,
    annualization_start: str = "",
    annualization_end: str = "",
) -> dict[str, Any]:
    cfg = EconomicActionBacktestConfig(
        annualization_start=str(annualization_start),
        annualization_end=str(annualization_end),
        leverage=float(leverage),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        entry_delay_bars=int(entry_delay_bars),
        cooldown_bars=int(cooldown_bars),
        max_hold_bars=int(max_hold_bars),
    )
    raw_rows = load_prediction_rows(predictions_jsonl)
    rows = dedupe_signal_predictions(raw_rows)
    market = load_market_bars(market_csv)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "predictions_jsonl": str(Path(predictions_jsonl).resolve()),
        "market_csv": str(Path(market_csv).resolve()),
        "config": asdict(cfg),
        "dedupe": {"raw_prediction_rows": len(raw_rows), "unique_signal_rows": len(rows)},
        "backtest": strict_backtest_actions(rows, market, cfg),
        "leakage_guard": {
            "predictions_are_one_action_per_past_prompt_after_dedupe": True,
            "uses_forward_return_column": False,
            "entry_after_signal_by_bars": int(entry_delay_bars),
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict backtest generated economic-preference trader predictions")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--max-hold-bars", type=int, default=432)
    p.add_argument("--annualization-start", default="")
    p.add_argument("--annualization-end", default="")
    return p.parse_args()


def main() -> None:
    out = run_economic_action_backtest(**vars(parse_args()))
    print(json.dumps({"summary": out["backtest"]["sim"], "trade_stats": out["backtest"]["trade_stats"]}, indent=2))


if __name__ == "__main__":
    main()
