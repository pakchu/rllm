"""Strict stop/target backtest for pressure-analyzer predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.eval_economic_path_shape_sft import parse_analyzer
from training.strict_bar_backtest import _drawdown_from_trough, _trade_stats, load_market_bars


@dataclass(frozen=True)
class PressureBacktestConfig:
    horizon_bars: int = 36
    target_pct: float = 0.5
    stop_pct: float = 0.6
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1
    cooldown_bars: int = 0
    ambiguous_same_bar: str = "stop_first"


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no prediction rows loaded from {path}")
    return rows


def pressure_to_side(row: dict[str, Any], *, source: str = "prediction") -> str:
    obj = row.get(source, {})
    if isinstance(obj, str):
        parsed = parse_analyzer(obj)
    else:
        parsed = parse_analyzer(json.dumps(obj))
    p = parsed.get("direction_pressure", "NO_TRADE_FAVORED")
    if p == "LONG_FAVORED":
        return "LONG"
    if p == "SHORT_FAVORED":
        return "SHORT"
    return "NONE"


def strict_pressure_backtest(rows: list[dict[str, Any]], market, cfg: PressureBacktestConfig, *, source: str = "prediction") -> dict[str, Any]:
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    target = float(cfg.target_pct) / 100.0
    stop = float(cfg.stop_pct) / 100.0
    horizon = max(1, int(cfg.horizon_bars))
    entry_delay = max(0, int(cfg.entry_delay_bars))
    cooldown = max(0, int(cfg.cooldown_bars))
    next_allowed_pos = 0
    entries = 0
    skipped_no_trade = 0
    skipped_cooldown = 0
    skipped_missing_bars = 0
    target_exits = 0
    stop_exits = 0
    time_exits = 0
    ambiguous_exits = 0
    trade_returns: list[float] = []

    for row in sorted(rows, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1) or -1))):
        signal_pos = int(row.get("signal_pos", -1))
        if signal_pos < next_allowed_pos:
            skipped_cooldown += 1
            continue
        side = pressure_to_side(row, source=source)
        if side == "NONE":
            skipped_no_trade += 1
            continue
        entry_pos = signal_pos + entry_delay
        last_pos = entry_pos + horizon
        if entry_pos >= len(market) - 1 or last_pos >= len(market):
            skipped_missing_bars += 1
            continue
        entry_price = float(opens[entry_pos])
        if entry_price <= 0:
            skipped_missing_bars += 1
            continue
        entry_eq = eq
        entries += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        exit_pos = last_pos
        exit_reason = "time"
        signal = 1 if side == "LONG" else -1
        for j in range(entry_pos, last_pos):
            open_j = float(opens[j])
            if open_j <= 0:
                continue
            if signal > 0:
                adverse_ret = (float(lows[j]) - open_j) / open_j
                hit_target = (float(highs[j]) / entry_price - 1.0) >= target
                hit_stop = (float(lows[j]) / entry_price - 1.0) <= -stop
            else:
                adverse_ret = (open_j - float(highs[j])) / open_j
                hit_target = (entry_price / float(lows[j]) - 1.0) >= target if float(lows[j]) > 0 else False
                hit_stop = (entry_price / float(highs[j]) - 1.0) <= -stop if float(highs[j]) > 0 else False
            adverse_eq = eq * (1.0 + float(cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            if hit_target or hit_stop:
                exit_pos = j
                if hit_target and hit_stop:
                    ambiguous_exits += 1
                    exit_reason = "stop" if cfg.ambiguous_same_bar == "stop_first" else "target"
                else:
                    exit_reason = "target" if hit_target else "stop"
                break
        if exit_reason == "target":
            gross_ret = target
            target_exits += 1
        elif exit_reason == "stop":
            gross_ret = -stop
            stop_exits += 1
        else:
            exit_open = float(opens[last_pos])
            gross_ret = (exit_open / entry_price - 1.0) if side == "LONG" else (entry_price / exit_open - 1.0)
            time_exits += 1
        eq *= max(0.0, 1.0 + float(cfg.leverage) * gross_ret)
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed_pos = exit_pos + cooldown
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
        "period": {"start": str(rows[0]["date"]), "end": str(rows[-1]["date"]), "years": years},
        "sim": {
            "ret_pct": ret_pct,
            "cagr_pct": cagr_pct,
            "strict_mdd_pct": mdd_pct,
            "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
            "trade_entries": entries,
            "turnover_legs": entries * 2,
            "samples": len(rows),
            "skipped_no_trade": skipped_no_trade,
            "skipped_cooldown": skipped_cooldown,
            "skipped_missing_bars": skipped_missing_bars,
            "target_exits": target_exits,
            "stop_exits": stop_exits,
            "time_exits": time_exits,
            "ambiguous_same_bar_exits": ambiguous_exits,
            "return_application": "pressure_prediction_stop_target_strict_mdd_stop_first_ambiguous",
        },
        "trade_stats": _trade_stats(trade_returns),
    }


def run_pressure_backtest(
    *,
    predictions_jsonl: str,
    market_csv: str,
    output: str,
    source: str = "prediction",
    horizon_bars: int = 36,
    target_pct: float = 0.5,
    stop_pct: float = 0.6,
    leverage: float = 0.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    entry_delay_bars: int = 1,
    cooldown_bars: int = 0,
) -> dict[str, Any]:
    cfg = PressureBacktestConfig(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, leverage=leverage, fee_rate=fee_rate, slippage_rate=slippage_rate, entry_delay_bars=entry_delay_bars, cooldown_bars=cooldown_bars)
    rows = load_prediction_rows(predictions_jsonl)
    market = load_market_bars(market_csv)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "predictions_jsonl": str(Path(predictions_jsonl).resolve()), "market_csv": str(Path(market_csv).resolve()), "source": source, "config": asdict(cfg), "backtest": strict_pressure_backtest(rows, market, cfg, source=source), "leakage_guard": {"predictions_are_past_prompt_outputs": True, "uses_forward_return_column": False, "strict_mdd_includes_intrabar_adverse_excursion": True, "ambiguous_target_stop_same_bar_is_stop_first": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", default="prediction")
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--target-pct", type=float, default=0.5)
    p.add_argument("--stop-pct", type=float, default=0.6)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=0)
    return p.parse_args()


def main() -> None:
    out = run_pressure_backtest(**vars(parse_args()))
    print(json.dumps({"summary": out["backtest"]["sim"], "trade_stats": out["backtest"]["trade_stats"]}, indent=2))


if __name__ == "__main__":
    main()
