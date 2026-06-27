"""Strict risk-managed backtest for fixed price-action episode templates.

Unlike fixed-horizon episode validation, this uses structural invalidation from
the signal bar: LONG stops below the signal low, SHORT stops above the signal
high, and optional R-multiple take-profit exits.  It is still a fixed-template
validator, not a searcher.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.fixed_episode_template_backtest import _parse_specs
from training.price_action_episode_policy import add_sequence_context_features, build_episode_event_features
from training.strict_bar_backtest import _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class StructuralRiskCfg:
    input_csv: str
    output: str
    specs: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    include_sequence_context: bool = True
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_hold_bars: int = 288
    take_profit_r: float = 1.5
    stop_buffer_bps: float = 0.0
    min_risk_bps: float = 5.0
    cooldown_bars: int = 0
    side_override: str = ""


def _period_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _years(dates: pd.Series, idxs: np.ndarray) -> float:
    if len(idxs) == 0:
        return 1.0 / 365.25
    return max(1.0 / 365.25, float((pd.Timestamp(dates.iloc[int(idxs[-1])]) - pd.Timestamp(dates.iloc[int(idxs[0])])).days) / 365.25)


def _build_triggers(features: pd.DataFrame, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    triggers = []
    for spec in specs:
        if spec["event"] not in features.columns:
            continue
        for pos in np.flatnonzero(features[spec["event"]].to_numpy(dtype=float) > 0.5):
            triggers.append(dict(spec) | {"pos": int(pos)})
    return sorted(triggers, key=lambda r: (int(r["pos"]), str(r["event"])))


def _empty(dates: pd.Series, idxs: np.ndarray) -> dict[str, Any]:
    return {
        "period": {"start": str(dates.iloc[int(idxs[0])]) if len(idxs) else None, "end": str(dates.iloc[int(idxs[-1])]) if len(idxs) else None, "years": _years(dates, idxs)},
        "sim": {"ret_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "trade_entries": 0, "side_counts": {"LONG": 0, "SHORT": 0}, "samples": int(len(idxs)), "skipped_missing_bars": 0, "return_application": "structural_episode_risk_strict_mdd"},
        "trade_stats": _trade_stats([]),
        "executed": [],
    }


def simulate(market: pd.DataFrame, dates: pd.Series, triggers: list[dict[str, Any]], *, start: str, end: str, cfg: StructuralRiskCfg) -> dict[str, Any]:
    mask = _period_mask(dates, start, end)
    period_idxs = np.flatnonzero(mask)
    if len(period_idxs) == 0:
        return _empty(dates, period_idxs)
    by_pos: dict[int, list[dict[str, Any]]] = {}
    for t in triggers:
        pos = int(t["pos"])
        if 0 <= pos < len(mask) and mask[pos]:
            by_pos.setdefault(pos, []).append(t)

    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    executed: list[dict[str, Any]] = []
    skipped = 0
    stop_buffer = float(cfg.stop_buffer_bps) / 10_000.0
    min_risk = float(cfg.min_risk_bps) / 10_000.0
    for pos in period_idxs:
        pos = int(pos)
        if pos < next_allowed or pos not in by_pos:
            continue
        trig = by_pos[pos][0]
        side = str(trig["side"])
        signal = 1 if side == "LONG" else -1
        entry_pos = pos + int(cfg.entry_delay_bars)
        if entry_pos >= len(market) - 1:
            skipped += 1
            continue
        entry = float(opens[entry_pos])
        if entry <= 0.0:
            skipped += 1
            continue
        if signal > 0:
            stop = float(lows[pos]) * (1.0 - stop_buffer)
            risk = entry - stop
            tp = entry + float(cfg.take_profit_r) * risk
        else:
            stop = float(highs[pos]) * (1.0 + stop_buffer)
            risk = stop - entry
            tp = entry - float(cfg.take_profit_r) * risk
        if risk / entry < min_risk or risk <= 0.0:
            skipped += 1
            continue
        entry_eq = eq
        side_counts[side] += 1
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        exit_reason = "max_hold"
        exit_pos = min(len(market) - 2, entry_pos + int(cfg.max_hold_bars))
        for j in range(entry_pos, exit_pos + 1):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            if signal > 0:
                # Pessimistic same-bar ordering: stop before TP if both are touched.
                adverse_eq = eq * (1.0 + float(cfg.leverage) * ((float(lows[j]) - open_j) / open_j))
                max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
                if float(lows[j]) <= stop:
                    close_ret = (stop - open_j) / open_j
                    eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
                    exit_reason = "stop"
                    exit_pos = j
                    break
                if float(highs[j]) >= tp:
                    close_ret = (tp - open_j) / open_j
                    eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
                    exit_reason = "take_profit"
                    exit_pos = j
                    break
                close_ret = (float(opens[j + 1]) - open_j) / open_j
            else:
                adverse_eq = eq * (1.0 + float(cfg.leverage) * ((open_j - float(highs[j])) / open_j))
                max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
                if float(highs[j]) >= stop:
                    close_ret = (open_j - stop) / open_j
                    eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
                    exit_reason = "stop"
                    exit_pos = j
                    break
                if float(lows[j]) <= tp:
                    close_ret = (open_j - tp) / open_j
                    eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
                    exit_reason = "take_profit"
                    exit_pos = j
                    break
                close_ret = (open_j - float(opens[j + 1])) / open_j
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        trade_returns.append(trade_ret)
        executed.append({"date": str(dates.iloc[pos]), "signal_pos": pos, "entry_pos": entry_pos, "exit_pos": exit_pos, "side": side, "event": trig["event"], "exit_reason": exit_reason, "trade_ret_pct": trade_ret * 100.0, "equity": eq})
        next_allowed = int(exit_pos) + max(0, int(cfg.cooldown_bars))
        if eq <= 0.0:
            break
    years = _years(dates, period_idxs)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd = max_dd * 100.0
    return {
        "period": {"start": str(dates.iloc[int(period_idxs[0])]), "end": str(dates.iloc[int(period_idxs[-1])]), "years": years},
        "sim": {"ret_pct": ret_pct, "cagr_pct": cagr, "strict_mdd_pct": mdd, "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else float("inf"), "trade_entries": len(trade_returns), "side_counts": side_counts, "samples": int(len(period_idxs)), "skipped_missing_bars": skipped, "return_application": "structural_episode_risk_strict_mdd"},
        "trade_stats": _trade_stats(trade_returns),
        "executed": executed,
    }


def run(cfg: StructuralRiskCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    features = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        features = add_sequence_context_features(market, features, windows)
    specs = _parse_specs(cfg.specs)
    if cfg.side_override:
        side = cfg.side_override.upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("--side-override must be LONG or SHORT")
        specs = [dict(spec) | {"side": side, "episode": f"override_{side.lower()}"} for spec in specs]
    triggers = _build_triggers(features, specs)
    portfolio = {
        "train": simulate(market, dates, triggers, start=cfg.train_start, end=cfg.train_end, cfg=cfg),
        "test": simulate(market, dates, triggers, start=cfg.test_start, end=cfg.test_end, cfg=cfg),
        "eval": simulate(market, dates, triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=cfg),
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "templates": specs,
        "portfolio": {k: {"period": v["period"], "sim": v["sim"], "trade_stats": v["trade_stats"], "executed_sample": v["executed"][:20]} for k, v in portfolio.items()},
        "protocol": "fixed templates with signal-bar structural stop, R-multiple TP, max-hold timeout; no search/ranking in this script",
        "leakage_guard": {"template_selection_uses_eval": False, "features_use_rows_at_or_before_t": True, "entry_uses_next_open": int(cfg.entry_delay_bars) >= 1, "same_bar_stop_tp_order": "pessimistic_stop_first"},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--specs", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(StructuralRiskCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=StructuralRiskCfg.windows)
    p.add_argument("--no-sequence-context", dest="include_sequence_context", action="store_false")
    p.set_defaults(include_sequence_context=StructuralRiskCfg.include_sequence_context)
    p.add_argument("--entry-delay-bars", type=int, default=StructuralRiskCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=StructuralRiskCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=StructuralRiskCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=StructuralRiskCfg.slippage_rate)
    p.add_argument("--max-hold-bars", type=int, default=StructuralRiskCfg.max_hold_bars)
    p.add_argument("--take-profit-r", type=float, default=StructuralRiskCfg.take_profit_r)
    p.add_argument("--stop-buffer-bps", type=float, default=StructuralRiskCfg.stop_buffer_bps)
    p.add_argument("--min-risk-bps", type=float, default=StructuralRiskCfg.min_risk_bps)
    p.add_argument("--cooldown-bars", type=int, default=StructuralRiskCfg.cooldown_bars)
    p.add_argument("--side-override", default=StructuralRiskCfg.side_override, help="optional LONG/SHORT override for inversion diagnostics")
    return p.parse_args()


def main() -> None:
    report = run(StructuralRiskCfg(**vars(parse_args())))
    print(json.dumps({
        "output": report["config"]["output"],
        "templates": report["templates"],
        "portfolio": {k: v["sim"] | {"p": v["trade_stats"].get("p_value_mean_ret_approx")} for k, v in report["portfolio"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
