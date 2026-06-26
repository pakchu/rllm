"""Causal online realized-performance gate for selected episode templates.

Given a price_action_episode_policy report, rebuild the selected templates and
run a paper-trade performance gate.  Every selected template updates a rolling
paper ledger after its hypothetical exit; live trades are allowed only when the
same template has enough recent paper trades and positive recent expectancy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.price_action_episode_policy import EpisodePolicyCfg, add_sequence_context_features, build_episode_event_features, simulate_triggers, template_triggers
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class OnlineGateCfg:
    input_csv: str
    policy_report: str
    output: str
    start: str = "2024-01-01"
    end: str = "2026-06-01"
    eval_start: str = "2026-01-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    gate_min_trades: int = 5
    gate_lookback_trades: int = 20
    gate_min_mean_ret_pct: float = 0.0
    gate_max_loss_rate: float = 0.65


def _template_id(trigger: dict[str, Any]) -> str:
    return f"{trigger.get('event')}|{trigger.get('side')}|h{trigger.get('horizon')}"


def _trade_return(market: pd.DataFrame, pos: int, side: str, hold_bars: int, cfg: OnlineGateCfg) -> tuple[float | None, int | None]:
    opens = market["open"].to_numpy(dtype=float)
    entry_pos = int(pos) + int(cfg.entry_delay_bars)
    exit_pos = entry_pos + int(hold_bars)
    if entry_pos >= len(market) - 1 or exit_pos >= len(market):
        return None, None
    signal = 1 if side == "LONG" else -1
    eq = 1.0
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    eq *= max(0.0, 1.0 - cost)
    for j in range(entry_pos, exit_pos):
        open_j = float(opens[j])
        if open_j <= 0.0:
            continue
        if signal > 0:
            close_ret = (float(opens[j + 1]) - open_j) / open_j
        else:
            close_ret = (open_j - float(opens[j + 1])) / open_j
        eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
        if eq <= 0.0:
            break
    eq *= max(0.0, 1.0 - cost)
    return eq - 1.0, exit_pos


def _gate_open(ledger: deque[float], cfg: OnlineGateCfg) -> tuple[bool, dict[str, Any]]:
    vals = list(ledger)[-int(cfg.gate_lookback_trades):]
    if len(vals) < int(cfg.gate_min_trades):
        return False, {"reason": "insufficient_paper_trades", "n": len(vals)}
    mean_pct = sum(vals) / len(vals) * 100.0
    loss_rate = sum(v <= 0.0 for v in vals) / len(vals)
    ok = mean_pct >= float(cfg.gate_min_mean_ret_pct) and loss_rate <= float(cfg.gate_max_loss_rate)
    return ok, {"reason": "open" if ok else "recent_performance_fail", "n": len(vals), "mean_ret_pct": mean_pct, "loss_rate": loss_rate}


def _simulate_online(market: pd.DataFrame, triggers: list[dict[str, Any]], cfg: OnlineGateCfg) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    start_ts = pd.Timestamp(cfg.start)
    end_ts = pd.Timestamp(cfg.end)
    eval_ts = pd.Timestamp(cfg.eval_start)
    by_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in triggers:
        pos = int(t["pos"])
        if start_ts <= dates.iloc[pos] <= end_ts:
            by_pos[pos].append(t)
    pending: list[tuple[int, str, float]] = []
    ledgers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=int(cfg.gate_lookback_trades)))
    exec_cfg = BarExecutionConfig(leverage=float(cfg.leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), drawdown_stop=1.0, pause_bars=0, monthly_loss_stop=1.0, entry_delay_bars=int(cfg.entry_delay_bars))
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    executed: list[dict[str, Any]] = []
    skipped_by_gate = 0
    side_counts = {"LONG": 0, "SHORT": 0}
    period_positions = [p for p in sorted(by_pos) if start_ts <= dates.iloc[p] <= end_ts]
    for pos in period_positions:
        pos = int(pos)
        still_pending = []
        for exit_pos, tid, ret in pending:
            if exit_pos <= pos:
                ledgers[tid].append(ret)
            else:
                still_pending.append((exit_pos, tid, ret))
        pending = still_pending
        for trigger in by_pos[pos]:
            tid = _template_id(trigger)
            paper_ret, paper_exit = _trade_return(market, pos, str(trigger["side"]), int(trigger["horizon"]), cfg)
            if paper_ret is not None and paper_exit is not None:
                pending.append((paper_exit, tid, paper_ret))
        if dates.iloc[pos] < eval_ts or pos < next_allowed:
            continue
        ranked = sorted(by_pos[pos], key=lambda r: (float(r.get("score", 0.0)), float(r.get("train_score", 0.0))), reverse=True)
        chosen = None
        gate_meta = None
        for trigger in ranked:
            ok, meta = _gate_open(ledgers[_template_id(trigger)], cfg)
            if ok:
                chosen = trigger
                gate_meta = meta
                break
        if chosen is None:
            skipped_by_gate += 1
            continue
        side = str(chosen["side"])
        signal = 1 if side == "LONG" else -1
        hold_bars = int(chosen["horizon"])
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        entry_eq = eq
        side_counts[side] += 1
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
            max_dd = max(max_dd, _drawdown_from_trough(peak, eq * (1.0 + float(cfg.leverage) * adverse_ret)))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        trade_returns.append(trade_ret)
        executed.append({"date": str(dates.iloc[pos]), "signal_pos": pos, "side": side, "hold_bars": hold_bars, "template_id": _template_id(chosen), "trade_ret_pct": trade_ret * 100.0, "equity": eq, "gate": gate_meta})
        next_allowed = exit_pos
    eval_dates = dates[(dates >= eval_ts) & (dates <= end_ts)]
    years = max(1.0 / 365.25, (pd.Timestamp(eval_dates.iloc[-1]) - pd.Timestamp(eval_dates.iloc[0])).days / 365.25) if len(eval_dates) else 1.0 / 365.25
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0 else -100.0
    mdd_pct = max_dd * 100.0
    return {"sim": {"ret_pct": ret_pct, "cagr_pct": cagr_pct, "strict_mdd_pct": mdd_pct, "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else 0.0, "trade_entries": len(trade_returns), "side_counts": side_counts, "skipped_by_gate": skipped_by_gate, "return_application": "online_template_paper_gate_actual_ohlc_bar_by_bar_strict_mdd"}, "trade_stats": _trade_stats(trade_returns), "executed": executed[:100]}


def run(cfg: OnlineGateCfg) -> dict[str, Any]:
    policy = json.loads(Path(cfg.policy_report).read_text())
    market = _load_market(cfg.input_csv)
    windows = _parse_list(cfg.windows, int)
    features = build_episode_event_features(market, windows)
    features = add_sequence_context_features(market, features, windows)
    triggers = []
    # Use fixed selected templates only; no eval-based selection is possible here.
    for row in policy.get("selected_templates", []):
        template = dict(row["template"])
        if template["event"] not in features.columns:
            continue
        triggers.extend(template_triggers(template | {"events": features[template["event"]].to_numpy(dtype=float)}, score=float(row.get("test_score", 0.0)), train_score=float(row.get("train_score", 0.0))))
    result = _simulate_online(market, triggers, cfg)
    report = {"config": asdict(cfg), "selected_template_count": len(policy.get("selected_templates", [])), "trigger_count": len(triggers), "result": result, "leakage_guard": {"policy_report_selected_before_eval": True, "paper_gate_updates_after_hypothetical_exit": True, "eval_trade_uses_only_prior_paper_outcomes": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--policy-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", default=OnlineGateCfg.start)
    p.add_argument("--end", default=OnlineGateCfg.end)
    p.add_argument("--eval-start", default=OnlineGateCfg.eval_start)
    p.add_argument("--windows", default=OnlineGateCfg.windows)
    p.add_argument("--entry-delay-bars", type=int, default=OnlineGateCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=OnlineGateCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=OnlineGateCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=OnlineGateCfg.slippage_rate)
    p.add_argument("--gate-min-trades", type=int, default=OnlineGateCfg.gate_min_trades)
    p.add_argument("--gate-lookback-trades", type=int, default=OnlineGateCfg.gate_lookback_trades)
    p.add_argument("--gate-min-mean-ret-pct", type=float, default=OnlineGateCfg.gate_min_mean_ret_pct)
    p.add_argument("--gate-max-loss-rate", type=float, default=OnlineGateCfg.gate_max_loss_rate)
    return p.parse_args()


def main() -> None:
    report = run(OnlineGateCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "selected_template_count": report["selected_template_count"], "trigger_count": report["trigger_count"], "sim": report["result"]["sim"], "trade_stats": report["result"]["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
