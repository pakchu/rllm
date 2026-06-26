"""Leakage-safe price-action episode policy scan.

This is a stricter replacement for broad symbolic gate tuning.  It represents
trades as semantic price-action episodes with fixed causal interpretation, e.g.
prior-range breakout continuation or sweep rejection reversal.  Selection uses
train and test only; eval is reported after the template set is fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.price_action_event_scan import build_price_action_event_features
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class EpisodePolicyCfg:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    horizons: str = "36,72,144,288,432"
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_train_trades: int = 30
    min_test_trades: int = 20
    min_train_ratio: float = 0.0
    min_test_ratio: float = 0.5
    max_test_mdd_pct: float = 25.0
    max_test_p_value: float = 0.40
    max_templates: int = 8
    max_trigger_overlap: float = 0.80
    top_k_report: int = 50


EPISODE_SIDES: dict[str, tuple[str, str]] = {
    "lower_high_mid_reject": ("SHORT", "bearish_structure_reject"),
    "lower_low_mid_fail": ("SHORT", "bearish_structure_continuation"),
    "downtrend_pullback_reject": ("SHORT", "downtrend_pullback_reject"),
    "failed_mid_reclaim_short": ("SHORT", "failed_reclaim_short"),
    "higher_low_mid_reclaim": ("LONG", "bullish_structure_reclaim"),
    "higher_high_mid_hold": ("LONG", "bullish_structure_continuation"),
    "uptrend_pullback_reclaim": ("LONG", "uptrend_pullback_reclaim"),
    "failed_mid_loss_long": ("LONG", "failed_loss_long"),
    "break_above": ("LONG", "breakout_continuation"),
    "break_above_with_volume": ("LONG", "breakout_continuation_volume"),
    "break_below": ("SHORT", "breakdown_continuation"),
    "break_below_with_volume": ("SHORT", "breakdown_continuation_volume"),
    "high_sweep_reject": ("SHORT", "liquidity_sweep_reversal"),
    "high_sweep_reject_with_volume": ("SHORT", "liquidity_sweep_reversal_volume"),
    "low_sweep_reclaim": ("LONG", "liquidity_sweep_reversal"),
    "low_sweep_reclaim_with_volume": ("LONG", "liquidity_sweep_reversal_volume"),
    "failed_breakout_short": ("SHORT", "failed_breakout_reversal"),
    "failed_breakdown_long": ("LONG", "failed_breakdown_reversal"),
    "reclaim_mid_from_below": ("LONG", "range_mid_reclaim"),
    "reject_mid_from_above": ("SHORT", "range_mid_reject"),
}


def build_episode_event_features(market: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Build base PAE plus causal structure-transition episode events.

    The additional events compare current OHLC against shifted prior ranges and
    compare the prior range to an older prior range.  No future bar participates
    in either the range level or structure-slope calculation.
    """
    base = build_price_action_event_features(market, windows)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    open_ = market["open"].astype(float)
    close = market["close"].astype(float)
    extra: dict[str, np.ndarray] = {}
    for w in windows:
        w = int(w)
        lag = max(2, w // 4)
        prior_high = high.shift(1).rolling(w, min_periods=w).max()
        prior_low = low.shift(1).rolling(w, min_periods=w).min()
        prior_mid = (prior_high + prior_low) / 2.0
        older_high = prior_high.shift(lag)
        older_low = prior_low.shift(lag)
        prior_range = (prior_high - prior_low).replace(0.0, np.nan)
        valid = prior_high.notna() & prior_low.notna() & older_high.notna() & older_low.notna() & (prior_range > 0)
        lower_high = valid & (prior_high < older_high)
        lower_low = valid & (prior_low < older_low)
        higher_high = valid & (prior_high > older_high)
        higher_low = valid & (prior_low > older_low)
        bearish_body = close < open_
        bullish_body = close > open_
        prefix = f"pae_w{w}"
        extra[f"{prefix}_lower_high_mid_reject"] = (lower_high & (high >= prior_mid) & (close < prior_mid) & bearish_body).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_lower_low_mid_fail"] = (lower_low & (close < prior_mid) & bearish_body).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_downtrend_pullback_reject"] = (lower_high & lower_low & (high >= prior_mid) & (close < open_) & (close < prior_high)).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_failed_mid_reclaim_short"] = (lower_high & (open_ < prior_mid) & (high > prior_mid) & (close < prior_mid)).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_higher_low_mid_reclaim"] = (higher_low & (low <= prior_mid) & (close > prior_mid) & bullish_body).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_higher_high_mid_hold"] = (higher_high & (close > prior_mid) & bullish_body).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_uptrend_pullback_reclaim"] = (higher_high & higher_low & (low <= prior_mid) & (close > open_) & (close > prior_low)).astype(float).to_numpy(dtype=float)
        extra[f"{prefix}_failed_mid_loss_long"] = (higher_low & (open_ > prior_mid) & (low < prior_mid) & (close > prior_mid)).astype(float).to_numpy(dtype=float)
    if not extra:
        return base
    return pd.concat([base, pd.DataFrame(extra, index=market.index)], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _period_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _years(dates: pd.Series, idxs: np.ndarray) -> float:
    if len(idxs) == 0:
        return 1.0 / 365.25
    start_dt = pd.Timestamp(dates.iloc[int(idxs[0])]).to_pydatetime()
    end_dt = pd.Timestamp(dates.iloc[int(idxs[-1])]).to_pydatetime()
    return max(1.0 / 365.25, float((end_dt - start_dt).days) / 365.25)


def _empty_sim(dates: pd.Series, idxs: np.ndarray) -> dict[str, Any]:
    return {
        "period": {"start": str(dates.iloc[int(idxs[0])]) if len(idxs) else None, "end": str(dates.iloc[int(idxs[-1])]) if len(idxs) else None, "years": _years(dates, idxs)},
        "sim": {"ret_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "trade_entries": 0, "side_counts": {"LONG": 0, "SHORT": 0}, "samples": int(len(idxs)), "skipped_missing_bars": 0, "return_application": "episode_policy_actual_ohlc_bar_by_bar_strict_mdd"},
        "trade_stats": _trade_stats([]),
        "executed": [],
    }


def simulate_triggers(market: pd.DataFrame, dates: pd.Series, triggers: list[dict[str, Any]], *, start: str, end: str, cfg: EpisodePolicyCfg) -> dict[str, Any]:
    mask = _period_mask(dates, start, end)
    period_idxs = np.flatnonzero(mask)
    if len(period_idxs) == 0:
        return _empty_sim(dates, period_idxs)
    by_pos: dict[int, list[dict[str, Any]]] = {}
    for trigger in triggers:
        pos = int(trigger["pos"])
        if mask[pos]:
            by_pos.setdefault(pos, []).append(trigger)

    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    exec_cfg = BarExecutionConfig(leverage=float(cfg.leverage), fee_rate=float(cfg.fee_rate), slippage_rate=float(cfg.slippage_rate), drawdown_stop=1.0, pause_bars=0, monthly_loss_stop=1.0, entry_delay_bars=int(cfg.entry_delay_bars))
    cost = (float(exec_cfg.fee_rate) + float(exec_cfg.slippage_rate)) * float(exec_cfg.leverage)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    executed: list[dict[str, Any]] = []
    skipped = 0
    for pos in period_idxs:
        pos = int(pos)
        if pos < next_allowed or pos not in by_pos:
            continue
        trigger = max(by_pos[pos], key=lambda r: (float(r.get("score", 0.0)), float(r.get("train_score", 0.0))))
        side = str(trigger["side"])
        signal = 1 if side == "LONG" else -1 if side == "SHORT" else 0
        hold_bars = max(1, int(trigger["horizon"]))
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + hold_bars
        if signal == 0 or entry_pos >= len(market) - 1 or exit_pos >= len(market):
            skipped += 1
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
            adverse_eq = eq * (1.0 + float(cfg.leverage) * adverse_ret)
            max_dd = max(max_dd, _drawdown_from_trough(peak, adverse_eq))
            eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        trade_ret = eq / entry_eq - 1.0
        trade_returns.append(trade_ret)
        executed.append({"date": str(dates.iloc[pos]), "signal_pos": pos, "side": side, "hold_bars": hold_bars, "episode": trigger.get("episode"), "event": trigger.get("event"), "trade_ret_pct": trade_ret * 100.0, "equity": eq})
        next_allowed = exit_pos
        if eq <= 0.0:
            break
    years = _years(dates, period_idxs)
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "period": {"start": str(dates.iloc[int(period_idxs[0])]), "end": str(dates.iloc[int(period_idxs[-1])]), "years": years},
        "sim": {"ret_pct": ret_pct, "cagr_pct": cagr_pct, "strict_mdd_pct": mdd_pct, "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"), "trade_entries": len(trade_returns), "side_counts": side_counts, "samples": int(len(period_idxs)), "skipped_missing_bars": skipped, "return_application": "episode_policy_actual_ohlc_bar_by_bar_strict_mdd"},
        "trade_stats": _trade_stats(trade_returns),
        "executed": executed,
    }


def _template_score(bt: dict[str, Any]) -> float:
    sim = bt["sim"]
    stats = bt["trade_stats"]
    return float(sim["cagr_to_strict_mdd"]) + 0.01 * float(sim["cagr_pct"]) + min(1.0, float(sim["trade_entries"]) / 100.0) - float(stats.get("p_value_mean_ret_approx", 1.0))


def build_templates(features: pd.DataFrame, windows: list[int], horizons: list[int]) -> list[dict[str, Any]]:
    templates = []
    for w in windows:
        for suffix, (side, episode) in EPISODE_SIDES.items():
            col = f"pae_w{int(w)}_{suffix}"
            if col not in features.columns or float(features[col].sum()) <= 0.0:
                continue
            events = features[col].to_numpy(dtype=float)
            for horizon in horizons:
                templates.append({"event": col, "window": int(w), "event_type": suffix, "episode": episode, "side": side, "horizon": int(horizon), "events": events})
    return templates


def template_triggers(template: dict[str, Any], score: float = 0.0, train_score: float = 0.0) -> list[dict[str, Any]]:
    idxs = np.flatnonzero(np.asarray(template["events"], dtype=float) > 0.5)
    return [{k: template[k] for k in ("event", "window", "event_type", "episode", "side", "horizon")} | {"pos": int(pos), "score": float(score), "train_score": float(train_score)} for pos in idxs]


def run(cfg: EpisodePolicyCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    horizons = _parse_list(cfg.horizons, int)
    features = build_episode_event_features(market, windows)
    candidates = []
    for template in build_templates(features, windows, horizons):
        triggers = template_triggers(template)
        train_bt = simulate_triggers(market, dates, triggers, start=cfg.train_start, end=cfg.train_end, cfg=cfg)
        test_bt = simulate_triggers(market, dates, triggers, start=cfg.test_start, end=cfg.test_end, cfg=cfg)
        eval_bt = simulate_triggers(market, dates, triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=cfg)
        train_score = _template_score(train_bt)
        test_score = _template_score(test_bt)
        reject = []
        if int(train_bt["sim"]["trade_entries"]) < int(cfg.min_train_trades):
            reject.append("train_trades_below_min")
        if int(test_bt["sim"]["trade_entries"]) < int(cfg.min_test_trades):
            reject.append("test_trades_below_min")
        if float(train_bt["sim"]["cagr_to_strict_mdd"]) < float(cfg.min_train_ratio):
            reject.append("train_ratio_below_min")
        if float(test_bt["sim"]["cagr_to_strict_mdd"]) < float(cfg.min_test_ratio):
            reject.append("test_ratio_below_min")
        if float(test_bt["sim"]["strict_mdd_pct"]) > float(cfg.max_test_mdd_pct):
            reject.append("test_mdd_above_max")
        if float(test_bt["trade_stats"].get("p_value_mean_ret_approx", 1.0) or 1.0) > float(cfg.max_test_p_value):
            reject.append("test_p_value_above_max")
        public_template = {k: template[k] for k in ("event", "window", "event_type", "episode", "side", "horizon")}
        candidates.append({"template": public_template, "train": {"sim": train_bt["sim"], "trade_stats": train_bt["trade_stats"]}, "test": {"sim": test_bt["sim"], "trade_stats": test_bt["trade_stats"]}, "eval_diagnostic": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]}, "train_score": train_score, "test_score": test_score, "validation_passed": not reject, "reject_reasons": reject})
    ranked = sorted(candidates, key=lambda r: (r["validation_passed"], float(r["test_score"]), float(r["train_score"]), int(r["test"]["sim"]["trade_entries"])), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_triggers: list[dict[str, Any]] = []
    rejected_portfolio: list[dict[str, Any]] = []
    selected_position_sets: list[set[int]] = []
    for r in [x for x in ranked if x["validation_passed"]]:
        if len(selected) >= int(cfg.max_templates):
            break
        t = dict(r["template"])
        event_values = features[t["event"]].to_numpy(dtype=float)
        candidate_triggers = template_triggers(t | {"events": event_values}, score=float(r["test_score"]), train_score=float(r["train_score"]))
        candidate_positions = {int(x["pos"]) for x in candidate_triggers}
        if candidate_positions and any(len(candidate_positions & prev) / max(1, len(candidate_positions | prev)) > float(cfg.max_trigger_overlap) for prev in selected_position_sets):
            rejected_portfolio.append({"template": t, "reject_reasons": ["trigger_overlap_above_max"]})
            continue
        trial_triggers = selected_triggers + candidate_triggers
        trial_train = simulate_triggers(market, dates, trial_triggers, start=cfg.train_start, end=cfg.train_end, cfg=cfg)
        trial_test = simulate_triggers(market, dates, trial_triggers, start=cfg.test_start, end=cfg.test_end, cfg=cfg)
        reject = []
        if int(trial_train["sim"]["trade_entries"]) < int(cfg.min_train_trades):
            reject.append("portfolio_train_trades_below_min")
        if int(trial_test["sim"]["trade_entries"]) < int(cfg.min_test_trades):
            reject.append("portfolio_test_trades_below_min")
        if float(trial_train["sim"]["cagr_to_strict_mdd"]) < float(cfg.min_train_ratio):
            reject.append("portfolio_train_ratio_below_min")
        if float(trial_test["sim"]["cagr_to_strict_mdd"]) < float(cfg.min_test_ratio):
            reject.append("portfolio_test_ratio_below_min")
        if float(trial_test["sim"]["strict_mdd_pct"]) > float(cfg.max_test_mdd_pct):
            reject.append("portfolio_test_mdd_above_max")
        if float(trial_test["trade_stats"].get("p_value_mean_ret_approx", 1.0) or 1.0) > float(cfg.max_test_p_value):
            reject.append("portfolio_test_p_value_above_max")
        if reject:
            rejected_portfolio.append({"template": t, "reject_reasons": reject, "trial_train": {"sim": trial_train["sim"], "trade_stats": trial_train["trade_stats"]}, "trial_test": {"sim": trial_test["sim"], "trade_stats": trial_test["trade_stats"]}})
            continue
        nr = dict(r)
        nr["portfolio_after_add"] = {"train": {"sim": trial_train["sim"], "trade_stats": trial_train["trade_stats"]}, "test": {"sim": trial_test["sim"], "trade_stats": trial_test["trade_stats"]}}
        selected.append(nr)
        selected_triggers = trial_triggers
        selected_position_sets.append(candidate_positions)
    portfolio = {
        "train": simulate_triggers(market, dates, selected_triggers, start=cfg.train_start, end=cfg.train_end, cfg=cfg),
        "test": simulate_triggers(market, dates, selected_triggers, start=cfg.test_start, end=cfg.test_end, cfg=cfg),
        "eval": simulate_triggers(market, dates, selected_triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=cfg),
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "templates_scanned": len(candidates),
        "selected_templates": selected,
        "portfolio": {k: {"period": v["period"], "sim": v["sim"], "trade_stats": v["trade_stats"], "executed_sample": v["executed"][:20]} for k, v in portfolio.items()},
        "portfolio_rejected_candidates": rejected_portfolio[: int(cfg.top_k_report)],
        "top": ranked[: int(cfg.top_k_report)],
        "selection_protocol": "semantic episode sides are fixed a priori; template inclusion uses train+test only; eval is untouched holdout after selected set is fixed",
        "leakage_guard": {"prior_range_uses_shifted_rolling_levels": True, "features_use_rows_at_or_before_t": True, "template_selection_uses_eval": False, "strict_mdd_includes_intrabar_adverse_excursion": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(EpisodePolicyCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=EpisodePolicyCfg.windows)
    p.add_argument("--horizons", default=EpisodePolicyCfg.horizons)
    p.add_argument("--entry-delay-bars", type=int, default=EpisodePolicyCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=EpisodePolicyCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=EpisodePolicyCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=EpisodePolicyCfg.slippage_rate)
    p.add_argument("--min-train-trades", type=int, default=EpisodePolicyCfg.min_train_trades)
    p.add_argument("--min-test-trades", type=int, default=EpisodePolicyCfg.min_test_trades)
    p.add_argument("--min-train-ratio", type=float, default=EpisodePolicyCfg.min_train_ratio)
    p.add_argument("--min-test-ratio", type=float, default=EpisodePolicyCfg.min_test_ratio)
    p.add_argument("--max-test-mdd-pct", type=float, default=EpisodePolicyCfg.max_test_mdd_pct)
    p.add_argument("--max-test-p-value", type=float, default=EpisodePolicyCfg.max_test_p_value)
    p.add_argument("--max-templates", type=int, default=EpisodePolicyCfg.max_templates)
    p.add_argument("--max-trigger-overlap", type=float, default=EpisodePolicyCfg.max_trigger_overlap)
    p.add_argument("--top-k-report", type=int, default=EpisodePolicyCfg.top_k_report)
    return p.parse_args()


def main() -> None:
    report = run(EpisodePolicyCfg(**vars(parse_args())))
    print(json.dumps({
        "output": report["config"]["output"],
        "templates_scanned": report["templates_scanned"],
        "selected_count": len(report["selected_templates"]),
        "selected_templates": [r["template"] | {"test": r["test"]["sim"], "test_p": r["test"]["trade_stats"].get("p_value_mean_ret_approx")} for r in report["selected_templates"]],
        "portfolio": {k: v["sim"] | {"p": v["trade_stats"].get("p_value_mean_ret_approx")} for k, v in report["portfolio"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
