"""Regime-gated episode template policy.

This tests whether the episode-template failures are caused by trading the same
signals in the wrong market regime.  Selected templates are fixed from an
existing report, then (side, regime-key) buckets are selected on train/test only.
Eval trades are allowed only if their current past-only regime key was selected.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.price_action_episode_policy import add_sequence_context_features, build_episode_event_features, template_triggers
from training.strict_bar_backtest import BarExecutionConfig, _drawdown_from_trough, _trade_stats


@dataclass(frozen=True)
class RegimeGatedEpisodeCfg:
    input_csv: str
    policy_report: str
    output: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    regime_fields: str = "side,trend_phase,vol_phase,macro_phase,drawdown_phase"
    min_train_trades: int = 10
    min_test_trades: int = 5
    min_train_mean_ret_pct: float = 0.0
    min_test_mean_ret_pct: float = 0.0
    max_test_loss_rate: float = 0.65
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    entry_delay_bars: int = 1


def _bucket_signed(v: float, lo: float, hi: float) -> str:
    if v <= -abs(lo):
        return "down"
    if v >= abs(hi):
        return "up"
    return "flat"


def _bucket_abs(v: float, low: float, high: float) -> str:
    x = abs(float(v))
    if x < low:
        return "low"
    if x < high:
        return "mid"
    return "high"


def build_regime_frame(market: pd.DataFrame) -> pd.DataFrame:
    feats = build_market_feature_frame(market, window_size=144)
    out = pd.DataFrame(index=market.index)
    trend = 0.35 * feats.get("trend_96", 0.0) + 0.25 * feats.get("htf_4h_return_4", 0.0) + 0.25 * feats.get("htf_1d_return_1", 0.0) + 0.15 * feats.get("htf_3d_return_1", 0.0)
    vol = feats.get("range_vol", 0.0) + 0.5 * np.abs(feats.get("return_zscore_48", 0.0)) + 0.25 * np.maximum(0.0, feats.get("volume_zscore", 0.0))
    macro = feats.get("dxy_zscore", 0.0) + 0.5 * feats.get("usdkrw_zscore", 0.0) - 0.25 * feats.get("kimchi_premium_zscore", 0.0)
    stress = feats.get("funding_zscore", 0.0) + feats.get("oi_change", 0.0) + 0.25 * feats.get("oi_zscore", 0.0)
    drawdown = feats.get("window_drawdown", 0.0) + 0.5 * feats.get("htf_1d_drawdown_4", 0.0) + 0.25 * feats.get("htf_1w_drawdown_4", 0.0)
    range_pos = feats.get("range_pos", 0.5)
    out["trend_phase"] = [_bucket_signed(float(x), 0.004, 0.004) for x in trend]
    out["vol_phase"] = [_bucket_abs(float(x), 0.25, 1.25) for x in vol]
    out["macro_phase"] = np.select([macro + 0.5 * stress > 0.75, macro + 0.5 * stress < -0.75], ["risk_off", "risk_on"], default="neutral")
    out["drawdown_phase"] = pd.cut(drawdown.astype(float), bins=[-1e9, 0.01, 0.035, 1e9], labels=["dd_low", "dd_mid", "dd_high"]).astype(str)
    out["location_phase"] = pd.cut(range_pos.astype(float), bins=[-1e9, 0.25, 0.75, 1e9], labels=["range_low", "range_mid", "range_high"]).astype(str)
    return out.fillna("unknown")


def _period_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _trade_return(market: pd.DataFrame, pos: int, side: str, hold_bars: int, cfg: RegimeGatedEpisodeCfg) -> tuple[float | None, int | None]:
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
        close_ret = (float(opens[j + 1]) - open_j) / open_j if signal > 0 else (open_j - float(opens[j + 1])) / open_j
        eq *= max(0.0, 1.0 + float(cfg.leverage) * close_ret)
    eq *= max(0.0, 1.0 - cost)
    return eq - 1.0, exit_pos


def _key(trigger: dict[str, Any], regimes: pd.DataFrame, pos: int, fields: list[str]) -> tuple[str, ...]:
    parts = []
    for f in fields:
        if f == "side":
            parts.append(str(trigger.get("side")))
        elif f == "episode":
            parts.append(str(trigger.get("episode")))
        else:
            parts.append(str(regimes.iloc[int(pos)].get(f, "missing")))
    return tuple(parts)


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean_ret_pct": 0.0, "loss_rate": 0.0, "trade_stats": _trade_stats([])}
    return {"n": len(vals), "mean_ret_pct": float(np.mean(vals)) * 100.0, "loss_rate": sum(v <= 0 for v in vals) / len(vals), "trade_stats": _trade_stats(vals)}


def _simulate(market: pd.DataFrame, dates: pd.Series, triggers: list[dict[str, Any]], regimes: pd.DataFrame, selected_keys: set[tuple[str, ...]], fields: list[str], cfg: RegimeGatedEpisodeCfg) -> dict[str, Any]:
    mask = _period_mask(dates, cfg.eval_start, cfg.eval_end)
    by_pos: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in triggers:
        pos = int(t["pos"])
        if mask[pos] and _key(t, regimes, pos, fields) in selected_keys:
            by_pos[pos].append(t)
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    side_counts = {"LONG": 0, "SHORT": 0}
    executed: list[dict[str, Any]] = []
    for pos in sorted(by_pos):
        if pos < next_allowed:
            continue
        chosen = max(by_pos[pos], key=lambda r: (float(r.get("score", 0.0)), float(r.get("train_score", 0.0))))
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
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, _drawdown_from_trough(peak, eq))
        peak = max(peak, eq)
        ret = eq / entry_eq - 1.0
        trade_returns.append(ret)
        executed.append({"date": str(dates.iloc[pos]), "signal_pos": pos, "side": side, "hold_bars": hold_bars, "key": list(_key(chosen, regimes, pos, fields)), "trade_ret_pct": ret * 100.0, "equity": eq})
        next_allowed = exit_pos
    eval_dates = dates[mask]
    years = max(1.0 / 365.25, (pd.Timestamp(eval_dates.iloc[-1]) - pd.Timestamp(eval_dates.iloc[0])).days / 365.25) if len(eval_dates) else 1.0 / 365.25
    ret_pct = (eq - 1.0) * 100.0
    gross = 1.0 + ret_pct / 100.0
    cagr_pct = ((gross ** (1.0 / years) - 1.0) * 100.0) if gross > 0 else -100.0
    mdd_pct = max_dd * 100.0
    return {"sim": {"ret_pct": ret_pct, "cagr_pct": cagr_pct, "strict_mdd_pct": mdd_pct, "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else 0.0, "trade_entries": len(trade_returns), "side_counts": side_counts, "return_application": "regime_gated_episode_actual_ohlc_bar_by_bar_strict_mdd"}, "trade_stats": _trade_stats(trade_returns), "executed_sample": executed[:100]}


def run(cfg: RegimeGatedEpisodeCfg) -> dict[str, Any]:
    policy = json.loads(Path(cfg.policy_report).read_text())
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    fields = [x.strip() for x in cfg.regime_fields.split(",") if x.strip()]
    feats = build_episode_event_features(market, windows)
    feats = add_sequence_context_features(market, feats, windows)
    regimes = build_regime_frame(market)
    triggers: list[dict[str, Any]] = []
    for row in policy.get("selected_templates", []):
        t = dict(row["template"])
        if t["event"] not in feats.columns:
            continue
        triggers.extend(template_triggers(t | {"events": feats[t["event"]].to_numpy(dtype=float)}, score=float(row.get("test_score", 0.0)), train_score=float(row.get("train_score", 0.0))))
    train_mask = _period_mask(dates, cfg.train_start, cfg.train_end)
    test_mask = _period_mask(dates, cfg.test_start, cfg.test_end)
    train_buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    test_buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for t in triggers:
        pos = int(t["pos"])
        ret, _ = _trade_return(market, pos, str(t["side"]), int(t["horizon"]), cfg)
        if ret is None:
            continue
        k = _key(t, regimes, pos, fields)
        if train_mask[pos]:
            train_buckets[k].append(ret)
        elif test_mask[pos]:
            test_buckets[k].append(ret)
    selected = []
    selected_keys: set[tuple[str, ...]] = set()
    all_keys = set(train_buckets) | set(test_buckets)
    for k in sorted(all_keys):
        tr = _stats(train_buckets.get(k, []))
        te = _stats(test_buckets.get(k, []))
        reject = []
        if tr["n"] < int(cfg.min_train_trades): reject.append("train_trades_below_min")
        if te["n"] < int(cfg.min_test_trades): reject.append("test_trades_below_min")
        if tr["mean_ret_pct"] < float(cfg.min_train_mean_ret_pct): reject.append("train_mean_below_min")
        if te["mean_ret_pct"] < float(cfg.min_test_mean_ret_pct): reject.append("test_mean_below_min")
        if te["loss_rate"] > float(cfg.max_test_loss_rate): reject.append("test_loss_rate_above_max")
        item = {"key": list(k), "train": tr, "test": te, "validation_passed": not reject, "reject_reasons": reject}
        selected.append(item)
        if not reject:
            selected_keys.add(k)
    result = _simulate(market, dates, triggers, regimes, selected_keys, fields, cfg)
    report = {"config": asdict(cfg), "trigger_count": len(triggers), "selected_key_count": len(selected_keys), "selected_keys": [x for x in selected if x["validation_passed"]], "top_rejected": [x for x in selected if not x["validation_passed"]][:50], "result": result, "leakage_guard": {"regime_features_use_rows_at_or_before_signal": True, "regime_key_selection_uses_train_test_only": True, "eval_not_used_for_key_selection": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--policy-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=RegimeGatedEpisodeCfg.train_start)
    p.add_argument("--train-end", default=RegimeGatedEpisodeCfg.train_end)
    p.add_argument("--test-start", default=RegimeGatedEpisodeCfg.test_start)
    p.add_argument("--test-end", default=RegimeGatedEpisodeCfg.test_end)
    p.add_argument("--eval-start", default=RegimeGatedEpisodeCfg.eval_start)
    p.add_argument("--eval-end", default=RegimeGatedEpisodeCfg.eval_end)
    p.add_argument("--windows", default=RegimeGatedEpisodeCfg.windows)
    p.add_argument("--regime-fields", default=RegimeGatedEpisodeCfg.regime_fields)
    p.add_argument("--min-train-trades", type=int, default=RegimeGatedEpisodeCfg.min_train_trades)
    p.add_argument("--min-test-trades", type=int, default=RegimeGatedEpisodeCfg.min_test_trades)
    p.add_argument("--min-train-mean-ret-pct", type=float, default=RegimeGatedEpisodeCfg.min_train_mean_ret_pct)
    p.add_argument("--min-test-mean-ret-pct", type=float, default=RegimeGatedEpisodeCfg.min_test_mean_ret_pct)
    p.add_argument("--max-test-loss-rate", type=float, default=RegimeGatedEpisodeCfg.max_test_loss_rate)
    p.add_argument("--leverage", type=float, default=RegimeGatedEpisodeCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=RegimeGatedEpisodeCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RegimeGatedEpisodeCfg.slippage_rate)
    p.add_argument("--entry-delay-bars", type=int, default=RegimeGatedEpisodeCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    report = run(RegimeGatedEpisodeCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "trigger_count": report["trigger_count"], "selected_key_count": report["selected_key_count"], "sim": report["result"]["sim"], "trade_stats": report["result"]["trade_stats"], "selected_keys": report["selected_keys"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
