"""Mine causal text contexts for single-policy RLLM training.

This is not a backtest and not a deployable gate.  It converts causal market,
external, and derivatives features into compact textual state/action examples so
Gemma-style policies can learn regularized regime abstractions instead of raw
numeric threshold chasing.

Leakage boundaries:
- Bucket cut points are fit on the train split only.
- Prompts contain only current/past causal feature buckets.
- Targets may use future path returns, but only as supervised labels.
- Eval rows can be exported for frozen-model evaluation, but never for context
  selection or label-threshold tuning.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.sparse_setup_ensemble_audit import _load_market
from training.price_action_event_scan import build_price_action_event_features

ACTIONS = {"NO_TRADE", "LONG", "SHORT"}


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


@dataclass(frozen=True)
class LlmContextRegimeMinerCfg:
    market_csv: str
    output: str
    summary_output: str = ""
    sample_output: str = ""
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    binance_funding_csv: str = ""
    binance_premium_csv: str = ""
    binance_funding_tolerance: str = "12h"
    binance_premium_tolerance: str = "2h"
    train_start: str = "2020-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    window_size: int = 144
    horizon: int = 288
    stride_bars: int = 72
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_trade_net_pct: float = 0.25
    min_trade_edge_gap_pct: float = 0.15
    max_trade_mae_pct: float = 7.0
    max_rows: int = 0
    include_price_action_events: bool = True
    price_action_event_windows: str = "36,72,144,288,576,2016"


FEATURES: tuple[str, ...] = (
    "dxy_zscore",
    "dxy_momentum",
    "kimchi_premium_zscore",
    "kimchi_premium_change",
    "usdkrw_zscore",
    "usdkrw_momentum",
    "funding_zscore",
    "premium_index_zscore",
    "premium_index_change",
    "trend_12",
    "trend_96",
    "trend_288",
    "sma24_ratio",
    "bb_z",
    "rsi_norm",
    "mfi_norm",
    "range_pos",
    "window_drawdown",
    "volume_zscore",
    "taker_imbalance",
)


def _load_market_with_features(cfg: LlmContextRegimeMinerCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    if bool(cfg.include_price_action_events):
        event_features = build_price_action_event_features(market, _parse_int_list(cfg.price_action_event_windows))
        features = pd.concat([features, event_features], axis=1)
        features = features.loc[:, ~features.columns.duplicated(keep="last")]
    for col in FEATURES + ("external_any_available", "binance_aux_any_available"):
        if col not in features.columns:
            features[col] = 0.0
    return market, features


def _split(date: pd.Timestamp, cfg: LlmContextRegimeMinerCfg) -> str | None:
    if pd.Timestamp(cfg.train_start) <= date <= pd.Timestamp(cfg.train_end):
        return "train"
    if pd.Timestamp(cfg.test_start) <= date <= pd.Timestamp(cfg.test_end):
        return "test"
    if pd.Timestamp(cfg.eval_start) <= date <= pd.Timestamp(cfg.eval_end):
        return "eval"
    return None


def _fit_bucket_edges(features: pd.DataFrame, dates: pd.Series, cfg: LlmContextRegimeMinerCfg, feature_names: tuple[str, ...] = FEATURES) -> dict[str, list[float]]:
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    edges: dict[str, list[float]] = {}
    for name in feature_names:
        vals = pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)
        train_vals = vals[train_mask & np.isfinite(vals)]
        if len(train_vals) < 100:
            edges[name] = [-np.inf, np.inf]
            continue
        qs = np.quantile(train_vals, [0.10, 0.30, 0.50, 0.70, 0.90]).astype(float)
        # Deduplicate nearly constant columns while preserving deterministic labels.
        uniq = []
        for q in qs:
            if not uniq or abs(float(q) - float(uniq[-1])) > 1e-12:
                uniq.append(float(q))
        edges[name] = uniq
    return edges


def _bucket_value(value: float, edges: list[float]) -> str:
    if not np.isfinite(value):
        return "missing"
    labels = ["very_low", "low", "mid_low", "mid_high", "high", "very_high"]
    idx = int(np.searchsorted(np.asarray(edges, dtype=float), float(value), side="right"))
    return labels[min(max(idx, 0), len(labels) - 1)]


def _event_active(features: pd.DataFrame, pos: int, window: int, names: tuple[str, ...]) -> bool:
    for name in names:
        col = f"pae_w{int(window)}_{name}"
        if col in features.columns and float(features[col].iloc[pos]) > 0.5:
            return True
    return False


def _price_action_event_tokens(features: pd.DataFrame, pos: int) -> dict[str, str]:
    windows = sorted({int(c.split("_")[1][1:]) for c in features.columns if c.startswith("pae_w") and len(c.split("_")) > 1})
    if not windows:
        return {
            "pa_event_pressure": "missing",
            "pa_downside_reclaim": "missing",
            "pa_upside_rejection": "missing",
            "pa_long_window_event": "missing",
        }
    downside_names = ("break_below", "failed_breakdown_long", "low_sweep_reclaim", "low_sweep_reclaim_with_volume")
    upside_names = ("break_above", "failed_breakout_short", "high_sweep_reject", "high_sweep_reject_with_volume")
    downside = [w for w in windows if _event_active(features, pos, w, downside_names)]
    upside = [w for w in windows if _event_active(features, pos, w, upside_names)]
    low_sweep = [w for w in windows if _event_active(features, pos, w, ("low_sweep_reclaim", "low_sweep_reclaim_with_volume", "failed_breakdown_long"))]
    high_sweep = [w for w in windows if _event_active(features, pos, w, ("high_sweep_reject", "high_sweep_reject_with_volume", "failed_breakout_short"))]
    long_ws = [w for w in windows if w >= 576]
    long_down = any(w in downside for w in long_ws)
    long_up = any(w in upside for w in long_ws)
    if downside and upside:
        pressure = "two_sided_range_stress"
    elif downside:
        pressure = "downside_break_or_reclaim"
    elif upside:
        pressure = "upside_break_or_reject"
    else:
        pressure = "no_major_event"
    if long_down and long_up:
        long_event = "long_window_two_sided"
    elif long_down:
        long_event = "long_window_downside_reclaim_candidate"
    elif long_up:
        long_event = "long_window_upside_rejection_candidate"
    else:
        long_event = "none"
    return {
        "pa_event_pressure": pressure,
        "pa_downside_reclaim": f"active_w{max(low_sweep)}" if low_sweep else "inactive",
        "pa_upside_rejection": f"active_w{max(high_sweep)}" if high_sweep else "inactive",
        "pa_long_window_event": long_event,
    }


def _state_tokens(features: pd.DataFrame, pos: int, edges: dict[str, list[float]]) -> dict[str, str]:
    tokens = {name: _bucket_value(float(features[name].iloc[pos]), edges[name]) for name in FEATURES}
    tokens["external_availability"] = "available" if float(features["external_any_available"].iloc[pos]) > 0.5 else "missing_or_partial"
    tokens["binance_aux_availability"] = "available" if float(features["binance_aux_any_available"].iloc[pos]) > 0.5 else "missing_or_partial"
    tokens["trend_alignment"] = _trend_alignment(tokens.get("trend_96", "missing"), tokens.get("trend_288", "missing"))
    tokens["risk_state"] = _risk_state(tokens.get("window_drawdown", "missing"), tokens.get("range_pos", "missing"))
    tokens.update(_price_action_event_tokens(features, pos))
    return tokens


def _trend_alignment(short_trend: str, long_trend: str) -> str:
    up = {"mid_high", "high", "very_high"}
    down = {"very_low", "low", "mid_low"}
    if short_trend in up and long_trend in up:
        return "aligned_up"
    if short_trend in down and long_trend in down:
        return "aligned_down"
    return "mixed"


def _risk_state(drawdown: str, range_pos: str) -> str:
    if drawdown in {"high", "very_high"} and range_pos in {"very_low", "low"}:
        return "deep_drawdown_near_lows"
    if drawdown in {"very_low", "low"} and range_pos in {"high", "very_high"}:
        return "extended_near_highs"
    return "normal"


def _path_audit(market: pd.DataFrame, pos: int, side: str, cfg: LlmContextRegimeMinerCfg) -> dict[str, float] | None:
    if side not in {"LONG", "SHORT"}:
        return None
    entry = int(pos) + int(cfg.entry_delay_bars)
    exit_pos = entry + int(cfg.horizon)
    if entry >= len(market) or exit_pos >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    entry_price = float(opens[entry])
    exit_price = float(opens[exit_pos])
    if entry_price <= 0.0 or exit_price <= 0.0:
        return None
    sign = 1.0 if side == "LONG" else -1.0
    raw = sign * (exit_price / entry_price - 1.0)
    path_high = np.asarray(highs[entry : exit_pos + 1], dtype=float)
    path_low = np.asarray(lows[entry : exit_pos + 1], dtype=float)
    if side == "LONG":
        mae = float(max(0.0, 1.0 - np.min(path_low / entry_price)))
        mfe = float(np.max(path_high / entry_price - 1.0))
    else:
        mae = float(max(0.0, np.max(path_high / entry_price) - 1.0))
        mfe = float(np.max(1.0 - path_low / entry_price))
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
    net = raw * float(cfg.leverage) - cost
    return {"net_return_pct": net * 100.0, "mae_pct": mae * float(cfg.leverage) * 100.0, "mfe_pct": mfe * float(cfg.leverage) * 100.0}


def _target(market: pd.DataFrame, pos: int, cfg: LlmContextRegimeMinerCfg) -> tuple[dict[str, Any], dict[str, Any]]:
    long_audit = _path_audit(market, pos, "LONG", cfg)
    short_audit = _path_audit(market, pos, "SHORT", cfg)
    if long_audit is None or short_audit is None:
        return _no_trade("path_unavailable"), {}
    long_net = float(long_audit["net_return_pct"])
    short_net = float(short_audit["net_return_pct"])
    if long_net >= short_net:
        side, best, other = "LONG", long_audit, short_audit
    else:
        side, best, other = "SHORT", short_audit, long_audit
    edge_gap = float(best["net_return_pct"] - other["net_return_pct"])
    if float(best["net_return_pct"]) >= float(cfg.min_trade_net_pct) and edge_gap >= float(cfg.min_trade_edge_gap_pct) and float(best["mae_pct"]) <= float(cfg.max_trade_mae_pct):
        target = {
            "action": side,
            "confidence": "HIGH" if float(best["net_return_pct"]) >= 1.0 and float(best["mae_pct"]) <= 4.0 else "MEDIUM",
            "reason_code": "future_label_best_side_reward_ok",
            "hold_bars": int(cfg.horizon),
        }
    else:
        target = _no_trade("future_label_edge_or_risk_rejected")
    return target, {"LONG": long_audit, "SHORT": short_audit, "edge_gap_pct": edge_gap}


def _no_trade(reason: str) -> dict[str, Any]:
    return {"action": "NO_TRADE", "confidence": "LOW", "reason_code": reason, "hold_bars": 0}


def _prompt(date: str, tokens: dict[str, str], cfg: LlmContextRegimeMinerCfg) -> str:
    lines = [
        "You are a single compact BTCUSDT futures RLLM policy.",
        "Use only causal text state buckets; do not infer from future returns.",
        "Return one JSON object with keys: action, confidence, reason_code, hold_bars.",
        "Allowed action: NO_TRADE, LONG, SHORT. Prefer NO_TRADE unless state edge is clear.",
        "No raw prices, thresholds, PnL, or exchange orders.",
        "",
        f"date: {date}",
        f"decision_horizon: enter next 5m open; max_hold_bars={int(cfg.horizon)}",
        "causal_state_tokens:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    lines.append("Policy intent: act only when causal context supports a side and expected path risk is acceptable; otherwise abstain.")
    return "\n".join(lines)


def build_rows(cfg: LlmContextRegimeMinerCfg) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    market, features = _load_market_with_features(cfg)
    dates = pd.to_datetime(market["date"])
    edges = _fit_bucket_edges(features, dates, cfg)
    max_pos = len(market) - int(cfg.entry_delay_bars) - int(cfg.horizon) - 1
    rows: list[dict[str, Any]] = []
    for pos in range(max(int(cfg.window_size), 1), max_pos, max(1, int(cfg.stride_bars))):
        date = pd.Timestamp(dates.iloc[pos])
        split = _split(date, cfg)
        if split is None:
            continue
        tokens = _state_tokens(features, pos, edges)
        target, audit = _target(market, pos, cfg)
        rows.append({
            "task": "llm_context_regime_policy_sft",
            "split": split,
            "date": str(dates.iloc[pos]),
            "signal_pos": int(pos),
            "prompt": _prompt(str(dates.iloc[pos]), tokens, cfg),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "state_tokens": tokens,
            "reward_audit": audit,
            "leakage_guard": {
                "prompt_uses_future_path": False,
                "target_uses_future_path_for_training_only": True,
                "bucket_edges_fit_train_only": True,
                "eval_not_for_selection": split == "eval",
                "not_analyzer_trader_cascade": True,
            },
        })
        if int(cfg.max_rows) > 0 and len(rows) >= int(cfg.max_rows):
            break
    return rows, edges


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summarize(rows: list[dict[str, Any]], edges: dict[str, list[float]], cfg: LlmContextRegimeMinerCfg) -> dict[str, Any]:
    split_counts = Counter(str(r["split"]) for r in rows)
    action_counts: dict[str, Counter[str]] = {}
    prompt_lens = []
    for r in rows:
        split = str(r["split"])
        action_counts.setdefault(split, Counter())[json.loads(str(r["target"])).get("action", "NO_TRADE")] += 1
        prompt_lens.append(len(str(r.get("prompt", ""))))
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts_by_split": {k: dict(sorted(v.items())) for k, v in sorted(action_counts.items())},
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "bucket_features": list(edges.keys()),
        "config": asdict(cfg),
        "leakage_guard": {
            "prompts_are_past_only": True,
            "bucket_edges_fit_train_only": True,
            "targets_use_future_path_for_training_only": True,
            "not_a_backtest_result": True,
            "active_rllm_path": "single_policy_context_abstraction",
            "price_action_event_tokens_are_causal": bool(cfg.include_price_action_events),
        },
    }


def run(cfg: LlmContextRegimeMinerCfg) -> dict[str, Any]:
    rows, edges = build_rows(cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(20, len(rows))])
    summary = _summarize(rows, edges, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine causal text contexts for single-policy RLLM SFT")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--sample-output", default="")
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=LlmContextRegimeMinerCfg.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=LlmContextRegimeMinerCfg.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=LlmContextRegimeMinerCfg.binance_premium_tolerance)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(LlmContextRegimeMinerCfg, name.replace("-", "_")))
    p.add_argument("--window-size", type=int, default=LlmContextRegimeMinerCfg.window_size)
    p.add_argument("--horizon", type=int, default=LlmContextRegimeMinerCfg.horizon)
    p.add_argument("--stride-bars", type=int, default=LlmContextRegimeMinerCfg.stride_bars)
    p.add_argument("--entry-delay-bars", type=int, default=LlmContextRegimeMinerCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=LlmContextRegimeMinerCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=LlmContextRegimeMinerCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LlmContextRegimeMinerCfg.slippage_rate)
    p.add_argument("--min-trade-net-pct", type=float, default=LlmContextRegimeMinerCfg.min_trade_net_pct)
    p.add_argument("--min-trade-edge-gap-pct", type=float, default=LlmContextRegimeMinerCfg.min_trade_edge_gap_pct)
    p.add_argument("--max-trade-mae-pct", type=float, default=LlmContextRegimeMinerCfg.max_trade_mae_pct)
    p.add_argument("--max-rows", type=int, default=LlmContextRegimeMinerCfg.max_rows)
    p.add_argument("--include-price-action-events", action=argparse.BooleanOptionalAction, default=LlmContextRegimeMinerCfg.include_price_action_events)
    p.add_argument("--price-action-event-windows", default=LlmContextRegimeMinerCfg.price_action_event_windows)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(LlmContextRegimeMinerCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
