"""Build RLLM text-policy rows for the DXY-low / Kimchi regime prior.

The dataset is intentionally narrow.  Recent alpha work found that broad scans
are mostly unstable, while DXY-low regimes with Kimchi z-score signals are the
first non-oracle family with positive test/eval but below the target ratio.

This builder converts that family into a single compact policy surface:
- prompt: causal text state + train-fitted rule-prior description;
- target: JSON policy deciding activate/abstain for the prior signal;
- labels: future path utility for supervised training only.

No future values are emitted in the prompt.  DXY/Kimchi thresholds and rule side
mapping are fit on the train window only.
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
from training.alpha_feature_backtest import _forward_return, fit_rule
from training.sparse_setup_ensemble_audit import _load_market
from training.text_state_action_value_dataset import _bucket_abs, _bucket_signed, _bucket_unit, _bucket_z, _safe

ACTIONS = {"NO_TRADE", "LONG", "SHORT"}
EXIT_PROFILES = {"AVOID", "FAST", "NORMAL"}


@dataclass(frozen=True)
class DxyKimchiPolicyDatasetCfg:
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
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:59:59"
    test_start: str = "2024-07-01"
    test_end: str = "2025-08-31 23:59:59"
    eval_start: str = "2025-09-01"
    eval_end: str = "2026-05-31 15:00:00"
    window_size: int = 144
    horizon: int = 144
    stride_bars: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    signal_quantile: float = 0.2
    regime_quantile: float = 0.33
    min_activate_net_pct: float = 0.20
    max_activate_mae_pct: float = 6.0
    max_rows: int = 0


def _load_market_with_features(cfg: DxyKimchiPolicyDatasetCfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(
            market,
            wave_trading_root=cfg.wave_trading_root,
            tolerance=cfg.external_tolerance,
        )
    if cfg.binance_funding_csv or cfg.binance_premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.binance_funding_csv or None,
            premium_csv=cfg.binance_premium_csv or None,
            funding_tolerance=cfg.binance_funding_tolerance,
            premium_tolerance=cfg.binance_premium_tolerance,
        )
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    return market, features


def _split(date: pd.Timestamp, cfg: DxyKimchiPolicyDatasetCfg) -> str | None:
    if pd.Timestamp(cfg.train_start) <= date <= pd.Timestamp(cfg.train_end):
        return "train"
    if pd.Timestamp(cfg.test_start) <= date <= pd.Timestamp(cfg.test_end):
        return "test"
    if pd.Timestamp(cfg.eval_start) <= date <= pd.Timestamp(cfg.eval_end):
        return "eval"
    return None


def _signal_for_value(value: float, rule: dict[str, Any]) -> str:
    if not np.isfinite(value):
        return "NONE"
    if value >= float(rule["high_threshold"]):
        return str(rule["high_side"])
    if value <= float(rule["low_threshold"]):
        return str(rule["low_side"])
    return "NONE"


def _path_audit(market: pd.DataFrame, pos: int, side: str, cfg: DxyKimchiPolicyDatasetCfg) -> dict[str, float] | None:
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
        mfe = float(np.max(path_high / entry_price - 1.0))
        mae = float(max(0.0, 1.0 - np.min(path_low / entry_price)))
    else:
        mfe = float(np.max(1.0 - path_low / entry_price))
        mae = float(max(0.0, np.max(path_high / entry_price) - 1.0))
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
    net = raw * float(cfg.leverage) - cost
    return {
        "net_return_pct": net * 100.0,
        "mae_pct": mae * float(cfg.leverage) * 100.0,
        "mfe_pct": mfe * float(cfg.leverage) * 100.0,
        "raw_return_pct": raw * 100.0,
    }


def _exit_profile(horizon: int) -> str:
    return "FAST" if int(horizon) <= 144 else "NORMAL"


def _target(*, prior_side: str, audit: dict[str, float] | None, cfg: DxyKimchiPolicyDatasetCfg) -> dict[str, Any]:
    if prior_side not in {"LONG", "SHORT"} or audit is None:
        return {
            "activate": False,
            "action": "NO_TRADE",
            "exit_profile": "AVOID",
            "confidence": "LOW",
            "reason_code": "no_prior_signal",
        }
    net = float(audit["net_return_pct"])
    mae = float(audit["mae_pct"])
    if net >= float(cfg.min_activate_net_pct) and mae <= float(cfg.max_activate_mae_pct):
        conf = "HIGH" if net >= 1.0 and mae <= 3.5 else "MEDIUM"
        return {
            "activate": True,
            "action": prior_side,
            "exit_profile": _exit_profile(int(cfg.horizon)),
            "confidence": conf,
            "reason_code": "prior_signal_path_reward_ok",
        }
    return {
        "activate": False,
        "action": "NO_TRADE",
        "exit_profile": "AVOID",
        "confidence": "LOW",
        "reason_code": "prior_signal_path_reward_rejected",
    }


def _state_tokens(features: pd.DataFrame, pos: int) -> dict[str, str]:
    return {
        "dxy_zscore_bucket": _bucket_z(_safe(features, pos, "dxy_zscore")),
        "dxy_momentum": _bucket_signed(_safe(features, pos, "dxy_momentum"), small=0.001, large=0.004),
        "kimchi_zscore_bucket": _bucket_z(_safe(features, pos, "kimchi_premium_zscore")),
        "kimchi_change": _bucket_signed(_safe(features, pos, "kimchi_premium_change"), small=0.001, large=0.004),
        "funding_pressure": _bucket_z(_safe(features, pos, "funding_zscore")),
        "premium_index_pressure": _bucket_z(_safe(features, pos, "premium_index_zscore")),
        "session_trend": _bucket_signed(_safe(features, pos, "trend_96"), small=0.008, large=0.026),
        "weekly_context": _bucket_signed(_safe(features, pos, "htf_1w_return_4"), small=0.030, large=0.100),
        "window_drawdown": _bucket_abs(_safe(features, pos, "window_drawdown"), low=0.020, high=0.080),
        "range_location": _bucket_unit(_safe(features, pos, "range_pos")),
        "taker_imbalance": _bucket_signed(_safe(features, pos, "taker_imbalance"), small=0.04, large=0.12),
        "external_availability": "available" if _safe(features, pos, "external_any_available") > 0.5 else "missing_or_partial",
        "binance_aux_availability": "available" if _safe(features, pos, "binance_aux_any_available") > 0.5 else "missing_or_partial",
    }


def _fmt(x: float) -> str:
    return f"{float(x):+.3f}"


def _threshold_distance_bucket(value: float, threshold: float, *, direction: str) -> str:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return "missing"
    signed = (float(threshold) - float(value)) if direction == "below" else (float(value) - float(threshold))
    if signed < 0.0:
        return "not_triggered"
    if signed < 0.25:
        return "near"
    if signed < 0.75:
        return "medium"
    return "deep"


def _kimchi_signal_strength_bucket(kimchi_value: float, prior_side: str, rule: dict[str, Any]) -> str:
    if prior_side not in {"LONG", "SHORT"} or not np.isfinite(kimchi_value):
        return "none"
    if prior_side == str(rule.get("high_side")):
        return _threshold_distance_bucket(kimchi_value, float(rule["high_threshold"]), direction="above")
    if prior_side == str(rule.get("low_side")):
        return _threshold_distance_bucket(kimchi_value, float(rule["low_threshold"]), direction="below")
    return "none"


def _trend_alignment(prior_side: str, trend_token: str) -> str:
    if prior_side not in {"LONG", "SHORT"}:
        return "no_prior"
    trend = str(trend_token)
    if trend in {"flat", "missing"}:
        return "flat_or_missing"
    if prior_side == "LONG":
        return "aligned" if trend in {"up", "strong_up"} else "opposed"
    return "aligned" if trend in {"down", "strong_down"} else "opposed"


def _prompt(*, date: str, tokens: dict[str, str], prior_side: str, dxy_value: float, kimchi_value: float, rule: dict[str, Any], dxy_low_threshold: float, cfg: DxyKimchiPolicyDatasetCfg) -> str:
    lines = [
        "You are a compact BTCUSDT futures RLLM policy.",
        "Use only the causal state and the train-fitted prior below.",
        "Decide whether to activate the DXY-low / Kimchi-z prior signal or abstain.",
        "Return one compact JSON object with keys: activate, action, exit_profile, confidence, reason_code.",
        "Allowed action: NO_TRADE, LONG, SHORT. If activate=false, action must be NO_TRADE and exit_profile must be AVOID.",
        "Do not output raw thresholds, prices, returns, or exchange orders.",
        "",
        f"date: {date}",
        f"decision_horizon: enter next 5m open; hold_bars={int(cfg.horizon)}.",
        "train_fitted_prior:",
        f"- prior_family: dxy_low_kimchi_zscore",
        f"- dxy_regime_now: {'LOW_BUCKET' if dxy_value <= dxy_low_threshold else 'NOT_LOW_BUCKET'}",
        f"- kimchi_prior_signal: {prior_side}",
        f"- prior_direction_mapping: high_kimchi={rule['high_side']}; low_kimchi={rule['low_side']}",
        f"- dxy_low_depth_bucket: {_threshold_distance_bucket(dxy_value, dxy_low_threshold, direction='below')}",
        f"- kimchi_signal_strength_bucket: {_kimchi_signal_strength_bucket(kimchi_value, prior_side, rule)}",
        f"- prior_side_trend_alignment: {_trend_alignment(prior_side, tokens.get('session_trend', 'missing'))}",
        "causal_state_tokens:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    lines.extend([
        "causal_numeric_context_rounded:",
        f"- dxy_zscore_now={_fmt(dxy_value)}",
        f"- kimchi_zscore_now={_fmt(kimchi_value)}",
        "Policy intent: activate only when current context supports the prior and path risk is acceptable; otherwise abstain.",
    ])
    return "\n".join(lines)


def _fit_prior_rule(features: pd.DataFrame, market: pd.DataFrame, dates: pd.Series, cfg: DxyKimchiPolicyDatasetCfg) -> dict[str, Any]:
    train_mask = np.asarray((dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)), dtype=bool)
    dxy = features["dxy_zscore"].to_numpy(dtype=float)
    kimchi = features["kimchi_premium_zscore"].to_numpy(dtype=float)
    train_dxy = dxy[train_mask & np.isfinite(dxy)]
    if len(train_dxy) < 100:
        raise ValueError("not enough train rows for DXY regime threshold")
    dxy_low_threshold = float(np.quantile(train_dxy, float(cfg.regime_quantile)))
    regime_mask = dxy <= dxy_low_threshold
    gated_kimchi = np.where(regime_mask, kimchi, np.nan)
    fwd = _forward_return(market["open"].astype(float), horizon=int(cfg.horizon), entry_delay_bars=int(cfg.entry_delay_bars))
    tmp_cfg = argparse.Namespace(
        fit_start=cfg.train_start,
        fit_end=cfg.train_end,
        quantile=float(cfg.signal_quantile),
    )
    rule = fit_rule(dates=dates, feature_values=gated_kimchi, forward_returns=fwd, cfg=tmp_cfg)  # type: ignore[arg-type]
    return {"dxy_low_threshold": dxy_low_threshold, "kimchi_rule": rule}


def build_rows(cfg: DxyKimchiPolicyDatasetCfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    market, features = _load_market_with_features(cfg)
    for col in ("dxy_zscore", "kimchi_premium_zscore", "kimchi_premium_change", "funding_zscore", "premium_index_zscore", "binance_aux_any_available", "external_any_available"):
        if col not in features.columns:
            features[col] = 0.0
    dates = pd.to_datetime(market["date"])
    prior = _fit_prior_rule(features, market, dates, cfg)
    dxy_low_threshold = float(prior["dxy_low_threshold"])
    rule = dict(prior["kimchi_rule"])
    dxy = features["dxy_zscore"].to_numpy(dtype=float)
    kimchi = features["kimchi_premium_zscore"].to_numpy(dtype=float)
    max_pos = len(market) - int(cfg.entry_delay_bars) - int(cfg.horizon) - 1
    rows: list[dict[str, Any]] = []
    for pos in range(max(int(cfg.window_size), 1), max_pos, max(1, int(cfg.stride_bars))):
        split = _split(pd.Timestamp(dates.iloc[pos]), cfg)
        if split is None:
            continue
        dxy_value = float(dxy[pos])
        kimchi_value = float(kimchi[pos])
        regime_ok = bool(np.isfinite(dxy_value) and dxy_value <= dxy_low_threshold)
        prior_side = _signal_for_value(kimchi_value, rule) if regime_ok else "NONE"
        audit = _path_audit(market, pos, prior_side, cfg)
        target = _target(prior_side=prior_side, audit=audit, cfg=cfg)
        tokens = _state_tokens(features, pos)
        rows.append(
            {
                "task": "dxy_kimchi_regime_policy_sft",
                "split": split,
                "date": str(dates.iloc[pos]),
                "signal_pos": int(pos),
                "prompt": _prompt(date=str(dates.iloc[pos]), tokens=tokens, prior_side=prior_side, dxy_value=dxy_value, kimchi_value=kimchi_value, rule=rule, dxy_low_threshold=dxy_low_threshold, cfg=cfg),
                "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "state_tokens": tokens,
                "prior_signal": {"regime_ok": regime_ok, "side": prior_side, "horizon": int(cfg.horizon)},
                "reward_audit": audit or {},
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_path_for_training_only": True,
                    "dxy_threshold_fit_train_only": True,
                    "kimchi_rule_fit_train_only": True,
                    "not_analyzer_trader_cascade": True,
                },
            }
        )
        if int(cfg.max_rows) > 0 and len(rows) >= int(cfg.max_rows):
            break
    return rows, prior


def _summarize(rows: list[dict[str, Any]], prior: dict[str, Any], cfg: DxyKimchiPolicyDatasetCfg) -> dict[str, Any]:
    split_counts = Counter(str(r.get("split")) for r in rows)
    target_counts: dict[str, Counter[str]] = {"action": Counter(), "activate": Counter(), "reason_code": Counter(), "confidence": Counter()}
    prior_counts = Counter(str(r.get("prior_signal", {}).get("side", "NONE")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    for r in rows:
        obj = json.loads(str(r["target"]))
        for k in target_counts:
            target_counts[k][str(obj.get(k))] += 1
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "target_counts": {k: dict(sorted(v.items())) for k, v in target_counts.items()},
        "prior_side_counts": dict(sorted(prior_counts.items())),
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "prior": prior,
        "config": asdict(cfg),
        "leakage_guard": {
            "prompts_are_past_only": True,
            "prior_thresholds_fit_train_only": True,
            "targets_use_future_path_for_training_only": True,
            "not_a_backtest_result": True,
            "active_rllm_path": "single_policy_no_analyzer_trader_cascade",
        },
    }


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def run(cfg: DxyKimchiPolicyDatasetCfg) -> dict[str, Any]:
    rows, prior = build_rows(cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(20, len(rows))])
    summary = _summarize(rows, prior, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DXY-low/Kimchi prior RLLM policy rows")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--sample-output", default="")
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=DxyKimchiPolicyDatasetCfg.external_tolerance)
    p.add_argument("--binance-funding-csv", default="")
    p.add_argument("--binance-premium-csv", default="")
    p.add_argument("--binance-funding-tolerance", default=DxyKimchiPolicyDatasetCfg.binance_funding_tolerance)
    p.add_argument("--binance-premium-tolerance", default=DxyKimchiPolicyDatasetCfg.binance_premium_tolerance)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(DxyKimchiPolicyDatasetCfg, name.replace("-", "_")))
    p.add_argument("--window-size", type=int, default=DxyKimchiPolicyDatasetCfg.window_size)
    p.add_argument("--horizon", type=int, default=DxyKimchiPolicyDatasetCfg.horizon)
    p.add_argument("--stride-bars", type=int, default=DxyKimchiPolicyDatasetCfg.stride_bars)
    p.add_argument("--entry-delay-bars", type=int, default=DxyKimchiPolicyDatasetCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=DxyKimchiPolicyDatasetCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=DxyKimchiPolicyDatasetCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=DxyKimchiPolicyDatasetCfg.slippage_rate)
    p.add_argument("--signal-quantile", type=float, default=DxyKimchiPolicyDatasetCfg.signal_quantile)
    p.add_argument("--regime-quantile", type=float, default=DxyKimchiPolicyDatasetCfg.regime_quantile)
    p.add_argument("--min-activate-net-pct", type=float, default=DxyKimchiPolicyDatasetCfg.min_activate_net_pct)
    p.add_argument("--max-activate-mae-pct", type=float, default=DxyKimchiPolicyDatasetCfg.max_activate_mae_pct)
    p.add_argument("--max-rows", type=int, default=DxyKimchiPolicyDatasetCfg.max_rows)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(DxyKimchiPolicyDatasetCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
