"""Build text-state action-value rows for LLM/RL trading experiments.

The goal is to avoid feeding the LLM raw numeric tables.  Prompts describe a
past-only market state as compact categorical tokens, then ask for the value of a
single candidate action.  Future path metrics are labels/metadata only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.sparse_setup_ensemble_audit import _load_market


@dataclass(frozen=True)
class TextStateActionValueCfg:
    market_csv: str
    output: str
    summary_output: str
    sample_output: str = ""
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-01"
    train_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    window_size: int = 144
    stride_bars: int = 576
    hold_bars_list: str = "72,144,288"
    sides: str = "LONG,SHORT"
    entry_delay_bars: int = 1
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    take_min_net_return_pct: float = 1.5
    max_mae_pct: float = 4.0
    min_mfe_to_mae: float = 0.7
    abstain_deadzone_pct: float = 0.35
    max_rows: int = 0


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_sides(raw: str) -> list[str]:
    sides = [x.strip().upper() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in sides if x not in {"LONG", "SHORT"}]
    if bad:
        raise ValueError(f"unsupported sides: {bad}")
    return sides


def _bucket_signed(x: float, *, small: float, large: float) -> str:
    x = float(x)
    if x <= -large:
        return "strong_down"
    if x <= -small:
        return "down"
    if x < small:
        return "flat"
    if x < large:
        return "up"
    return "strong_up"


def _bucket_unit(x: float, *, low: float = -0.35, high: float = 0.35) -> str:
    x = float(x)
    if x <= low:
        return "low_zone"
    if x >= high:
        return "high_zone"
    return "middle_zone"


def _bucket_abs(x: float, *, low: float, high: float) -> str:
    ax = abs(float(x))
    if ax < low:
        return "low"
    if ax < high:
        return "medium"
    return "high"


def _bucket_z(x: float) -> str:
    return _bucket_signed(float(x), small=0.75, large=1.75)


def _safe(features: pd.DataFrame, pos: int, col: str) -> float:
    if col not in features.columns or pos < 0 or pos >= len(features):
        return 0.0
    val = features.iloc[pos][col]
    try:
        if pd.isna(val):
            return 0.0
    except TypeError:
        return 0.0
    return float(val)


def _state_tokens(features: pd.DataFrame, pos: int, side: str) -> dict[str, str]:
    sign = 1.0 if side == "LONG" else -1.0
    t12 = _safe(features, pos, "trend_12")
    t24 = _safe(features, pos, "trend_24")
    t96 = _safe(features, pos, "trend_96")
    h4 = _safe(features, pos, "htf_4h_return_4")
    d1 = _safe(features, pos, "htf_1d_return_4")
    d3 = _safe(features, pos, "htf_3d_return_4")
    w1 = _safe(features, pos, "htf_1w_return_4")
    return {
        "short_trend": _bucket_signed(t12, small=0.003, large=0.010),
        "intraday_trend": _bucket_signed(t24, small=0.004, large=0.014),
        "session_trend": _bucket_signed(t96, small=0.008, large=0.026),
        "four_hour_context": _bucket_signed(h4, small=0.008, large=0.030),
        "daily_context": _bucket_signed(d1, small=0.012, large=0.045),
        "three_day_context": _bucket_signed(d3, small=0.020, large=0.070),
        "weekly_context": _bucket_signed(w1, small=0.030, large=0.100),
        "candidate_alignment_short": _bucket_signed(sign * t12, small=0.003, large=0.010),
        "candidate_alignment_daily": _bucket_signed(sign * d1, small=0.012, large=0.045),
        "range_location": _bucket_unit(_safe(features, pos, "range_pos")),
        "htf_4h_location": _bucket_unit(_safe(features, pos, "htf_4h_range_pos")),
        "htf_1d_location": _bucket_unit(_safe(features, pos, "htf_1d_range_pos")),
        "oscillator_pressure": _bucket_signed(_safe(features, pos, "rsi_norm"), small=0.20, large=0.55),
        "money_flow_pressure": _bucket_signed(_safe(features, pos, "mfi_norm"), small=0.20, large=0.55),
        "volatility": _bucket_abs(_safe(features, pos, "range_vol"), low=0.020, high=0.060),
        "window_drawdown": _bucket_abs(_safe(features, pos, "window_drawdown"), low=0.020, high=0.080),
        "volume_pressure": _bucket_z(_safe(features, pos, "volume_zscore")),
        "taker_imbalance": _bucket_signed(_safe(features, pos, "taker_imbalance"), small=0.04, large=0.12),
        "funding_pressure": _bucket_z(_safe(features, pos, "funding_zscore")),
        "open_interest_pressure": _bucket_z(_safe(features, pos, "oi_zscore")),
        "dxy_pressure": _bucket_signed(_safe(features, pos, "dxy_momentum"), small=0.001, large=0.004),
        "kimchi_pressure": _bucket_z(_safe(features, pos, "kimchi_premium_zscore")),
        "usdkrw_pressure": _bucket_signed(_safe(features, pos, "usdkrw_momentum"), small=0.001, large=0.004),
        "external_availability": "available" if _safe(features, pos, "external_any_available") > 0.5 else "missing_or_partial",
    }


def _path_audit(market: pd.DataFrame, pos: int, *, side: str, hold_bars: int, cfg: TextStateActionValueCfg) -> dict[str, float] | None:
    entry = int(pos) + int(cfg.entry_delay_bars)
    exit_pos = entry + int(hold_bars)
    if entry >= len(market) or exit_pos >= len(market):
        return None
    opens = market["open"].to_numpy(dtype=float)
    highs = market["high"].to_numpy(dtype=float)
    lows = market["low"].to_numpy(dtype=float)
    entry_price = float(opens[entry])
    exit_price = float(opens[exit_pos])
    if entry_price <= 0 or exit_price <= 0:
        return None
    sign = 1.0 if side == "LONG" else -1.0
    raw_ret = sign * (exit_price / entry_price - 1.0)
    path_high = np.asarray(highs[entry:exit_pos + 1], dtype=float)
    path_low = np.asarray(lows[entry:exit_pos + 1], dtype=float)
    if side == "LONG":
        mfe = float(np.max(path_high / entry_price - 1.0))
        mae = float(max(0.0, 1.0 - np.min(path_low / entry_price)))
    else:
        mfe = float(np.max(1.0 - path_low / entry_price))
        mae = float(max(0.0, np.max(path_high / entry_price) - 1.0))
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
    net = raw_ret * float(cfg.leverage) - cost
    return {
        "raw_return_pct": raw_ret * 100.0,
        "net_return_pct": net * 100.0,
        "mae_pct": mae * float(cfg.leverage) * 100.0,
        "mfe_pct": mfe * float(cfg.leverage) * 100.0,
        "mfe_to_mae": float(mfe / max(mae, 1e-12)),
    }


def _label(audit: dict[str, float], cfg: TextStateActionValueCfg) -> str:
    net = float(audit["net_return_pct"])
    mae = float(audit["mae_pct"])
    ratio = float(audit["mfe_to_mae"])
    if net >= float(cfg.take_min_net_return_pct) and mae <= float(cfg.max_mae_pct) and ratio >= float(cfg.min_mfe_to_mae):
        return "TAKE"
    if abs(net) <= float(cfg.abstain_deadzone_pct):
        return "ABSTAIN"
    return "SKIP"


def _prompt(*, date: str, side: str, hold_bars: int, tokens: dict[str, str]) -> str:
    lines = [
        "You are a BTCUSDT futures action-value judge.",
        "Use only the categorical past-state tokens. Do not use hidden numeric prices or future outcomes.",
        "Decide whether this single candidate action deserves risk capital.",
        "Output exactly one label: TAKE, SKIP, or ABSTAIN.",
        "",
        f"Date: {date}",
        f"Candidate: side={side}; hold_bars={int(hold_bars)}; entry=next_5m_open; fees=included_in_label.",
        "Past-state tokens:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    return "\n".join(lines)


def _load_market_with_external(cfg: TextStateActionValueCfg) -> pd.DataFrame:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    return market


def _split(date: pd.Timestamp, cfg: TextStateActionValueCfg) -> str | None:
    if date < pd.Timestamp(cfg.start_date) or date > pd.Timestamp(cfg.end_date):
        return None
    if date <= pd.Timestamp(cfg.train_end):
        return "train"
    if date >= pd.Timestamp(cfg.eval_start):
        return "eval"
    return None


def build_rows(cfg: TextStateActionValueCfg) -> list[dict[str, Any]]:
    market = _load_market_with_external(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    for col in EXTENDED_MARKET_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0
    dates = pd.to_datetime(market["date"])
    hold_bars = _parse_ints(cfg.hold_bars_list)
    sides = _parse_sides(cfg.sides)
    max_hold = max(hold_bars)
    rows: list[dict[str, Any]] = []
    start_pos = max(int(cfg.window_size), 1)
    end_pos = len(market) - int(cfg.entry_delay_bars) - max_hold - 1
    for pos in range(start_pos, max(start_pos, end_pos), max(1, int(cfg.stride_bars))):
        split = _split(pd.Timestamp(dates.iloc[pos]), cfg)
        if split is None:
            continue
        for side in sides:
            tokens = _state_tokens(features, pos, side)
            for hold in hold_bars:
                audit = _path_audit(market, pos, side=side, hold_bars=int(hold), cfg=cfg)
                if audit is None:
                    continue
                label = _label(audit, cfg)
                rows.append({
                    "task": "text_state_action_value",
                    "split": split,
                    "date": str(dates.iloc[pos]),
                    "signal_pos": int(pos),
                    "candidate": {"side": side, "hold_bars": int(hold)},
                    "prompt": _prompt(date=str(dates.iloc[pos]), side=side, hold_bars=int(hold), tokens=tokens),
                    "target": label,
                    "state_tokens": tokens,
                    "reward_audit": audit,
                    "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_for_training_only": True, "features_signal_time_or_prior": True},
                })
                if int(cfg.max_rows) > 0 and len(rows) >= int(cfg.max_rows):
                    return rows
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]], cfg: TextStateActionValueCfg) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for r in rows:
        by_split.setdefault(str(r["split"]), []).append(r)
    def one(xs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(r["target"]) for r in xs)
        side_counts = Counter(str(r["candidate"]["side"]) for r in xs)
        hold_counts = Counter(str(r["candidate"]["hold_bars"]) for r in xs)
        rets = np.asarray([float(r["reward_audit"]["net_return_pct"]) for r in xs], dtype=float) if xs else np.asarray([], dtype=float)
        prompt_lens = [len(str(r["prompt"])) for r in xs]
        return {
            "rows": len(xs),
            "period": {"start": xs[0]["date"] if xs else None, "end": xs[-1]["date"] if xs else None},
            "target_counts": dict(sorted(counts.items())),
            "side_counts": dict(sorted(side_counts.items())),
            "hold_counts": dict(sorted(hold_counts.items())),
            "target_rate": {k: v / max(1, len(xs)) for k, v in sorted(counts.items())},
            "net_return_pct": {"mean": float(np.mean(rets)) if len(rets) else 0.0, "std": float(np.std(rets)) if len(rets) else 0.0},
            "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        }
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "output": cfg.output,
        "sample_output": cfg.sample_output,
        "total_rows": len(rows),
        "train": one(by_split.get("train", [])),
        "eval": one(by_split.get("eval", [])),
        "prompt_contract": "categorical past-state tokens only; future path appears only in target/reward_audit",
        "leakage_guard": {"features_signal_time_or_prior": True, "external_join_backward_asof": bool(cfg.wave_trading_root), "eval_split_not_used_for_train": True},
    }


def run(cfg: TextStateActionValueCfg) -> dict[str, Any]:
    rows = build_rows(cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(200, len(rows))])
    summary = _summary(rows, cfg)
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build categorical text-state action-value rows")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--sample-output", default="")
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=TextStateActionValueCfg.external_tolerance)
    p.add_argument("--start-date", default=TextStateActionValueCfg.start_date)
    p.add_argument("--end-date", default=TextStateActionValueCfg.end_date)
    p.add_argument("--train-end", default=TextStateActionValueCfg.train_end)
    p.add_argument("--eval-start", default=TextStateActionValueCfg.eval_start)
    p.add_argument("--window-size", type=int, default=TextStateActionValueCfg.window_size)
    p.add_argument("--stride-bars", type=int, default=TextStateActionValueCfg.stride_bars)
    p.add_argument("--hold-bars-list", default=TextStateActionValueCfg.hold_bars_list)
    p.add_argument("--sides", default=TextStateActionValueCfg.sides)
    p.add_argument("--entry-delay-bars", type=int, default=TextStateActionValueCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=TextStateActionValueCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=TextStateActionValueCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=TextStateActionValueCfg.slippage_rate)
    p.add_argument("--take-min-net-return-pct", type=float, default=TextStateActionValueCfg.take_min_net_return_pct)
    p.add_argument("--max-mae-pct", type=float, default=TextStateActionValueCfg.max_mae_pct)
    p.add_argument("--min-mfe-to-mae", type=float, default=TextStateActionValueCfg.min_mfe_to_mae)
    p.add_argument("--abstain-deadzone-pct", type=float, default=TextStateActionValueCfg.abstain_deadzone_pct)
    p.add_argument("--max-rows", type=int, default=TextStateActionValueCfg.max_rows)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(TextStateActionValueCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
