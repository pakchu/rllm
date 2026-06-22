"""Build portfolio-level text-state decision rows for LLM/RL trading.

Each row is one timestamp, not one candidate action.  The label chooses the best
single portfolio action among LONG, SHORT, or NO_TRADE using future path utility
for training only.  This avoids the previous failure mode where many candidate
TAKE labels clustered into an untradeable action stream.
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

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.sparse_setup_ensemble_audit import _load_market
from training.text_state_action_value_dataset import _bucket_abs, _bucket_signed, _bucket_unit, _bucket_z, _safe


@dataclass(frozen=True)
class PortfolioDecisionCfg:
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
    stride_bars: int = 288
    hold_bars: int = 288
    entry_delay_bars: int = 1
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_trade_net_return_pct: float = 1.2
    max_trade_mae_pct: float = 5.0
    min_advantage_pct: float = 0.6
    mae_penalty: float = 0.35
    no_trade_buffer_pct: float = 0.25
    max_rows: int = 0
    prompt_style: str = "categorical"


def _state_tokens(features: pd.DataFrame, pos: int) -> dict[str, str]:
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


def _path(market: pd.DataFrame, pos: int, side: str, cfg: PortfolioDecisionCfg) -> dict[str, float] | None:
    entry = int(pos) + int(cfg.entry_delay_bars)
    exit_pos = entry + int(cfg.hold_bars)
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
    raw = sign * (exit_price / entry_price - 1.0)
    hi = np.asarray(highs[entry:exit_pos + 1], dtype=float)
    lo = np.asarray(lows[entry:exit_pos + 1], dtype=float)
    if side == "LONG":
        mfe = float(np.max(hi / entry_price - 1.0))
        mae = float(max(0.0, 1.0 - np.min(lo / entry_price)))
    else:
        mfe = float(np.max(1.0 - lo / entry_price))
        mae = float(max(0.0, np.max(hi / entry_price) - 1.0))
    cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * 2.0 * float(cfg.leverage)
    net = raw * float(cfg.leverage) - cost
    return {
        "net_return_pct": net * 100.0,
        "mae_pct": mae * float(cfg.leverage) * 100.0,
        "mfe_pct": mfe * float(cfg.leverage) * 100.0,
        "utility": net * 100.0 - float(cfg.mae_penalty) * mae * float(cfg.leverage) * 100.0,
    }


def _choose(long: dict[str, float], short: dict[str, float], cfg: PortfolioDecisionCfg) -> str:
    best_side = "LONG" if long["utility"] >= short["utility"] else "SHORT"
    best = long if best_side == "LONG" else short
    other = short if best_side == "LONG" else long
    if best["net_return_pct"] < float(cfg.min_trade_net_return_pct):
        return "NO_TRADE"
    if best["mae_pct"] > float(cfg.max_trade_mae_pct):
        return "NO_TRADE"
    if best["utility"] - other["utility"] < float(cfg.min_advantage_pct):
        return "NO_TRADE"
    if best["utility"] < float(cfg.no_trade_buffer_pct):
        return "NO_TRADE"
    return best_side


def _fmt_pct(x: float) -> str:
    return f"{float(x) * 100.0:+.2f}%"


def _fmt_num(x: float) -> str:
    return f"{float(x):+.3f}"


def _feature_snapshot(features: pd.DataFrame, pos: int) -> dict[str, float]:
    cols = [
        "trend_12", "trend_24", "trend_96",
        "htf_4h_return_4", "htf_1d_return_4", "htf_3d_return_4", "htf_1w_return_4",
        "range_pos", "htf_4h_range_pos", "htf_1d_range_pos",
        "rsi_norm", "mfi_norm", "bb_z", "return_zscore_48", "close_zscore_48",
        "range_vol", "window_drawdown", "volume_zscore", "trades_ratio",
        "taker_buy_ratio", "taker_imbalance",
        "dxy_momentum", "dxy_zscore", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_momentum", "usdkrw_zscore",
        "external_any_available",
    ]
    return {c: _safe(features, pos, c) for c in cols}


def _rich_prompt(date: str, tokens: dict[str, str], snap: dict[str, float], cfg: PortfolioDecisionCfg) -> str:
    lines = [
        "You are a BTCUSDT futures portfolio decision policy.",
        "Use only past and current market evidence below; future path is not shown.",
        "Choose exactly one label: LONG, SHORT, or NO_TRADE.",
        "Prefer trades only when directional evidence is strong enough to overcome fees, slippage, and path drawdown risk.",
        "Avoid trading when signals conflict or volatility/drawdown risk dominates.",
        "",
        f"Date: {date}",
        f"Decision horizon: enter on next 5m open; hold_bars={int(cfg.hold_bars)}.",
        "",
        "Categorical regime summary:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    lines.extend([
        "",
        "Numeric evidence snapshot (all computed at or before decision time):",
        f"- price momentum: 1h={_fmt_pct(snap['trend_12'])}, 2h={_fmt_pct(snap['trend_24'])}, 8h={_fmt_pct(snap['trend_96'])}",
        f"- higher timeframe returns: 4h_ctx={_fmt_pct(snap['htf_4h_return_4'])}, 1d_ctx={_fmt_pct(snap['htf_1d_return_4'])}, 3d_ctx={_fmt_pct(snap['htf_3d_return_4'])}, 1w_ctx={_fmt_pct(snap['htf_1w_return_4'])}",
        f"- range location: short_window={_fmt_num(snap['range_pos'])}, 4h={_fmt_num(snap['htf_4h_range_pos'])}, 1d={_fmt_num(snap['htf_1d_range_pos'])} (-1 low, +1 high)",
        f"- oscillators: rsi_norm={_fmt_num(snap['rsi_norm'])}, mfi_norm={_fmt_num(snap['mfi_norm'])}, bb_z={_fmt_num(snap['bb_z'])}, return_z48={_fmt_num(snap['return_zscore_48'])}, close_z48={_fmt_num(snap['close_zscore_48'])}",
        f"- risk state: range_vol={_fmt_pct(snap['range_vol'])}, recent_window_drawdown={_fmt_pct(snap['window_drawdown'])}",
        f"- participation: volume_z={_fmt_num(snap['volume_zscore'])}, trades_ratio={_fmt_num(snap['trades_ratio'])}, taker_buy_ratio={_fmt_num(snap['taker_buy_ratio'])}, taker_imbalance={_fmt_num(snap['taker_imbalance'])}",
        f"- macro/external: dxy_mom={_fmt_pct(snap['dxy_momentum'])}, dxy_z={_fmt_num(snap['dxy_zscore'])}, kimchi_z={_fmt_num(snap['kimchi_premium_zscore'])}, kimchi_change={_fmt_num(snap['kimchi_premium_change'])}, usdkrw_mom={_fmt_pct(snap['usdkrw_momentum'])}, usdkrw_z={_fmt_num(snap['usdkrw_zscore'])}, external_available={bool(snap['external_any_available'] > 0.5)}",
        "",
        "Reasoning checklist:",
        "1. Decide whether risk-adjusted directional edge exists.",
        "2. If edge is weak or mixed, output NO_TRADE.",
        "3. If edge is directional, choose LONG or SHORT.",
        "Output exactly one label only.",
    ])
    return "\n".join(lines)


def _prompt(date: str, tokens: dict[str, str], cfg: PortfolioDecisionCfg) -> str:
    lines = [
        "You are a BTCUSDT futures portfolio decision policy.",
        "Use only the categorical past-state tokens. Choose one action for the next entry window.",
        "Output exactly one label: LONG, SHORT, or NO_TRADE.",
        "The label is trained to prefer net reward after path-risk, fees, and side advantage.",
        "",
        f"Date: {date}",
        f"Decision horizon: hold_bars={int(cfg.hold_bars)}; entry=next_5m_open.",
        "Past-state tokens:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    return "\n".join(lines)


def _load_market_with_external(cfg: PortfolioDecisionCfg) -> pd.DataFrame:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    return market


def _split(date: pd.Timestamp, cfg: PortfolioDecisionCfg) -> str | None:
    if date < pd.Timestamp(cfg.start_date) or date > pd.Timestamp(cfg.end_date):
        return None
    if date <= pd.Timestamp(cfg.train_end):
        return "train"
    if date >= pd.Timestamp(cfg.eval_start):
        return "eval"
    return None


def build_rows(cfg: PortfolioDecisionCfg) -> list[dict[str, Any]]:
    market = _load_market_with_external(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    for col in EXTENDED_MARKET_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0
    dates = pd.to_datetime(market["date"])
    rows: list[dict[str, Any]] = []
    start_pos = max(int(cfg.window_size), 1)
    end_pos = len(market) - int(cfg.entry_delay_bars) - int(cfg.hold_bars) - 1
    for pos in range(start_pos, max(start_pos, end_pos), max(1, int(cfg.stride_bars))):
        split = _split(pd.Timestamp(dates.iloc[pos]), cfg)
        if split is None:
            continue
        long = _path(market, pos, "LONG", cfg)
        short = _path(market, pos, "SHORT", cfg)
        if long is None or short is None:
            continue
        label = _choose(long, short, cfg)
        tokens = _state_tokens(features, pos)
        snap = _feature_snapshot(features, pos)
        prompt = _rich_prompt(str(dates.iloc[pos]), tokens, snap, cfg) if str(cfg.prompt_style).lower() == "rich" else _prompt(str(dates.iloc[pos]), tokens, cfg)
        rows.append({
            "task": "text_state_portfolio_decision",
            "split": split,
            "date": str(dates.iloc[pos]),
            "signal_pos": int(pos),
            "prompt": prompt,
            "target": label,
            "state_tokens": tokens,
            "feature_snapshot": snap if str(cfg.prompt_style).lower() == "rich" else {},
            "reward_audit": {"LONG": long, "SHORT": short, "chosen": label},
            "candidate": {"hold_bars": int(cfg.hold_bars)},
            "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_for_training_only": True, "features_signal_time_or_prior": True},
        })
        if int(cfg.max_rows) > 0 and len(rows) >= int(cfg.max_rows):
            break
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]], cfg: PortfolioDecisionCfg) -> dict[str, Any]:
    def one(xs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(r["target"]) for r in xs)
        utils = np.asarray([float(r["reward_audit"][str(r["target"] if r["target"] != "NO_TRADE" else "LONG")]["utility"]) if r["target"] != "NO_TRADE" else max(float(r["reward_audit"]["LONG"]["utility"]), float(r["reward_audit"]["SHORT"]["utility"])) for r in xs], dtype=float) if xs else np.asarray([], dtype=float)
        prompt_lens = [len(str(r["prompt"])) for r in xs]
        return {
            "rows": len(xs),
            "period": {"start": xs[0]["date"] if xs else None, "end": xs[-1]["date"] if xs else None},
            "target_counts": dict(sorted(counts.items())),
            "target_rate": {k: v / max(1, len(xs)) for k, v in sorted(counts.items())},
            "chosen_oracle_utility": {"mean": float(np.mean(utils)) if len(utils) else 0.0, "std": float(np.std(utils)) if len(utils) else 0.0},
            "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        }
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "output": cfg.output,
        "sample_output": cfg.sample_output,
        "total_rows": len(rows),
        "train": one(train),
        "eval": one(eval_rows),
        "prompt_contract": f"one timestamp -> one portfolio action; prompt_style={cfg.prompt_style}; future path only in target/reward_audit",
        "leakage_guard": {"features_signal_time_or_prior": True, "external_join_backward_asof": bool(cfg.wave_trading_root), "eval_split_not_used_for_train": True},
    }


def run(cfg: PortfolioDecisionCfg) -> dict[str, Any]:
    rows = build_rows(cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(200, len(rows))])
    summary = _summary(rows, cfg)
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build text-state portfolio decision rows")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--sample-output", default="")
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=PortfolioDecisionCfg.external_tolerance)
    p.add_argument("--start-date", default=PortfolioDecisionCfg.start_date)
    p.add_argument("--end-date", default=PortfolioDecisionCfg.end_date)
    p.add_argument("--train-end", default=PortfolioDecisionCfg.train_end)
    p.add_argument("--eval-start", default=PortfolioDecisionCfg.eval_start)
    p.add_argument("--window-size", type=int, default=PortfolioDecisionCfg.window_size)
    p.add_argument("--stride-bars", type=int, default=PortfolioDecisionCfg.stride_bars)
    p.add_argument("--hold-bars", type=int, default=PortfolioDecisionCfg.hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=PortfolioDecisionCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=PortfolioDecisionCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=PortfolioDecisionCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=PortfolioDecisionCfg.slippage_rate)
    p.add_argument("--min-trade-net-return-pct", type=float, default=PortfolioDecisionCfg.min_trade_net_return_pct)
    p.add_argument("--max-trade-mae-pct", type=float, default=PortfolioDecisionCfg.max_trade_mae_pct)
    p.add_argument("--min-advantage-pct", type=float, default=PortfolioDecisionCfg.min_advantage_pct)
    p.add_argument("--mae-penalty", type=float, default=PortfolioDecisionCfg.mae_penalty)
    p.add_argument("--no-trade-buffer-pct", type=float, default=PortfolioDecisionCfg.no_trade_buffer_pct)
    p.add_argument("--max-rows", type=int, default=PortfolioDecisionCfg.max_rows)
    p.add_argument("--prompt-style", choices=["categorical", "rich"], default=PortfolioDecisionCfg.prompt_style)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(PortfolioDecisionCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
