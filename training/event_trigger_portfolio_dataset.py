"""Build portfolio decision rows only at no-leak market event triggers.

This replaces fixed stride sampling with event-triggered sampling from features
known at decision time: volatility/volume/taker spikes, trend shocks, range
extremes, and oscillator extremes.  Targets remain future-path labels for
training/evaluation audits only.
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

from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from training.text_state_portfolio_decision_dataset import (
    PortfolioDecisionCfg,
    _choose,
    _feature_snapshot,
    _load_market_with_external,
    _path,
    _rich_prompt,
    _split,
    _state_tokens,
)


@dataclass(frozen=True)
class EventTriggerCfg:
    market_csv: str
    output: str
    summary_output: str
    sample_output: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-01"
    train_end: str = "2024-12-31 23:59:59"
    eval_start: str = "2025-01-01"
    window_size: int = 144
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
    cooldown_bars: int = 72
    min_trigger_score: int = 2
    min_abs_trend_24: float = 0.012
    min_abs_trend_96: float = 0.025
    min_range_vol: float = 0.025
    min_volume_z: float = 1.5
    min_abs_taker_imbalance: float = 0.12
    min_abs_range_pos: float = 0.75
    min_abs_oscillator: float = 0.55
    max_rows: int = 0


def _portfolio_cfg(cfg: EventTriggerCfg) -> PortfolioDecisionCfg:
    return PortfolioDecisionCfg(
        market_csv=cfg.market_csv,
        output=cfg.output,
        summary_output=cfg.summary_output,
        sample_output=cfg.sample_output,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        train_end=cfg.train_end,
        eval_start=cfg.eval_start,
        window_size=cfg.window_size,
        stride_bars=cfg.cooldown_bars,
        hold_bars=cfg.hold_bars,
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        min_trade_net_return_pct=cfg.min_trade_net_return_pct,
        max_trade_mae_pct=cfg.max_trade_mae_pct,
        min_advantage_pct=cfg.min_advantage_pct,
        mae_penalty=cfg.mae_penalty,
        no_trade_buffer_pct=cfg.no_trade_buffer_pct,
        prompt_style="rich",
    )


def _trigger_tags(snap: dict[str, float], cfg: EventTriggerCfg) -> list[str]:
    tags: list[str] = []
    if abs(float(snap["trend_24"])) >= cfg.min_abs_trend_24:
        tags.append("trend_2h_shock_up" if snap["trend_24"] > 0 else "trend_2h_shock_down")
    if abs(float(snap["trend_96"])) >= cfg.min_abs_trend_96:
        tags.append("trend_8h_impulse_up" if snap["trend_96"] > 0 else "trend_8h_impulse_down")
    if float(snap["range_vol"]) >= cfg.min_range_vol:
        tags.append("volatility_expansion")
    if float(snap["volume_zscore"]) >= cfg.min_volume_z:
        tags.append("volume_spike")
    if abs(float(snap["taker_imbalance"])) >= cfg.min_abs_taker_imbalance:
        tags.append("taker_buy_pressure" if snap["taker_imbalance"] > 0 else "taker_sell_pressure")
    if abs(float(snap["range_pos"])) >= cfg.min_abs_range_pos:
        tags.append("range_high_extreme" if snap["range_pos"] > 0 else "range_low_extreme")
    osc = max(abs(float(snap["rsi_norm"])), abs(float(snap["mfi_norm"])))
    if osc >= cfg.min_abs_oscillator:
        tags.append("oscillator_high_extreme" if max(float(snap["rsi_norm"]), float(snap["mfi_norm"])) > 0 else "oscillator_low_extreme")
    return tags


FEATURE_SNAPSHOT_COLS = [
    "trend_12", "trend_24", "trend_96",
    "htf_4h_return_4", "htf_1d_return_4", "htf_3d_return_4", "htf_1w_return_4",
    "range_pos", "htf_4h_range_pos", "htf_1d_range_pos",
    "rsi_norm", "mfi_norm", "bb_z", "return_zscore_48", "close_zscore_48",
    "range_vol", "window_drawdown", "volume_zscore", "trades_ratio",
    "taker_buy_ratio", "taker_imbalance",
    "dxy_momentum", "dxy_zscore", "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_momentum", "usdkrw_zscore",
    "external_any_available",
]


def _snapshot_arrays(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: features[c].to_numpy(dtype=float) if c in features.columns else np.zeros(len(features), dtype=float) for c in FEATURE_SNAPSHOT_COLS}


def _feature_snapshot_fast(arrays: dict[str, np.ndarray], pos: int) -> dict[str, float]:
    return {c: float(a[pos]) if pos < len(a) and np.isfinite(a[pos]) else 0.0 for c, a in arrays.items()}


def build_rows(cfg: EventTriggerCfg) -> list[dict[str, Any]]:
    pcfg = _portfolio_cfg(cfg)
    market = _load_market_with_external(pcfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    for col in EXTENDED_MARKET_FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0
    arrays = _snapshot_arrays(features)
    dates = pd.to_datetime(market["date"])
    rows: list[dict[str, Any]] = []
    last_pos = -10**9
    start_pos = max(int(cfg.window_size), 1)
    end_pos = len(market) - int(cfg.entry_delay_bars) - int(cfg.hold_bars) - 1
    for pos in range(start_pos, max(start_pos, end_pos)):
        split = _split(pd.Timestamp(dates.iloc[pos]), pcfg)
        if split is None:
            continue
        snap = _feature_snapshot_fast(arrays, pos)
        tags = _trigger_tags(snap, cfg)
        if len(tags) < int(cfg.min_trigger_score):
            continue
        if pos - last_pos < int(cfg.cooldown_bars):
            continue
        long = _path(market, pos, "LONG", pcfg)
        short = _path(market, pos, "SHORT", pcfg)
        if long is None or short is None:
            continue
        label = _choose(long, short, pcfg)
        tokens = _state_tokens(features, pos)
        prompt = _rich_prompt(str(dates.iloc[pos]), tokens, snap, pcfg) + "\n\nEvent trigger tags: " + ", ".join(tags)
        rows.append({
            "task": "event_trigger_portfolio_decision",
            "split": split,
            "date": str(dates.iloc[pos]),
            "signal_pos": int(pos),
            "prompt": prompt,
            "target": label,
            "state_tokens": {**tokens, "event_trigger_family": "+".join(sorted(t.split("_")[0] for t in tags))},
            "feature_snapshot": snap,
            "event_triggers": tags,
            "reward_audit": {"LONG": long, "SHORT": short, "chosen": label},
            "candidate": {"hold_bars": int(cfg.hold_bars)},
            "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_path_for_training_only": True, "features_signal_time_or_prior": True, "event_trigger_uses_future_path": False},
        })
        last_pos = pos
        if int(cfg.max_rows) > 0 and len(rows) >= int(cfg.max_rows):
            break
    return rows


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]], cfg: EventTriggerCfg) -> dict[str, Any]:
    def one(xs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(r["target"]) for r in xs)
        trig = Counter(t for r in xs for t in r.get("event_triggers", []))
        lens = [len(str(r["prompt"])) for r in xs]
        return {"rows": len(xs), "period": {"start": xs[0]["date"] if xs else None, "end": xs[-1]["date"] if xs else None}, "target_counts": dict(sorted(counts.items())), "trigger_counts": dict(trig.most_common(20)), "prompt_chars": {"min": min(lens) if lens else 0, "max": max(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens))}}
    train = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    return {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "total_rows": len(rows), "train": one(train), "eval": one(eval_rows), "contract": "event trigger uses only signal-time features; target/reward_audit may use future path for offline labels only"}


def run(cfg: EventTriggerCfg) -> dict[str, Any]:
    rows = build_rows(cfg)
    _write_jsonl(cfg.output, rows)
    if cfg.sample_output:
        _write_jsonl(cfg.sample_output, rows[: min(200, len(rows))])
    report = _summary(rows, cfg)
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event-triggered portfolio decision rows")
    for f in EventTriggerCfg.__dataclass_fields__.values():
        name = "--" + f.name.replace("_", "-")
        default = f.default
        if f.type is int:
            p.add_argument(name, type=int, default=default)
        elif f.type is float:
            p.add_argument(name, type=float, default=default)
        else:
            required = f.name in {"market_csv", "output", "summary_output"}
            p.add_argument(name, default=None if required else default, required=required)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventTriggerCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
