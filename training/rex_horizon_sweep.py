"""Focused no-leak horizon/stride sweep for REX pullback candidate families."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import (
    EventPoolConfig,
    _feature_candidates,
    _load_market,
    _simulate_rows,
    _split_mask,
)


@dataclass(frozen=True)
class RexHorizonSweepConfig:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2025-01-01"
    val_start: str = "2025-01-01"
    val_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    family_include: str = "rex_htf_pullback_resume"
    hold_bars_grid: tuple[int, ...] = (72, 144, 216, 288, 432, 576)
    stride_bars_grid: tuple[int, ...] = (12, 24, 36, 72)
    quantile_grid: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90)
    window_size: int = 144
    entry_delay_bars: int = 1
    min_train_trades: int = 80
    min_val_trades: int = 30
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"
    top_k: int = 40


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(raw).split(",") if x.strip())


def _rank(row: dict[str, Any]) -> float:
    tr = row["train"]["sim"]
    va = row["val"]["sim"]
    vt = row["val"].get("trade_stats", {})
    if int(tr.get("trade_entries", 0)) <= 0 or int(va.get("trade_entries", 0)) <= 0:
        return -1e9
    if float(tr.get("cagr_pct", -1e9)) <= 0 or float(va.get("cagr_pct", -1e9)) <= 0:
        return -1e9
    if float(tr.get("strict_mdd_pct", 1e9)) > 45 or float(va.get("strict_mdd_pct", 1e9)) > 18:
        return -1e9
    p = float(vt.get("p_value_mean_ret_approx", 1.0) or 1.0)
    return (
        float(va.get("cagr_to_strict_mdd", 0.0) or 0.0)
        + 0.20 * float(tr.get("cagr_to_strict_mdd", 0.0) or 0.0)
        + min(int(va.get("trade_entries", 0) or 0), 180) / 250.0
        - 0.25 * p
    )


def _fast_candidate_rows(
    dates: list[str],
    strength: np.ndarray,
    direction: np.ndarray,
    *,
    family: str,
    threshold: float,
    mask: np.ndarray,
    hold_bars: int,
    entry_delay_bars: int,
    stride_bars: int,
    window_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_pos = len(dates) - int(entry_delay_bars) - int(hold_bars) - 1
    start = max(0, int(window_size) - 1)
    step = max(1, int(stride_bars))
    for pos in range(start, max(0, last_pos) + 1, step):
        if not mask[pos]:
            continue
        val = float(strength[pos])
        if (not np.isfinite(val)) or val <= max(0.0, float(threshold)):
            continue
        side_dir = float(direction[pos])
        if side_dir == 0.0 or not np.isfinite(side_dir):
            continue
        entry_pos = pos + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        side = "LONG" if side_dir > 0 else "SHORT"
        rows.append({
            "date": dates[pos],
            "signal_date": dates[pos],
            "entry_date": dates[entry_pos],
            "exit_date": dates[exit_pos],
            "side": side,
            "family": family,
            "strength": val,
            "score_mean": 1.0,
        })
    return rows


def _mk_cfg(base: RexHorizonSweepConfig, *, hold: int, stride: int) -> EventPoolConfig:
    return EventPoolConfig(
        input_csv=base.input_csv,
        output=base.output,
        train_start=base.train_start,
        train_end=base.train_end,
        val_start=base.val_start,
        val_end=base.val_end,
        eval_start=base.eval_start,
        eval_end=base.eval_end,
        hold_bars=int(hold),
        entry_delay_bars=int(base.entry_delay_bars),
        window_size=int(base.window_size),
        stride_bars=int(stride),
        quantile=0.8,
        min_train_trades=int(base.min_train_trades),
        min_val_trades=int(base.min_val_trades),
        leverage=float(base.leverage),
        fee_rate=float(base.fee_rate),
        slippage_rate=float(base.slippage_rate),
        wave_trading_root=base.wave_trading_root,
        external_tolerance=base.external_tolerance,
    )


def run(cfg: RexHorizonSweepConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    date_strings = [str(x) for x in market["date"].tolist()]
    masks = {
        "train": _split_mask(dates, cfg.train_start, cfg.train_end),
        "val": _split_mask(dates, cfg.val_start, cfg.val_end),
        "eval": _split_mask(dates, cfg.eval_start, cfg.eval_end),
    }
    needles = [x.strip() for x in str(cfg.family_include).split(",") if x.strip()]
    families = {k: v for k, v in _feature_candidates(features).items() if any(n in k for n in needles)}
    if not families:
        raise ValueError(f"no families match {cfg.family_include!r}")

    rows: list[dict[str, Any]] = []
    for family, (strength, direction) in families.items():
        train_x = strength[masks["train"] & np.isfinite(strength) & (strength > 0.0)]
        if train_x.size < 100:
            continue
        for q in cfg.quantile_grid:
            threshold = float(np.quantile(train_x, float(q)))
            for hold in cfg.hold_bars_grid:
                for stride in cfg.stride_bars_grid:
                    ecfg = _mk_cfg(cfg, hold=int(hold), stride=int(stride))
                    split_results: dict[str, Any] = {}
                    counts: dict[str, int] = {}
                    for split, mask in masks.items():
                        cand_rows = _fast_candidate_rows(date_strings, strength, direction, family=family, threshold=threshold, mask=mask, hold_bars=int(hold), entry_delay_bars=int(cfg.entry_delay_bars), stride_bars=int(stride), window_size=int(cfg.window_size))
                        counts[split] = len(cand_rows)
                        sim = _simulate_rows(cand_rows, market, ecfg)
                        split_results[split] = {"sim": sim["sim"], "trade_stats": sim["trade_stats"], "candidate_count": len(cand_rows)}
                    row = {
                        "family": family,
                        "quantile": float(q),
                        "threshold": threshold,
                        "hold_bars": int(hold),
                        "stride_bars": int(stride),
                        **split_results,
                    }
                    row["score_train_val_only"] = _rank(row)
                    rows.append(row)
    rows.sort(key=lambda r: r["score_train_val_only"], reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"hold_bars_grid": list(cfg.hold_bars_grid), "stride_bars_grid": list(cfg.stride_bars_grid), "quantile_grid": list(cfg.quantile_grid)},
        "inputs": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "score and ordering use train+2025 validation only; 2026 eval is report-only",
        "trial_count": len(rows),
        "top": rows[: int(cfg.top_k)],
        "leakage_guard": {
            "thresholds_fit_on_train_only": True,
            "selection_uses_train_and_val_only": True,
            "eval_not_used_for_selection": True,
            "features_use_rows_at_or_before_signal": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default=RexHorizonSweepConfig.train_start)
    p.add_argument("--train-end", default=RexHorizonSweepConfig.train_end)
    p.add_argument("--val-start", default=RexHorizonSweepConfig.val_start)
    p.add_argument("--val-end", default=RexHorizonSweepConfig.val_end)
    p.add_argument("--eval-start", default=RexHorizonSweepConfig.eval_start)
    p.add_argument("--eval-end", default=RexHorizonSweepConfig.eval_end)
    p.add_argument("--family-include", default=RexHorizonSweepConfig.family_include)
    p.add_argument("--hold-bars-grid", default=",".join(map(str, RexHorizonSweepConfig.hold_bars_grid)))
    p.add_argument("--stride-bars-grid", default=",".join(map(str, RexHorizonSweepConfig.stride_bars_grid)))
    p.add_argument("--quantile-grid", default=",".join(map(str, RexHorizonSweepConfig.quantile_grid)))
    p.add_argument("--window-size", type=int, default=RexHorizonSweepConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=RexHorizonSweepConfig.entry_delay_bars)
    p.add_argument("--min-train-trades", type=int, default=RexHorizonSweepConfig.min_train_trades)
    p.add_argument("--min-val-trades", type=int, default=RexHorizonSweepConfig.min_val_trades)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default=RexHorizonSweepConfig.external_tolerance)
    p.add_argument("--top-k", type=int, default=RexHorizonSweepConfig.top_k)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = RexHorizonSweepConfig(
        input_csv=a.input_csv,
        output=a.output,
        train_start=a.train_start,
        train_end=a.train_end,
        val_start=a.val_start,
        val_end=a.val_end,
        eval_start=a.eval_start,
        eval_end=a.eval_end,
        family_include=a.family_include,
        hold_bars_grid=_parse_ints(a.hold_bars_grid),
        stride_bars_grid=_parse_ints(a.stride_bars_grid),
        quantile_grid=_parse_floats(a.quantile_grid),
        window_size=a.window_size,
        entry_delay_bars=a.entry_delay_bars,
        min_train_trades=a.min_train_trades,
        min_val_trades=a.min_val_trades,
        wave_trading_root=a.wave_trading_root,
        external_tolerance=a.external_tolerance,
        top_k=a.top_k,
    )
    out = run(cfg)
    print(json.dumps({"trial_count": out["trial_count"], "top": out["top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
