"""Walk-forward strict backtest for a fixed event family hypothesis.

This is for validating a chosen alpha surface over longer history without using
future data for thresholds.  The family/quantile/hold are fixed before the run;
for each fold, the signal threshold is fit only on rows before that fold, then
fold events are stitched and strict-bar backtested over the full out-of-sample
period.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import EventPoolConfig, _candidate_rows_for_family, _feature_candidates, _load_market, _simulate_rows, _split_mask


@dataclass(frozen=True)
class FixedFamilyWalkforwardConfig:
    input_csv: str
    output: str
    family: str = "rex_htf_pullback_reclaim"
    quantile: float = 0.75
    train_start: str = "2020-01-01"
    eval_start: str = "2021-01-01"
    eval_end: str = "2026-06-01"
    fold_months: int = 1
    hold_bars: int = 144
    entry_delay_bars: int = 1
    window_size: int = 144
    stride_bars: int = 24
    leverage_grid: str = "0.5,1.0,1.5,2.0"
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001


def _folds(start: str, end: str, months: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    cur = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    while cur < stop:
        nxt = min(cur + pd.DateOffset(months=int(months)), stop)
        out.append({"name": f"{cur:%Y%m}_{nxt:%Y%m}", "start": str(cur.date()), "end": str(nxt.date())})
        cur = nxt
    return out


def _leverage_values(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _split_name(start: str) -> str:
    if start < "2025-01-01":
        return "history_oos"
    if start < "2026-01-01":
        return "test_2025"
    return "eval_2026h1"


def _period_rows(events: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    for row in events:
        dt = pd.Timestamp(str(row.get("signal_date")))
        if s <= dt < e:
            out.append(row)
    return out


def run(cfg: FixedFamilyWalkforwardConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    if cfg.family not in families:
        raise ValueError(f"family not found: {cfg.family}")
    strength, direction = families[cfg.family]
    dates = pd.to_datetime(market["date"])
    folds = _folds(cfg.eval_start, cfg.eval_end, cfg.fold_months)
    base_cfg = EventPoolConfig(
        input_csv=cfg.input_csv,
        output=cfg.output,
        train_start=cfg.train_start,
        train_end=cfg.eval_start,
        val_start=cfg.eval_start,
        val_end=cfg.eval_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
        hold_bars=cfg.hold_bars,
        entry_delay_bars=cfg.entry_delay_bars,
        window_size=cfg.window_size,
        stride_bars=cfg.stride_bars,
        quantile=cfg.quantile,
        leverage=1.0,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )

    stitched: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    q = float(np.clip(cfg.quantile, 0.5, 0.99))
    for fold in folds:
        train_mask = _split_mask(dates, cfg.train_start, fold["start"])
        fold_mask = _split_mask(dates, fold["start"], fold["end"])
        x = strength[train_mask & np.isfinite(strength) & (strength > 0.0)]
        if x.size < 100:
            fold_reports.append({"fold": fold, "skip": "too_few_positive_train_strengths", "positive_train_strengths": int(x.size)})
            continue
        threshold = float(np.quantile(x, q))
        rows = _candidate_rows_for_family(market, strength, direction, family=cfg.family, threshold=threshold, mask=fold_mask, cfg=base_cfg)
        stitched.extend(rows)
        fold_reports.append({"fold": fold, "split": _split_name(fold["start"]), "threshold": threshold, "candidate_rows": len(rows), "positive_train_strengths": int(x.size)})

    stitched.sort(key=lambda r: (str(r.get("entry_date")), str(r.get("family")), str(r.get("side"))))
    leverages = []
    for lev in _leverage_values(cfg.leverage_grid):
        sim_cfg = replace(base_cfg, leverage=float(lev))
        full = _simulate_rows(stitched, market, sim_cfg)
        periods = {
            "history_oos_2021_2024": _simulate_rows(_period_rows(stitched, "2021-01-01", "2025-01-01"), market, sim_cfg),
            "test_2025": _simulate_rows(_period_rows(stitched, "2025-01-01", "2026-01-01"), market, sim_cfg),
            "eval_2026h1": _simulate_rows(_period_rows(stitched, "2026-01-01", cfg.eval_end), market, sim_cfg),
        }
        leverages.append({"leverage": float(lev), "full": {"sim": full["sim"], "trade_stats": full["trade_stats"]}, "periods": {k: {"sim": v["sim"], "trade_stats": v["trade_stats"]} for k, v in periods.items()}})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "folds": fold_reports,
        "stitched_candidate_rows": len(stitched),
        "leverage_results": leverages,
        "leakage_guard": {
            "family_quantile_hold_fixed_before_run": True,
            "each_fold_threshold_fit_uses_only_rows_before_fold_start": True,
            "events_stitched_after_fold_local_thresholding": True,
            "no_target_fold_outcomes_used_for_thresholds": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward strict backtest for a fixed event family")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", default=FixedFamilyWalkforwardConfig.family)
    p.add_argument("--quantile", type=float, default=FixedFamilyWalkforwardConfig.quantile)
    p.add_argument("--train-start", default=FixedFamilyWalkforwardConfig.train_start)
    p.add_argument("--eval-start", default=FixedFamilyWalkforwardConfig.eval_start)
    p.add_argument("--eval-end", default=FixedFamilyWalkforwardConfig.eval_end)
    p.add_argument("--fold-months", type=int, default=FixedFamilyWalkforwardConfig.fold_months)
    p.add_argument("--hold-bars", type=int, default=FixedFamilyWalkforwardConfig.hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=FixedFamilyWalkforwardConfig.entry_delay_bars)
    p.add_argument("--window-size", type=int, default=FixedFamilyWalkforwardConfig.window_size)
    p.add_argument("--stride-bars", type=int, default=FixedFamilyWalkforwardConfig.stride_bars)
    p.add_argument("--leverage-grid", default=FixedFamilyWalkforwardConfig.leverage_grid)
    p.add_argument("--fee-rate", type=float, default=FixedFamilyWalkforwardConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=FixedFamilyWalkforwardConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    rep = run(FixedFamilyWalkforwardConfig(**vars(parse_args())))
    print(json.dumps({"stitched_candidate_rows": rep["stitched_candidate_rows"], "leverage_results": rep["leverage_results"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
