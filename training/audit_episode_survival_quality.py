"""Vectorized MAE/MFE survival audit for episode setup-quality features.

This is a diagnostic bridge toward RLLM labels: it measures whether causal
setup-quality buckets predict executable path survival (positive net return,
low MAE, useful MFE/MAE) across chronological train/test/eval splits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.audit_setup_quality_filters import _bucket_masks
from training.price_action_episode_policy import EPISODE_SIDES, add_sequence_context_features, build_episode_event_features


@dataclass(frozen=True)
class SurvivalQualityCfg:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31 23:59:59"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "288,576,2016,4032"
    horizons: str = "72,144,288,432"
    include_sequence_context: bool = False
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 0.5
    min_split_triggers: int = 40
    max_survival_mae_pct: float = 2.0
    min_mfe_to_mae: float = 1.25
    top_k: int = 120


def _mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)), dtype=bool)


def _event_type(event: str) -> str:
    for suffix in sorted(EPISODE_SIDES, key=len, reverse=True):
        if event.endswith("_" + suffix):
            return suffix
    return "unknown"


def _future_path_arrays(market: pd.DataFrame, horizons: list[int], cfg: SurvivalQualityCfg) -> dict[int, dict[str, np.ndarray]]:
    open_ = market["open"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    n = len(market)
    out: dict[int, dict[str, np.ndarray]] = {}
    idx = np.arange(n)
    safe_open = np.where(open_ > 0.0, open_, np.nan)
    cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate)) * float(cfg.leverage)
    for h in horizons:
        entry_idx = idx + int(cfg.entry_delay_bars)
        exit_idx = entry_idx + int(h)
        ok = (entry_idx >= 0) & (exit_idx < n)
        max_high = np.full(n, np.nan, dtype=float)
        min_low = np.full(n, np.nan, dtype=float)
        # Horizon count is small; this vectorized loop is fast enough and avoids
        # rolling alignment mistakes for forward windows.
        mh = np.full(ok.sum(), -np.inf, dtype=float)
        ml = np.full(ok.sum(), np.inf, dtype=float)
        base_entries = entry_idx[ok]
        for off in range(int(h)):
            p = base_entries + off
            mh = np.maximum(mh, high[p])
            ml = np.minimum(ml, low[p])
        max_high[ok] = mh
        min_low[ok] = ml
        entry = np.full(n, np.nan, dtype=float)
        exitp = np.full(n, np.nan, dtype=float)
        entry[ok] = safe_open[entry_idx[ok]]
        exitp[ok] = safe_open[exit_idx[ok]]
        long_gross = exitp / entry - 1.0
        short_gross = entry / exitp - 1.0
        out[int(h)] = {
            "LONG_net": float(cfg.leverage) * long_gross - cost,
            "SHORT_net": float(cfg.leverage) * short_gross - cost,
            "LONG_mae": np.maximum(0.0, (entry - min_low) / entry),
            "SHORT_mae": np.maximum(0.0, (max_high - entry) / entry),
            "LONG_mfe": np.maximum(0.0, (max_high - entry) / entry),
            "SHORT_mfe": np.maximum(0.0, (entry - min_low) / entry),
        }
    return out


def _quality_arrays(market: pd.DataFrame, cfg: SurvivalQualityCfg) -> dict[str, np.ndarray]:
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    open_ = market["open"].to_numpy(dtype=float)
    close = market["close"].to_numpy(dtype=float)
    entry_idx = np.arange(len(market)) + int(cfg.entry_delay_bars)
    entry = np.full(len(market), np.nan, dtype=float)
    ok = entry_idx < len(market)
    entry[ok] = open_[entry_idx[ok]]
    rng = np.maximum(1e-12, high - low)
    close_pos = (close - low) / rng
    body_frac = np.abs(close - open_) / rng
    upper_wick_frac = (high - np.maximum(open_, close)) / rng
    lower_wick_frac = (np.minimum(open_, close) - low) / rng
    range_bps = rng / np.maximum(1e-12, close) * 10_000.0
    return {
        "range_bps": range_bps,
        "body_frac": body_frac,
        "LONG_risk_bps": np.maximum(0.0, (entry - low) / entry * 10_000.0),
        "SHORT_risk_bps": np.maximum(0.0, (high - entry) / entry * 10_000.0),
        "LONG_favorable_wick_frac": lower_wick_frac,
        "SHORT_favorable_wick_frac": upper_wick_frac,
        "LONG_close_quality": close_pos,
        "SHORT_close_quality": 1.0 - close_pos,
    }


def _stats(net: np.ndarray, mae: np.ndarray, mfe: np.ndarray, mask: np.ndarray, cfg: SurvivalQualityCfg) -> dict[str, Any]:
    valid = mask & np.isfinite(net) & np.isfinite(mae) & np.isfinite(mfe)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mean_net_pct": 0.0, "mean_mae_pct": 0.0, "win_rate": 0.0, "survival_rate": 0.0, "mean_utility_pct": 0.0, "mfe_to_mae": 0.0}
    nr = net[valid]
    mr = mae[valid]
    fr = mfe[valid]
    mfe_to_mae = fr / np.maximum(mr, 1e-9)
    survival = (nr > 0.0) & (mr * 100.0 <= float(cfg.max_survival_mae_pct)) & (mfe_to_mae >= float(cfg.min_mfe_to_mae))
    util = nr - float(cfg.mae_penalty) * mr
    return {
        "n": n,
        "mean_net_pct": float(np.mean(nr) * 100.0),
        "mean_mae_pct": float(np.mean(mr) * 100.0),
        "win_rate": float(np.mean(nr > 0.0)),
        "survival_rate": float(np.mean(survival)),
        "mean_utility_pct": float(np.mean(util) * 100.0),
        "mfe_to_mae": float(np.mean(mfe_to_mae)),
    }


def _score(st: dict[str, Any]) -> float:
    return float(st["mean_utility_pct"]) + 0.6 * float(st["mean_net_pct"]) - 0.25 * float(st["mean_mae_pct"]) + 5.0 * float(st["survival_rate"]) + min(1.0, float(st["n"]) / 200.0)


def run(cfg: SurvivalQualityCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    horizons = _parse_list(cfg.horizons, int)
    feats = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        feats = add_sequence_context_features(market, feats, windows)
    train_mask = _mask(dates, cfg.train_start, cfg.train_end)
    test_mask = _mask(dates, cfg.test_start, cfg.test_end)
    eval_mask = _mask(dates, cfg.eval_start, cfg.eval_end)
    path = _future_path_arrays(market, horizons, cfg)
    q = _quality_arrays(market, cfg)
    rows = []
    for col in feats.columns:
        et = _event_type(col)
        if et not in EPISODE_SIDES:
            continue
        side, episode = EPISODE_SIDES[et]
        ev = feats[col].to_numpy(dtype=float) > 0.5
        if int((ev & train_mask).sum()) < int(cfg.min_split_triggers):
            continue
        for h in horizons:
            p = path[int(h)]
            net = p[f"{side}_net"]
            mae = p[f"{side}_mae"]
            mfe = p[f"{side}_mfe"]
            for feature, vals in {
                "all": np.zeros(len(market), dtype=float),
                "risk_bps": q[f"{side}_risk_bps"],
                "range_bps": q["range_bps"],
                "body_frac": q["body_frac"],
                "favorable_wick_frac": q[f"{side}_favorable_wick_frac"],
                "close_quality": q[f"{side}_close_quality"],
            }.items():
                bucket_masks = {"all": np.ones(len(market), dtype=bool)} if feature == "all" else _bucket_masks(pd.Series(vals[ev & train_mask]), pd.Series(vals))
                for bucket, bm in bucket_masks.items():
                    rule_mask = ev & np.asarray(bm, dtype=bool)
                    st_train = _stats(net, mae, mfe, rule_mask & train_mask, cfg)
                    if int(st_train["n"]) < int(cfg.min_split_triggers):
                        continue
                    st_test = _stats(net, mae, mfe, rule_mask & test_mask, cfg)
                    st_eval = _stats(net, mae, mfe, rule_mask & eval_mask, cfg)
                    rows.append({
                        "event": col,
                        "event_type": et,
                        "episode": episode,
                        "side": side,
                        "horizon": int(h),
                        "filter": {"feature": feature, "bucket": bucket, "threshold_source": "train_quantiles" if feature != "all" else "none"},
                        "train": st_train,
                        "test": st_test,
                        "eval_diagnostic": st_eval,
                        "train_score": _score(st_train),
                        "test_score": _score(st_test),
                        "robust_train_test": (
                            st_train["n"] >= int(cfg.min_split_triggers)
                            and st_test["n"] >= int(cfg.min_split_triggers)
                            and st_train["mean_utility_pct"] > 0
                            and st_test["mean_utility_pct"] > 0
                            and st_train["survival_rate"] >= 0.45
                            and st_test["survival_rate"] >= 0.45
                        ),
                    })
    ranked = sorted(rows, key=lambda r: (bool(r["robust_train_test"]), float(r["test_score"]), float(r["train_score"]), int(r["test"]["n"])), reverse=True)
    robust = [r for r in ranked if r["robust_train_test"]]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "feature_columns": len(feats.columns),
        "candidates": len(rows),
        "robust_train_test_count": len(robust),
        "top": ranked[: int(cfg.top_k)],
        "top_robust_train_test": robust[: int(cfg.top_k)],
        "protocol": "future MAE/MFE are labels only; setup buckets use train-trigger quantiles and causal signal-bar attributes",
        "leakage_guard": {"bucket_thresholds_fit_on_eval": False, "future_path_used_as_input": False, "eval_used_for_selection": False},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(SurvivalQualityCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=SurvivalQualityCfg.windows)
    p.add_argument("--horizons", default=SurvivalQualityCfg.horizons)
    p.add_argument("--include-sequence-context", action="store_true")
    p.add_argument("--entry-delay-bars", type=int, default=SurvivalQualityCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=SurvivalQualityCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=SurvivalQualityCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SurvivalQualityCfg.slippage_rate)
    p.add_argument("--mae-penalty", type=float, default=SurvivalQualityCfg.mae_penalty)
    p.add_argument("--min-split-triggers", type=int, default=SurvivalQualityCfg.min_split_triggers)
    p.add_argument("--max-survival-mae-pct", type=float, default=SurvivalQualityCfg.max_survival_mae_pct)
    p.add_argument("--min-mfe-to-mae", type=float, default=SurvivalQualityCfg.min_mfe_to_mae)
    p.add_argument("--top-k", type=int, default=SurvivalQualityCfg.top_k)
    return p.parse_args()


def main() -> None:
    r = run(SurvivalQualityCfg(**vars(parse_args())))
    print(json.dumps({
        "output": r["config"]["output"],
        "feature_columns": r["feature_columns"],
        "candidates": r["candidates"],
        "robust_train_test_count": r["robust_train_test_count"],
        "top": r["top"][:10],
        "top_robust_train_test": r["top_robust_train_test"][:10],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
