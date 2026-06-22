"""Augment event candidate rows with causal micro-path OHLCV features.

The augmentation joins each signal by market date and computes features using only
bars up to and including the signal bar. This is intended to test whether the
weak event signal is caused by over-compressed inputs rather than label format.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    market_csv: str
    train_output: str
    eval_output: str
    summary_output: str
    windows: str = "3,6,12,24,48,96,288"


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)+("\n" if rows else ""))


def safe_z(x: pd.Series, w: int) -> pd.Series:
    mu = x.rolling(w, min_periods=max(3, w//4)).mean()
    sd = x.rolling(w, min_periods=max(3, w//4)).std().replace(0, np.nan)
    return ((x-mu)/sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_market_features(market_csv: str, windows: list[int]) -> pd.DataFrame:
    df = pd.read_csv(market_csv)
    df["date"] = df["date"].astype(str)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    trades = df["number_of_trades"].astype(float)
    taker_buy = df["taker_buy_base"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    out = pd.DataFrame({"date": df["date"]})
    out["mp_bar_ret"] = (close/open_ - 1.0).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["mp_bar_range"] = ((high-low)/close).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["mp_close_loc_bar"] = ((close-low)/(high-low).replace(0,np.nan)).fillna(0.5) - 0.5
    out["mp_upper_wick"] = ((high-np.maximum(open_, close))/close).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["mp_lower_wick"] = ((np.minimum(open_, close)-low)/close).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["mp_taker_buy_ratio_bar"] = (taker_buy/volume.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.5) - 0.5
    for w in windows:
        roll_ret = close/close.shift(w) - 1.0
        out[f"mp_ret_{w}"] = roll_ret.replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out[f"mp_absret_sum_{w}"] = ret1.abs().rolling(w, min_periods=max(3, w//4)).sum().fillna(0.0)
        out[f"mp_realized_vol_{w}"] = ret1.rolling(w, min_periods=max(3, w//4)).std().fillna(0.0) * np.sqrt(w)
        hh = high.rolling(w, min_periods=max(3, w//4)).max()
        ll = low.rolling(w, min_periods=max(3, w//4)).min()
        rng = (hh-ll).replace(0, np.nan)
        out[f"mp_range_{w}"] = (rng/close).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        out[f"mp_range_pos_{w}"] = ((close-ll)/rng).replace([np.inf,-np.inf], np.nan).fillna(0.5) - 0.5
        out[f"mp_volume_z_{w}"] = safe_z(volume, w)
        out[f"mp_trades_z_{w}"] = safe_z(trades, w)
        imb = (taker_buy/volume.replace(0,np.nan)).fillna(0.5) - 0.5
        out[f"mp_taker_imbalance_mean_{w}"] = imb.rolling(w, min_periods=max(3, w//4)).mean().fillna(0.0)
    # Acceleration / compression terms from already-causal windows.
    if 12 in windows and 48 in windows:
        out["mp_ret_accel_12_48"] = out["mp_ret_12"] - out["mp_ret_48"]
        out["mp_vol_compress_12_48"] = out["mp_realized_vol_12"] / out["mp_realized_vol_48"].replace(0, np.nan)
    if 24 in windows and 96 in windows:
        out["mp_ret_accel_24_96"] = out["mp_ret_24"] - out["mp_ret_96"]
        out["mp_vol_compress_24_96"] = out["mp_realized_vol_24"] / out["mp_realized_vol_96"].replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def bucket(v: float, cuts: tuple[float, float, float, float]) -> str:
    a,b,c,d = cuts
    if v <= a: return "very_low"
    if v <= b: return "low"
    if v < c: return "mid"
    if v < d: return "high"
    return "very_high"


def add_tokens(row: dict[str, Any], feats: dict[str, float]) -> dict[str, str]:
    tokens = dict(row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {})
    tokens["mp_short_momentum"] = bucket(float(feats.get("mp_ret_12", 0.0)), (-0.01, -0.002, 0.002, 0.01))
    tokens["mp_session_momentum"] = bucket(float(feats.get("mp_ret_96", 0.0)), (-0.03, -0.008, 0.008, 0.03))
    tokens["mp_volatility"] = bucket(float(feats.get("mp_realized_vol_96", 0.0)), (0.01, 0.02, 0.04, 0.08))
    tokens["mp_location_24"] = bucket(float(feats.get("mp_range_pos_24", 0.0)), (-0.35, -0.15, 0.15, 0.35))
    tokens["mp_flow_24"] = bucket(float(feats.get("mp_taker_imbalance_mean_24", 0.0)), (-0.12, -0.03, 0.03, 0.12))
    return tokens


def augment(rows: list[dict[str, Any]], feature_by_date: dict[str, dict[str, float]]) -> tuple[list[dict[str, Any]], int]:
    out=[]; matched=0
    for row in rows:
        feats = feature_by_date.get(str(row.get("date")))
        merged = dict(row)
        snap = dict(row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {})
        if feats is not None:
            snap.update(feats)
            merged["state_tokens"] = add_tokens(row, feats)
            matched += 1
        merged["feature_snapshot"] = snap
        lg = dict(row.get("leakage_guard", {}) if isinstance(row.get("leakage_guard"), dict) else {})
        lg["micro_path_features_signal_time_or_prior"] = feats is not None
        merged["leakage_guard"] = lg
        out.append(merged)
    return out, matched


def run(cfg: Cfg) -> dict[str, Any]:
    windows = [int(x) for x in cfg.windows.split(",") if x.strip()]
    mf = compute_market_features(cfg.market_csv, windows)
    feature_cols = [c for c in mf.columns if c != "date"]
    feature_by_date = {str(rec["date"]): {c: float(rec[c]) for c in feature_cols} for rec in mf.to_dict("records")}
    train = load_jsonl(cfg.train_candidates)
    eval_rows = load_jsonl(cfg.eval_candidates)
    train_aug, train_matched = augment(train, feature_by_date)
    eval_aug, eval_matched = augment(eval_rows, feature_by_date)
    write_jsonl(cfg.train_output, train_aug)
    write_jsonl(cfg.eval_output, eval_aug)
    report = {
        "config": cfg.__dict__,
        "feature_count_added": len(feature_cols),
        "feature_examples": feature_cols[:20],
        "train": {"rows": len(train_aug), "matched_rows": train_matched, "output": cfg.train_output},
        "eval": {"rows": len(eval_aug), "matched_rows": eval_matched, "output": cfg.eval_output},
        "leakage_guard": "All micro-path features use rolling/shifted OHLCV values at or before the signal bar date.",
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument("--train-candidates", required=True)
    p.add_argument("--eval-candidates", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--windows", default="3,6,12,24,48,96,288")
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
