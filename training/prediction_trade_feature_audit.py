"""Audit past-only market features on executed prediction trades.

Given a prediction stream and its strict overlay backtest, attach feature values at
signal time and report which features separate winning vs losing executed trades.
This is diagnostic only: realized trade returns are labels, not deployment inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame


@dataclass(frozen=True)
class TradeFeatureAuditCfg:
    market_csv: str
    backtest_json: str
    output: str
    window_size: int = 144
    min_trades: int = 20
    top_k: int = 80


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / (sd / math.sqrt(len(x)))) if sd > 1e-12 else 0.0


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _feature_row(feature: str, x: np.ndarray, y: np.ndarray, side_sign: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]; ss = side_sign[ok]
    if len(x) < 10:
        return {"feature": feature, "n": int(len(x)), "score": -1e9}
    qlo, qhi = np.quantile(x, [0.25, 0.75])
    low = y[x <= qlo]
    high = y[x >= qhi]
    spread = float(np.mean(high) - np.mean(low)) if len(low) and len(high) else 0.0
    signed_x = x * ss
    signed_corr = _corr(signed_x, y)
    raw_corr = _corr(x, y)
    win = y > 0.0
    win_mean = float(np.mean(x[win])) if np.any(win) else 0.0
    loss_mean = float(np.mean(x[~win])) if np.any(~win) else 0.0
    sep = win_mean - loss_mean
    score = abs(spread) + 5.0 * abs(raw_corr) + 5.0 * abs(signed_corr) + abs(sep)
    return {
        "feature": feature,
        "n": int(len(x)),
        "raw_corr": raw_corr,
        "signed_corr": signed_corr,
        "q25": float(qlo),
        "q75": float(qhi),
        "low_q_mean_trade_ret_pct": float(np.mean(low)) if len(low) else 0.0,
        "high_q_mean_trade_ret_pct": float(np.mean(high)) if len(high) else 0.0,
        "high_minus_low_mean_trade_ret_pct": spread,
        "win_mean": win_mean,
        "loss_mean": loss_mean,
        "win_minus_loss_mean": sep,
        "score": score,
    }


def run(cfg: TradeFeatureAuditCfg) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bt = json.loads(Path(cfg.backtest_json).read_text())
    trades = bt.get("executed", [])
    rows = []
    for e in trades:
        pos = int(e.get("signal_pos", -1) or -1)
        if 0 <= pos < len(market):
            rows.append(e)
    if len(rows) < int(cfg.min_trades):
        raise ValueError(f"not enough executed trades: {len(rows)}")
    idx = np.asarray([int(e["signal_pos"]) for e in rows], dtype=int)
    y = np.asarray([float(e.get("trade_ret_pct", 0.0) or 0.0) for e in rows], dtype=float)
    side_sign = np.asarray([1.0 if str(e.get("side")) == "LONG" else -1.0 for e in rows], dtype=float)
    feature_rows = []
    for col in features.columns:
        vals = features[col].to_numpy(dtype=float)[idx]
        if float(np.nanstd(vals)) <= 1e-12:
            continue
        feature_rows.append(_feature_row(str(col), vals, y, side_sign))
    feature_rows.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
    by_month: dict[str, dict[str, Any]] = {}
    for e in rows:
        m = str(e.get("date", ""))[:7]
        slot = by_month.setdefault(m, {"trades": 0, "sum_trade_ret_pct": 0.0, "long": 0, "short": 0})
        slot["trades"] += 1
        slot["sum_trade_ret_pct"] += float(e.get("trade_ret_pct", 0.0) or 0.0)
        slot["long" if str(e.get("side")) == "LONG" else "short"] += 1
    report = {
        "config": asdict(cfg),
        "trades": {"n": len(rows), "mean_trade_ret_pct": float(np.mean(y)), "t_stat": _t_stat(y), "months": by_month},
        "top_features": feature_rows[: int(cfg.top_k)],
        "leakage_guard": {"features_at_signal_time_only": True, "trade_returns_used_for_diagnostic_labels_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit signal-time features against executed trade returns")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--backtest-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--top-k", type=int, default=80)
    return p.parse_args()


def main() -> None:
    r = run(TradeFeatureAuditCfg(**vars(parse_args())))
    print(json.dumps({"trades": r["trades"], "top10": r["top_features"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
