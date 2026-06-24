"""Audit fold-level regime signatures for walk-forward failures.

Diagnostic only: test performance labels are used after the fact to identify
candidate pre-test regime signatures. Candidate gates must later be selected
without peeking at the target test fold.
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


@dataclass(frozen=True)
class FoldRegimeAuditCfg:
    market_csv: str
    walkforward_report: str
    output: str
    pretest_days: int = 14
    val_tail_days: int = 14


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def _window(market: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = market["date"]
    return market[(d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))].copy()


def _tail_window(market: pd.DataFrame, end: str, days: int) -> pd.DataFrame:
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=int(days))
    d = market["date"]
    return market[(d >= start_ts) & (d < end_ts)].copy()


def _regime_metrics(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    if len(df) < 10:
        return {f"{prefix}_rows": float(len(df))}
    open_ = df["open"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy() if "close" in df.columns else open_
    ret = np.diff(np.log(np.maximum(open_, 1e-12)))
    total_ret = open_[-1] / max(open_[0], 1e-12) - 1.0
    vol = float(np.std(ret) * math.sqrt(288.0)) if ret.size else 0.0
    realized_range = np.divide(high - low, np.maximum(open_, 1e-12))
    peak = np.maximum.accumulate(open_)
    dd = np.max(1.0 - open_ / np.maximum(peak, 1e-12))
    trough = np.minimum.accumulate(open_)
    runup = np.max(open_ / np.maximum(trough, 1e-12) - 1.0)
    last = open_[-1]
    range_pos = (last - np.min(low)) / max(np.max(high) - np.min(low), 1e-12)
    up_frac = float(np.mean(ret > 0.0)) if ret.size else 0.0
    return {
        f"{prefix}_rows": float(len(df)),
        f"{prefix}_ret_pct": _safe(total_ret * 100.0),
        f"{prefix}_ann_5m_vol_proxy": _safe(vol),
        f"{prefix}_mean_bar_range_pct": _safe(float(np.mean(realized_range)) * 100.0),
        f"{prefix}_max_drawdown_pct": _safe(float(dd) * 100.0),
        f"{prefix}_max_runup_pct": _safe(float(runup) * 100.0),
        f"{prefix}_range_pos": _safe(float(range_pos)),
        f"{prefix}_up_frac": _safe(up_frac),
    }


def _score_separation(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    good = np.asarray([float(r[feature]) for r in rows if not r["is_bad"] and feature in r], dtype=float)
    bad = np.asarray([float(r[feature]) for r in rows if r["is_bad"] and feature in r], dtype=float)
    if good.size < 2 or bad.size < 1:
        return {"feature": feature, "good_n": int(good.size), "bad_n": int(bad.size)}
    pooled = float(np.std(np.concatenate([good, bad]), ddof=1))
    effect = (float(np.mean(bad)) - float(np.mean(good))) / pooled if pooled > 0 else 0.0
    return {
        "feature": feature,
        "good_n": int(good.size),
        "bad_n": int(bad.size),
        "good_mean": float(np.mean(good)),
        "bad_mean": float(np.mean(bad)),
        "effect_bad_minus_good": effect,
        "good_min": float(np.min(good)),
        "good_max": float(np.max(good)),
        "bad_min": float(np.min(bad)),
        "bad_max": float(np.max(bad)),
    }


def run(cfg: FoldRegimeAuditCfg) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    wf = json.loads(Path(cfg.walkforward_report).read_text())
    fold_rows: list[dict[str, Any]] = []
    for fr in wf.get("folds", []):
        fold = fr["fold"]
        test_sim = fr.get("test", {}).get("test_backtest", {}).get("sim")
        if not test_sim:
            continue
        row: dict[str, Any] = {
            "fold_id": int(fold["fold_id"]),
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "status": fr.get("status"),
            "test_cagr_pct": float(test_sim.get("cagr_pct", 0.0) or 0.0),
            "test_strict_mdd_pct": float(test_sim.get("strict_mdd_pct", 0.0) or 0.0),
            "test_ratio": float(test_sim.get("cagr_to_strict_mdd", 0.0) or 0.0),
            "test_trades": int(test_sim.get("trade_entries", 0) or 0),
        }
        row["is_bad"] = bool(row["test_ratio"] < 0.0)
        row.update(_regime_metrics(_tail_window(market, fold["test_start"], cfg.pretest_days), "pretest"))
        row.update(_regime_metrics(_tail_window(market, fold["val_end"], cfg.val_tail_days), "val_tail"))
        row.update(_regime_metrics(_window(market, fold["val_start"], fold["val_end"]), "val_full"))
        fold_rows.append(row)
    numeric = sorted({k for r in fold_rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in {"fold_id", "is_bad"}})
    separations = [_score_separation(fold_rows, f) for f in numeric]
    separations.sort(key=lambda x: abs(float(x.get("effect_bad_minus_good", 0.0))), reverse=True)
    out = {
        "config": asdict(cfg),
        "folds": fold_rows,
        "separation_top": separations[:30],
        "leakage_note": "diagnostic uses test outcomes to label bad folds; future gates need nested validation",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit fold-level regime signatures")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--walkforward-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pretest-days", type=int, default=FoldRegimeAuditCfg.pretest_days)
    p.add_argument("--val-tail-days", type=int, default=FoldRegimeAuditCfg.val_tail_days)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FoldRegimeAuditCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
