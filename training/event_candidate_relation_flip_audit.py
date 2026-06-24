"""Audit feature/reward relation flips across walk-forward folds.

This is diagnostic only: test rewards are read to explain failures after the fact,
not to select live policies.
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

from training.event_candidate_ridge_ranker import _date, _load


@dataclass(frozen=True)
class RelationFlipAuditCfg:
    input_jsonl: str
    walkforward_report: str
    output: str
    features: str = ""
    top_n_features: int = 40
    min_rows: int = 200
    quantile: float = 0.2


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("rank_utility", reward.get("net_return_pct", 0.0)) or 0.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx = float(np.std(xr))
    sy = float(np.std(yr))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _t_stat(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(values) / (sd / math.sqrt(values.size)))


def _metrics(rows: list[dict[str, Any]], feature: str, *, q: float, min_rows: int) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        if feature not in snap:
            continue
        try:
            x = float(snap.get(feature, 0.0) or 0.0)
            y = _utility(r)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < int(min_rows):
        return {"n": int(x.size)}
    qq = float(np.clip(q, 0.01, 0.49))
    lo = float(np.quantile(x, qq))
    hi = float(np.quantile(x, 1.0 - qq))
    low = y[x <= lo]
    high = y[x >= hi]
    spread = _mean(high) - _mean(low)
    return {
        "n": int(x.size),
        "spearman_ic": _spearman(x, y),
        "q_low": lo,
        "q_high": hi,
        "q_low_mean": _mean(low),
        "q_high_mean": _mean(high),
        "q_high_minus_low": spread,
        "q_spread_t": _t_stat(np.concatenate([high, -low])) if high.size and low.size else 0.0,
    }


def _sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _feature_names(rows: list[dict[str, Any]], cfg: RelationFlipAuditCfg) -> list[str]:
    explicit = [x.strip() for x in cfg.features.split(",") if x.strip()]
    if explicit:
        return explicit
    scores: dict[str, float] = {}
    for r in rows:
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        for k, v in snap.items():
            if not isinstance(v, (int, float)):
                continue
            if str(k).startswith("pa_ext_") or str(k) in {"action_side_sign", "taker_imbalance", "bb_z", "rsi_norm", "range_pos", "htf_1d_return_1", "usdkrw_zscore"}:
                scores[str(k)] = scores.get(str(k), 0.0) + abs(float(v or 0.0))
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: int(cfg.top_n_features)]]


def _in_half_open(row: dict[str, Any], start: str, end: str) -> bool:
    d = _date(row)
    return start <= d < end


def _rows_for(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [r for r in rows if _in_half_open(r, start, end)]


def run(cfg: RelationFlipAuditCfg) -> dict[str, Any]:
    rows = sorted(_load(cfg.input_jsonl), key=lambda r: (_date(r), int(r.get("signal_pos", -1) or -1), str(r.get("side", ""))))
    report = json.loads(Path(cfg.walkforward_report).read_text())
    features = _feature_names(rows, cfg)
    fold_reports: list[dict[str, Any]] = []
    aggregate_flip_counts: dict[str, int] = {"fit_val_ic_flip": 0, "val_test_ic_flip": 0, "fit_val_spread_flip": 0, "val_test_spread_flip": 0}
    for fold_report in report.get("folds", []):
        fold = fold_report["fold"]
        split_rows = {
            "fit": _rows_for(rows, fold["fit_start"], fold["fit_end"]),
            "val": _rows_for(rows, fold["val_start"], fold["val_end"]),
            "test": _rows_for(rows, fold["test_start"], fold["test_end"]),
        }
        feature_rows: list[dict[str, Any]] = []
        for feat in features:
            mets = {name: _metrics(rs, feat, q=cfg.quantile, min_rows=cfg.min_rows) for name, rs in split_rows.items()}
            fit_ic = _sign(float(mets["fit"].get("spearman_ic", 0.0)))
            val_ic = _sign(float(mets["val"].get("spearman_ic", 0.0)))
            test_ic = _sign(float(mets["test"].get("spearman_ic", 0.0)))
            fit_sp = _sign(float(mets["fit"].get("q_high_minus_low", 0.0)))
            val_sp = _sign(float(mets["val"].get("q_high_minus_low", 0.0)))
            test_sp = _sign(float(mets["test"].get("q_high_minus_low", 0.0)))
            flags = {
                "fit_val_ic_flip": bool(fit_ic and val_ic and fit_ic != val_ic),
                "val_test_ic_flip": bool(val_ic and test_ic and val_ic != test_ic),
                "fit_val_spread_flip": bool(fit_sp and val_sp and fit_sp != val_sp),
                "val_test_spread_flip": bool(val_sp and test_sp and val_sp != test_sp),
            }
            for k, v in flags.items():
                aggregate_flip_counts[k] += int(v)
            strength = abs(float(mets["val"].get("spearman_ic", 0.0))) + abs(float(mets["val"].get("q_high_minus_low", 0.0))) * 100.0
            feature_rows.append({"feature": feat, "metrics": mets, "flip_flags": flags, "val_strength": strength})
        feature_rows.sort(key=lambda r: float(r["val_strength"]), reverse=True)
        test_sim = fold_report.get("test", {}).get("test_backtest", {}).get("sim")
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "fold": fold,
                "status": fold_report.get("status"),
                "test_sim": test_sim,
                "is_bad_test": bool(test_sim and float(test_sim.get("cagr_to_strict_mdd", 0.0) or 0.0) < 0.0),
                "top_features": feature_rows[:20],
                "flip_summary_top20": {
                    k: sum(1 for fr in feature_rows[:20] if fr["flip_flags"].get(k))
                    for k in aggregate_flip_counts
                },
            }
        )
    out = {
        "config": asdict(cfg),
        "rows": {"input": len(rows), "first_date": _date(rows[0]) if rows else None, "last_date": _date(rows[-1]) if rows else None},
        "features": features,
        "aggregate_flip_counts_all_fold_features": aggregate_flip_counts,
        "folds": fold_reports,
        "leakage_note": "diagnostic audit reads test rewards after the fact; do not use its test metrics for live policy selection",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit relation flips across walk-forward folds")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--walkforward-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--features", default=RelationFlipAuditCfg.features)
    p.add_argument("--top-n-features", type=int, default=RelationFlipAuditCfg.top_n_features)
    p.add_argument("--min-rows", type=int, default=RelationFlipAuditCfg.min_rows)
    p.add_argument("--quantile", type=float, default=RelationFlipAuditCfg.quantile)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RelationFlipAuditCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
