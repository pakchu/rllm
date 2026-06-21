"""Sweep wave_trading teacher entry thresholds with 2026 held out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.validate_wave_trading_best import _build_best_features, _load_wave_module, _stats


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def walk_forward(psr, data: dict, *, eval_start: str, eval_end: str, lr_C: float, lr_penalty: str, long_th: float, short_th: float) -> dict:
    X = data["X"]
    dates = np.asarray(data["dates"], dtype="datetime64[ns]")
    y_label = data["y_label"]
    long_ret = data["long_ret"]
    short_ret = data["short_ret"]
    holding = int(data["params"]["holding_period"])
    bars_per_month = int(data["bars_per_month"])
    train_bars = int(data["params"]["train_months"] * bars_per_month)
    test_bars = int(data["params"]["test_months"] * bars_per_month)
    purge_gap = holding * 2
    start_dt = np.datetime64(eval_start)
    end_dt = np.datetime64(eval_end)
    valid = (~np.any(~np.isfinite(X), axis=1)) & np.isfinite(y_label) & np.isfinite(long_ret) & np.isfinite(short_ret) & (y_label != 0)
    y_bin = (y_label > 0).astype(np.int32)
    trade_rets: list[float] = []
    folds = []
    pos = max(0, int(np.searchsorted(dates, start_dt)) - purge_gap - train_bars)
    while pos + train_bars + purge_gap < len(X):
        train_start = pos
        train_end = pos + train_bars
        test_start = train_end + purge_gap
        test_end = min(test_start + test_bars, len(X))
        if dates[test_start] > end_dt:
            break
        if dates[test_end - 1] < start_dt:
            pos += test_bars
            continue
        tr = np.arange(train_start, train_end)
        te = np.arange(test_start, test_end)
        tr = tr[valid[tr]]
        te = te[valid[te] & (dates[te] >= start_dt) & (dates[te] <= end_dt)]
        if len(tr) < 1000 or len(te) < 100:
            pos += test_bars
            continue
        proba = psr._predict_proba_lr(X[tr], y_bin[tr], X[te], lr_C, lr_penalty)
        rets, n_long, n_short, *_ = psr.get_trade_returns_numba(long_ret[te], short_ret[te], proba, long_th, short_th, holding, psr.TOTAL_COST)
        trade_rets.extend([float(x) for x in rets])
        folds.append({"start": str(dates[te[0]]), "end": str(dates[te[-1]]), "rows": int(len(te)), "trades": int(len(rets)), "n_long": int(n_long), "n_short": int(n_short), "return_pct": float((np.prod(1 + rets) - 1) * 100.0) if len(rets) else 0.0})
        pos += test_bars
    years = (np.datetime64(eval_end) - np.datetime64(eval_start)).astype("timedelta64[s]").astype(float) / (365.25 * 24 * 3600)
    return {"period": {"start": eval_start, "end": eval_end, "years": years}, "stats": _stats(trade_rets, years), "folds": folds}


def score(test: dict, *, min_trades: int) -> float:
    s = test["stats"]
    if s["trades"] < min_trades or s["cagr_pct"] <= 0:
        return -1000 + s["trades"] / 1000 + s["cagr_pct"] / 1000
    return s["calmar"] + min(1.0, s["trades"] / 200.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--end-date", default="2026-06-02")
    ap.add_argument("--output", required=True)
    ap.add_argument("--lr-c", type=float, default=0.05)
    ap.add_argument("--lr-penalty", default="l1")
    ap.add_argument("--long-thresholds", default="0.54,0.57,0.60,0.63,0.66")
    ap.add_argument("--short-thresholds", default="0.35,0.38,0.41,0.44,0.47")
    ap.add_argument("--min-test-trades", type=int, default=100)
    args = ap.parse_args()
    psr = _load_wave_module(args.wave_root)
    data = _build_best_features(psr, start_date=args.start_date, end_date=args.end_date, time_interval="15m")
    rows = []
    for lt in _parse_floats(args.long_thresholds):
        for st in _parse_floats(args.short_thresholds):
            if st >= lt:
                continue
            test = walk_forward(psr, data, eval_start="2024-07-01", eval_end="2025-12-31 23:59:59", lr_C=args.lr_c, lr_penalty=args.lr_penalty, long_th=lt, short_th=st)
            ev = None
            sc = score(test, min_trades=args.min_test_trades)
            if sc > -999 or test["stats"]["cagr_pct"] > 0:
                ev = walk_forward(psr, data, eval_start="2026-01-01", eval_end="2026-06-01 00:00:00", lr_C=args.lr_c, lr_penalty=args.lr_penalty, long_th=lt, short_th=st)
            rows.append({"long_th": lt, "short_th": st, "selection_score": sc, "test": test, "eval": ev})
    rows.sort(key=lambda r: (r["selection_score"], (r.get("eval") or {"stats": {"calmar": -999}})["stats"]["calmar"]), reverse=True)
    report = {"config": vars(args), "candidates": rows, "top_by_selection": rows[:20], "leakage_guard": {"threshold_selection_uses_test_only": True, "eval_not_used_for_selection": True}}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for r in rows[:20]:
        print(json.dumps({"long_th": r["long_th"], "short_th": r["short_th"], "score": r["selection_score"], "test": r["test"]["stats"], "eval": None if r["eval"] is None else r["eval"]["stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
