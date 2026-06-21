"""Export wave_trading 15m rolling teacher predictions onto rllm 5m bars.

Runs with wave_trading's venv.  The output is a dense 15m decision stream mapped
onto the nearest prior/exact 5m BTCUSDT bar, suitable as a teacher/gate signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.validate_wave_trading_best import _build_best_features, _load_wave_module


def _read_5m_dates(path: str) -> np.ndarray:
    df = pd.read_csv(path, usecols=["date"], compression="gzip" if path.endswith(".gz") else None)
    return pd.to_datetime(df["date"], errors="raise").to_numpy(dtype="datetime64[ns]")


def export_teacher(
    *,
    wave_root: str,
    market_5m_csv: str,
    output: str,
    summary_output: str,
    start_date: str,
    end_date: str,
    eval_start: str,
    eval_end: str,
    lr_c: float,
    lr_penalty: str,
    include_no_trade: bool,
) -> dict[str, Any]:
    psr = _load_wave_module(wave_root)
    data = _build_best_features(psr, start_date=start_date, end_date=end_date, time_interval="15m")
    X = data["X"]
    dates15 = np.asarray(data["dates"], dtype="datetime64[ns]")
    y_label = data["y_label"]
    long_ret = data["long_ret"]
    short_ret = data["short_ret"]
    holding = int(data["params"]["holding_period"])
    long_th = float(data["params"]["long_th"])
    short_th = float(data["params"]["short_th"])
    bars_per_month = int(data["bars_per_month"])
    train_bars = int(data["params"]["train_months"] * bars_per_month)
    test_bars = int(data["params"]["test_months"] * bars_per_month)
    purge_gap = holding * 2
    start_dt = np.datetime64(eval_start)
    end_dt = np.datetime64(eval_end)
    valid = (~np.any(~np.isfinite(X), axis=1)) & np.isfinite(y_label) & np.isfinite(long_ret) & np.isfinite(short_ret) & (y_label != 0)
    y_bin = (y_label > 0).astype(np.int32)
    dates5 = _read_5m_dates(market_5m_csv)

    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    pos = max(0, int(np.searchsorted(dates15, start_dt)) - purge_gap - train_bars)
    seen_pos: set[int] = set()
    while pos + train_bars + purge_gap < len(X):
        train_start = pos
        train_end = pos + train_bars
        test_start = train_end + purge_gap
        test_end = min(test_start + test_bars, len(X))
        if dates15[test_start] > end_dt:
            break
        if dates15[test_end - 1] < start_dt:
            pos += test_bars
            continue
        tr = np.arange(train_start, train_end)
        te = np.arange(test_start, test_end)
        tr = tr[valid[tr]]
        te = te[valid[te] & (dates15[te] >= start_dt) & (dates15[te] <= end_dt)]
        if len(tr) < 1000 or len(te) < 100:
            pos += test_bars
            continue
        proba = psr._predict_proba_lr(X[tr], y_bin[tr], X[te], lr_c, lr_penalty)
        fold_trade = 0
        fold_long = 0
        fold_short = 0
        for idx, p in zip(te, proba):
            dt = dates15[idx]
            # exact/prior 5m position; 15m timestamps should usually be exact.
            p5 = int(np.searchsorted(dates5, dt, side="right") - 1)
            if p5 < 0 or p5 in seen_pos:
                continue
            seen_pos.add(p5)
            prob = float(p)
            if prob >= long_th:
                side = "LONG"
            elif prob <= short_th:
                side = "SHORT"
            else:
                side = "NONE"
            if side == "NONE" and not include_no_trade:
                continue
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "wave_teacher", "confidence": "HIGH"}
            if side != "NONE":
                pred = {"gate": "TRADE", "side": side, "hold_bars": int(holding * 3), "family": "wave_teacher", "confidence": "HIGH"}
                fold_trade += 1
                fold_long += int(side == "LONG")
                fold_short += int(side == "SHORT")
            rows.append({"date": str(pd.Timestamp(dt)), "signal_pos": p5, "prediction": pred, "teacher_probability_long": prob, "teacher_thresholds": {"long": long_th, "short": short_th}, "teacher_15m_index": int(idx)})
        folds.append({"start": str(pd.Timestamp(dates15[te[0]])), "end": str(pd.Timestamp(dates15[te[-1]])), "rows": int(len(te)), "trade_rows": fold_trade, "long": fold_long, "short": fold_short})
        pos += test_bars
    rows.sort(key=lambda r: (r["signal_pos"], r["date"]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    trade_rows = [r for r in rows if r["prediction"]["gate"] == "TRADE"]
    summary = {
        "output": output,
        "rows": len(rows),
        "trade_rows": len(trade_rows),
        "side_counts": {"LONG": sum(1 for r in trade_rows if r["prediction"]["side"] == "LONG"), "SHORT": sum(1 for r in trade_rows if r["prediction"]["side"] == "SHORT")},
        "config": {"wave_root": wave_root, "market_5m_csv": market_5m_csv, "start_date": start_date, "end_date": end_date, "eval_start": eval_start, "eval_end": eval_end, "lr_c": lr_c, "lr_penalty": lr_penalty, "include_no_trade": include_no_trade},
        "teacher_params": data["params"],
        "folds": folds,
        "leakage_guard": {"rolling_train_before_test": True, "purge_gap_bars_15m": purge_gap, "mapped_to_5m_without_future": True},
    }
    Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    ap.add_argument("--market-5m-csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--start-date", default="2020-01-01")
    ap.add_argument("--end-date", default="2026-06-02")
    ap.add_argument("--eval-start", required=True)
    ap.add_argument("--eval-end", required=True)
    ap.add_argument("--lr-c", type=float, default=0.05)
    ap.add_argument("--lr-penalty", default="l1")
    ap.add_argument("--trades-only", dest="include_no_trade", action="store_false")
    ap.set_defaults(include_no_trade=True)
    args = ap.parse_args()
    print(json.dumps(export_teacher(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
