"""Side-specialist symbolic ridge ranker.

Fits separate symbolic ridge models for LONG and SHORT action rows so short
selection is not calibrated by long-heavy regimes.  The prediction artifact stays
compatible with the strict backtester.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.symbolic_action_ridge import (
    FeatureSpace,
    _candidate_rows_from_preds,
    choose_best,
    fit_ridge,
    load_jsonl,
    target_value,
    write_jsonl,
)


def _side(row: dict[str, Any]) -> str:
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    return str(action.get("side", "NONE")).upper()


def _fit_side(rows: list[dict[str, Any]], *, side: str, target: str, alpha: float, min_feature_count: int) -> dict[str, Any]:
    side_rows = [r for r in rows if _side(r) == side]
    if len(side_rows) < 100:
        raise ValueError(f"not enough {side} rows: {len(side_rows)}")
    fs = FeatureSpace.fit(side_rows, min_count=min_feature_count)
    x = fs.matrix(side_rows)
    y = np.asarray([target_value(r, target=target) for r in side_rows], dtype=np.float64)
    w = fit_ridge(x, y, alpha=float(alpha))
    pred = x @ w
    corr = 0.0 if len(y) < 2 or np.std(y) < 1e-12 or np.std(pred) < 1e-12 else float(np.corrcoef(y, pred)[0, 1])
    return {"side": side, "rows": side_rows, "fs": fs, "w": w, "fit": {"rows": len(side_rows), "features": len(fs.vocab), "train_corr": corr, "train_rmse_pct": math.sqrt(float(np.mean((pred - y) ** 2))) * 100.0}}


def _predict_rows(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for side, model in models.items():
        side_rows = [r for r in rows if _side(r) == side]
        if not side_rows:
            continue
        preds = model["fs"].matrix(side_rows) @ model["w"]
        out.extend(_candidate_rows_from_preds(side_rows, preds))
    out.sort(key=lambda r: (str(r["date"]), int(r["signal_pos"]), str(r["action"].get("side")), str(r["action"].get("family")), int(r["action"].get("hold_bars", 0))))
    return out


def train_predict(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    predictions_output: str,
    alpha: float = 10000.0,
    threshold: float = 0.003,
    min_gap: float = 0.0,
    target: str = "net_return",
    min_feature_count: int = 5,
) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    ev = load_jsonl(eval_jsonl)
    models = {side: _fit_side(train, side=side, target=target, alpha=alpha, min_feature_count=min_feature_count) for side in ("LONG", "SHORT")}
    chosen = choose_best(_predict_rows(ev, models), threshold=float(threshold), min_gap=float(min_gap))
    write_jsonl(predictions_output, chosen)
    return {
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "predictions_output": predictions_output,
        "config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count},
        "fits": {side: {k: v for k, v in model["fit"].items()} for side, model in models.items()},
        "chosen_counts": dict(Counter(f"{r['prediction']['gate']}/{r['prediction'].get('side')}" for r in chosen)),
        "leakage_guard": {"side_models_fit_on_train_only": True, "eval_labels_not_used_for_fit": True},
    }


def _date(row: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(str(row.get("date")))


def rolling(
    *,
    history_jsonl: str,
    eval_jsonl: str,
    predictions_output: str,
    summary_output: str,
    start_date: str,
    end_date: str,
    alpha: float = 10000.0,
    threshold: float = 0.003,
    min_gap: float = 0.0,
    target: str = "net_return",
    min_feature_count: int = 5,
) -> dict[str, Any]:
    history = load_jsonl(history_jsonl)
    ev = load_jsonl(eval_jsonl)
    all_rows = sorted(history + ev, key=_date)
    eval_rows = [r for r in ev if pd.Timestamp(start_date) <= _date(r) < pd.Timestamp(end_date)]
    months = pd.date_range(pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS")
    out: list[dict[str, Any]] = []
    month_summaries: list[dict[str, Any]] = []
    for mstart in months:
        if mstart >= pd.Timestamp(end_date):
            continue
        mend = min(mstart + pd.offsets.MonthBegin(1), pd.Timestamp(end_date))
        train = [r for r in all_rows if _date(r) < mstart]
        test = [r for r in eval_rows if mstart <= _date(r) < mend]
        if not test:
            continue
        models = {side: _fit_side(train, side=side, target=target, alpha=alpha, min_feature_count=min_feature_count) for side in ("LONG", "SHORT")}
        chosen = choose_best(_predict_rows(test, models), threshold=float(threshold), min_gap=float(min_gap))
        out.extend(chosen)
        month_summaries.append({
            "month": str(mstart.date())[:7],
            "train_rows": len(train),
            "candidate_rows": len(test),
            "signals": len(chosen),
            "trade_signals": sum(1 for r in chosen if r["prediction"].get("gate") == "TRADE"),
            "long_fit": models["LONG"]["fit"],
            "short_fit": models["SHORT"]["fit"],
        })
    write_jsonl(predictions_output, out)
    rep = {
        "config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count},
        "history_jsonl": history_jsonl,
        "eval_jsonl": eval_jsonl,
        "predictions_output": predictions_output,
        "period": {"start": start_date, "end": end_date},
        "rows": len(out),
        "months": month_summaries,
        "leakage_guard": {"each_month_fit_uses_rows_before_month_start_only": True, "side_models_fit_separately": True},
    }
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    return rep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-specialist symbolic ridge")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train-predict")
    tr.add_argument("--train-jsonl", required=True); tr.add_argument("--eval-jsonl", required=True); tr.add_argument("--predictions-output", required=True)
    ro = sub.add_parser("rolling")
    ro.add_argument("--history-jsonl", required=True); ro.add_argument("--eval-jsonl", required=True); ro.add_argument("--predictions-output", required=True); ro.add_argument("--summary-output", required=True); ro.add_argument("--start-date", required=True); ro.add_argument("--end-date", required=True)
    for sp in (tr, ro):
        sp.add_argument("--alpha", type=float, default=10000.0)
        sp.add_argument("--threshold", type=float, default=0.003)
        sp.add_argument("--min-gap", type=float, default=0.0)
        sp.add_argument("--target", choices=["utility", "net_return", "risk_adjusted", "tail_risk", "distributional_safety"], default="net_return")
        sp.add_argument("--min-feature-count", type=int, default=5)
        sp.add_argument("--market-csv", default="")
        sp.add_argument("--backtest-output", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.cmd == "train-predict":
        rep = train_predict(train_jsonl=a.train_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count)
    else:
        rep = rolling(history_jsonl=a.history_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, summary_output=a.summary_output, start_date=a.start_date, end_date=a.end_date, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count)
    out: dict[str, Any] = {"summary": rep}
    if a.market_csv and a.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=a.predictions_output, market_csv=a.market_csv, output=a.backtest_output))
        out["backtest"] = {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
