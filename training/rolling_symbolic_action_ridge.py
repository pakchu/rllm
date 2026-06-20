"""Monthly prior-only retraining for symbolic action ridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.symbolic_action_ridge import FeatureSpace, choose_best, fit_ridge, load_jsonl, target_value, write_jsonl, _candidate_rows_from_preds
import numpy as np


def _date(row: dict[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(str(row.get("date")))


def rolling_predict(*, history_jsonl: str, eval_jsonl: str, predictions_output: str, summary_output: str, start_date: str, end_date: str, alpha: float = 10000.0, threshold: float = 0.003, min_gap: float = 0.0, target: str = "net_return", min_feature_count: int = 5) -> dict[str, Any]:
    history = load_jsonl(history_jsonl)
    ev = load_jsonl(eval_jsonl)
    all_rows = sorted(history + ev, key=_date)
    eval_rows = [_r for _r in ev if pd.Timestamp(start_date) <= _date(_r) < pd.Timestamp(end_date)]
    months = pd.date_range(pd.Timestamp(start_date).replace(day=1), pd.Timestamp(end_date), freq="MS")
    out = []
    month_summaries = []
    for mstart in months:
        mend = mstart + pd.offsets.MonthBegin(1)
        if mstart >= pd.Timestamp(end_date):
            continue
        train = [r for r in all_rows if _date(r) < mstart]
        test = [r for r in eval_rows if mstart <= _date(r) < min(mend, pd.Timestamp(end_date))]
        if not test:
            continue
        fs = FeatureSpace.fit(train, min_count=int(min_feature_count))
        x_train = fs.matrix(train)
        y = np.asarray([target_value(r, target=target) for r in train], dtype=np.float64)
        w = fit_ridge(x_train, y, alpha=float(alpha))
        preds = fs.matrix(test) @ w
        chosen = choose_best(_candidate_rows_from_preds(test, preds), threshold=float(threshold), min_gap=float(min_gap))
        out.extend(chosen)
        month_summaries.append({"month": str(mstart.date())[:7], "train_rows": len(train), "candidate_rows": len(test), "signals": len(chosen), "trade_signals": sum(1 for r in chosen if r["prediction"].get("gate") == "TRADE"), "features": len(fs.vocab)})
    write_jsonl(predictions_output, out)
    report = {"config": {"alpha": alpha, "threshold": threshold, "min_gap": min_gap, "target": target, "min_feature_count": min_feature_count}, "history_jsonl": history_jsonl, "eval_jsonl": eval_jsonl, "predictions_output": predictions_output, "period": {"start": start_date, "end": end_date}, "rows": len(out), "months": month_summaries, "leakage_guard": {"each_month_fit_uses_rows_before_month_start_only": True, "current_month_labels_not_used_for_current_month": True, "config_fixed_before_eval": True}}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly prior-only symbolic ridge retraining")
    p.add_argument("--history-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--alpha", type=float, default=10000.0)
    p.add_argument("--threshold", type=float, default=0.003)
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--target", choices=["utility", "net_return", "risk_adjusted"], default="net_return")
    p.add_argument("--min-feature-count", type=int, default=5)
    p.add_argument("--market-csv", default="")
    p.add_argument("--backtest-output", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    rep = rolling_predict(history_jsonl=a.history_jsonl, eval_jsonl=a.eval_jsonl, predictions_output=a.predictions_output, summary_output=a.summary_output, start_date=a.start_date, end_date=a.end_date, alpha=a.alpha, threshold=a.threshold, min_gap=a.min_gap, target=a.target, min_feature_count=a.min_feature_count)
    out: dict[str, Any] = {"summary": rep}
    if a.market_csv and a.backtest_output:
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=a.predictions_output, market_csv=a.market_csv, output=a.backtest_output))
        out = {"summary": rep, "backtest": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}}
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
