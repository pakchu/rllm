"""Monthly no-leak selector for regime-conditioned symbolic policies.

For each evaluation month:
1. Use a validation window immediately before the month.
2. For each target/threshold candidate, fit monthly symbolic policy only with rows
   before the validation month and backtest the validation window.
3. Select the best candidate only if validation gates pass.
4. Refit/apply the selected fixed candidate to the eval month using history before
   that month. If no candidate passes, emit NO_TRADE rows for that month.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.regime_conditioned_symbolic_policy import rolling_predict
from training.regime_symbolic_threshold_selector import _candidate_score, _parse_csv
from training.symbolic_action_ridge import load_jsonl, write_jsonl

NO_TRADE = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "family": "MONTHLY_SYMBOLIC_SELECTOR", "confidence": "HIGH"}


@dataclass(frozen=True)
class MonthlySymbolicSelectorCfg:
    history_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/regime_symbolic_monthly_selector"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    validation_months: int = 3
    targets: str = "tail_risk,distributional_safety,risk_adjusted,net_return"
    thresholds: str = "-0.0154,-0.0134,-0.0125,-0.0108,-0.0101,-0.0094,-0.0002,0.0,0.0002"
    alpha: float = 10000.0
    min_gap: float = 0.0
    expert_margin: float = 0.0
    min_feature_count: int = 5
    min_train_rows: int = 2000
    leverage: float = 1.0
    min_val_trades: int = 20
    max_val_mdd_pct: float = 20.0
    min_val_cagr_pct: float = 0.0
    max_val_p_value: float = 0.8


def _date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def _slice_rows(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [r for r in rows if start <= _date(r) < end]


def _history_before(rows: list[dict[str, Any]], end: str) -> list[dict[str, Any]]:
    return [r for r in rows if _date(r) < end]


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    s = pd.Timestamp(start).replace(day=1)
    e = pd.Timestamp(end)
    months = []
    cur = s
    while cur < e:
        months.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)
    return months


def _tag_num(x: float) -> str:
    return str(x).replace("-", "m").replace(".", "p")


def _write_no_trade_month(rows: list[dict[str, Any]], path: Path, reason: str) -> dict[str, Any]:
    by_signal: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        key = (_date(r), int(r.get("signal_pos", -1) or -1))
        by_signal.setdefault(key, r)
    out = [
        {"date": k[0], "signal_pos": k[1], "prediction": {**NO_TRADE, "reason": reason}, "predicted_utility": 0.0, "selected_action": None}
        for k in sorted(by_signal)
    ]
    write_jsonl(path, out)
    return {"rows": len(out), "trade_signals": 0, "reason": reason, "predictions": str(path)}


def _run_policy(*, history_path: str, eval_path: str, start: str, end: str, target: str, threshold: float, cfg: MonthlySymbolicSelectorCfg, out_dir: Path, tag: str) -> dict[str, Any]:
    pred = out_dir / f"{tag}_predictions.jsonl"
    summ = out_dir / f"{tag}_summary.json"
    bt_path = out_dir / f"{tag}_backtest.json"
    summary = rolling_predict(
        history_jsonl=history_path,
        eval_jsonl=eval_path,
        predictions_output=str(pred),
        summary_output=str(summ),
        start_date=start,
        end_date=end,
        alpha=float(cfg.alpha),
        threshold=float(threshold),
        min_gap=float(cfg.min_gap),
        expert_margin=float(cfg.expert_margin),
        target=target,
        min_feature_count=int(cfg.min_feature_count),
        min_train_rows=int(cfg.min_train_rows),
    )
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_csv, output=str(bt_path), leverage=float(cfg.leverage)))
    return {"target": target, "threshold": threshold, "predictions": str(pred), "summary": summary, "backtest": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}}


def run(cfg: MonthlySymbolicSelectorCfg) -> dict[str, Any]:
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    history_all = load_jsonl(cfg.history_jsonl)
    eval_all = load_jsonl(cfg.eval_jsonl)
    combined = sorted(history_all + eval_all, key=_date)
    pred_paths: list[str] = []
    month_reports: list[dict[str, Any]] = []
    gate_cfg = type("GateCfg", (), {
        "min_val_trades": cfg.min_val_trades,
        "min_val_cagr_pct": cfg.min_val_cagr_pct,
        "max_val_mdd_pct": cfg.max_val_mdd_pct,
        "max_val_p_value": cfg.max_val_p_value,
    })()

    for mstart in _month_starts(cfg.eval_start, cfg.eval_end):
        mend = min(mstart + pd.offsets.MonthBegin(1), pd.Timestamp(cfg.eval_end))
        val_end = mstart
        val_start = mstart - pd.DateOffset(months=int(cfg.validation_months))
        month = str(mstart.date())[:7]
        month_dir = work / month
        month_dir.mkdir(parents=True, exist_ok=True)
        hist_rows = _history_before(combined, str(val_start))
        val_rows = _slice_rows(combined, str(val_start), str(val_end))
        eval_rows = _slice_rows(combined, str(mstart), str(mend))
        hist_path = month_dir / "history_before_validation.jsonl"
        val_path = month_dir / "validation.jsonl"
        eval_path = month_dir / "eval_month.jsonl"
        write_jsonl(hist_path, hist_rows)
        write_jsonl(val_path, val_rows)
        write_jsonl(eval_path, eval_rows)
        candidates: list[dict[str, Any]] = []
        for target in _parse_csv(cfg.targets, str):
            for threshold in _parse_csv(cfg.thresholds, float):
                tag = f"val_{target}_th{_tag_num(threshold)}"
                row = _run_policy(history_path=str(hist_path), eval_path=str(val_path), start=str(val_start.date()), end=str(val_end.date()), target=target, threshold=float(threshold), cfg=cfg, out_dir=month_dir, tag=tag)
                score, reasons = _candidate_score(row["backtest"], gate_cfg)  # type: ignore[arg-type]
                row["selection_score"] = score
                row["validation_passed"] = not reasons
                row["validation_reject_reasons"] = reasons
                candidates.append(row)
        candidates.sort(key=lambda r: float(r["selection_score"]), reverse=True)
        selected = candidates[0] if candidates else None
        if selected is None or not bool(selected.get("validation_passed")):
            pred = month_dir / "eval_no_trade_predictions.jsonl"
            no_trade = _write_no_trade_month(eval_rows, pred, "monthly_validation_gate_failed")
            pred_paths.append(str(pred))
            month_reports.append({"month": month, "status": "ABSTAIN", "validation_window": {"start": str(val_start.date()), "end": str(val_end.date())}, "rows": {"history": len(hist_rows), "validation": len(val_rows), "eval": len(eval_rows)}, "selected": selected, "top_validation": candidates[:5], "eval": no_trade})
            continue
        eval_result = _run_policy(history_path=str(hist_path), eval_path=str(eval_path), start=str(mstart.date()), end=str(mend.date()), target=str(selected["target"]), threshold=float(selected["threshold"]), cfg=cfg, out_dir=month_dir, tag=f"eval_selected_{selected['target']}_th{_tag_num(float(selected['threshold']))}")
        pred_paths.append(str(eval_result["predictions"]))
        month_reports.append({"month": month, "status": "TRADED", "validation_window": {"start": str(val_start.date()), "end": str(val_end.date())}, "rows": {"history": len(hist_rows), "validation": len(val_rows), "eval": len(eval_rows)}, "selected": selected, "eval": eval_result})

    combined_pred = work / "combined_eval_predictions.jsonl"
    with combined_pred.open("w") as out:
        for p in pred_paths:
            pp = Path(p)
            if pp.exists():
                for line in pp.read_text().splitlines():
                    if line.strip():
                        out.write(line + "\n")
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(combined_pred), market_csv=cfg.market_csv, output=str(work / "combined_eval_backtest.json"), leverage=float(cfg.leverage)))
    report = {
        "config": asdict(cfg),
        "months": month_reports,
        "aggregate": {"prediction_file": str(combined_pred), "backtest": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}},
        "leakage_guard": {"each_month_validation_precedes_eval_month": True, "eval_month_never_selects_target_threshold": True, "abstain_when_no_validation_candidate_passes": True, "monthly_policy_fit_uses_rows_before_validation_start_for_validation_selection": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly validation-selected symbolic policy")
    p.add_argument("--history-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=MonthlySymbolicSelectorCfg.work_dir)
    p.add_argument("--eval-start", default="2026-01-01")
    p.add_argument("--eval-end", default="2026-06-01")
    p.add_argument("--validation-months", type=int, default=3)
    p.add_argument("--targets", default=MonthlySymbolicSelectorCfg.targets)
    p.add_argument("--thresholds", default=MonthlySymbolicSelectorCfg.thresholds)
    p.add_argument("--alpha", type=float, default=10000.0)
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--expert-margin", type=float, default=0.0)
    p.add_argument("--min-feature-count", type=int, default=5)
    p.add_argument("--min-train-rows", type=int, default=2000)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--min-val-trades", type=int, default=20)
    p.add_argument("--max-val-mdd-pct", type=float, default=20.0)
    p.add_argument("--min-val-cagr-pct", type=float, default=0.0)
    p.add_argument("--max-val-p-value", type=float, default=0.8)
    return p.parse_args()


def main() -> None:
    report = run(MonthlySymbolicSelectorCfg(**vars(parse_args())))
    print(json.dumps({"sim": report["aggregate"]["backtest"]["sim"], "trade_stats": report["aggregate"]["backtest"]["trade_stats"], "months": [{"month": m["month"], "status": m["status"], "selected": None if not m.get("selected") else {"target": m["selected"].get("target"), "threshold": m["selected"].get("threshold"), "passed": m["selected"].get("validation_passed"), "val_sim": m["selected"].get("backtest", {}).get("sim")}} for m in report["months"]]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
