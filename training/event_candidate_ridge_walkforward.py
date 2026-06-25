"""Rolling no-leak walk-forward for ridge expected-return event rankers.

For each fold:
- fit window trains ridge expected-return scores.
- validation window selects score quantile/full-margin and must pass evidence gates.
- test window refits on fit+validation only, then applies the selected policy.

No test rows are used for fitting, threshold selection, or gate decisions.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_pairwise_walkforward import WalkForwardFold, make_folds
from training.event_candidate_ridge_ranker import _best_by_signal, _date, _fit_score, _load, _write_policy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventCandidateRidgeWalkForwardCfg:
    input_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_ridge_walkforward"
    start_date: str = ""
    end_date: str = ""
    fit_months: int = 6
    val_months: int = 1
    test_months: int = 1
    step_months: int = 1
    ridge_alpha: float = 100.0
    quantiles: str = "0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.5,1.0"
    min_fit_signals: int = 100
    min_val_trades: int = 20
    min_test_signals: int = 10
    min_val_cagr_pct: float = -999.0
    min_val_ratio: float = -999.0
    max_val_strict_mdd_pct: float = 0.0
    max_val_p_value: float = 1.0
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _in_half_open(row: dict[str, Any], start: str, end: str) -> bool:
    d = _date(row)
    return start <= d < end


def _count_signals(rows: list[dict[str, Any]]) -> int:
    return len({int(r.get("signal_pos", -1) or -1) for r in rows})


def _no_trade_predictions(rows: list[dict[str, Any]], output: str, *, reason: str) -> dict[str, Any]:
    best = _best_by_signal(rows, np.zeros(len(rows), dtype=float)) if rows else []
    out: list[dict[str, Any]] = []
    for item in best:
        r = item["row"]
        out.append(
            {
                "date": r.get("date"),
                "signal_pos": r.get("signal_pos"),
                "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "ridge_walkforward", "reason": reason},
                "position_scale": 0.0,
                "score": 0.0,
                "side_candidate": "NONE",
            }
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + ("\n" if out else ""))
    return {"rows": len(out), "counts": {"TRADE": 0, "NO_TRADE": len(out), "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}, "reason": reason, "output": output}


def _passes_validation(sim: dict[str, Any], stats: dict[str, Any], cfg: EventCandidateRidgeWalkForwardCfg) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(sim.get("trade_entries", 0) or 0) < int(cfg.min_val_trades):
        reasons.append("val_trades_below_min")
    if float(sim.get("cagr_pct", -999.0) or -999.0) < float(cfg.min_val_cagr_pct):
        reasons.append("val_cagr_below_min")
    if float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0) < float(cfg.min_val_ratio):
        reasons.append("val_ratio_below_min")
    if float(cfg.max_val_strict_mdd_pct) > 0.0 and float(sim.get("strict_mdd_pct", 999.0) or 999.0) > float(cfg.max_val_strict_mdd_pct):
        reasons.append("val_mdd_above_max")
    if float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0) > float(cfg.max_val_p_value):
        reasons.append("val_p_value_above_max")
    return not reasons, reasons


def _select_on_validation(
    fit_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    cfg: EventCandidateRidgeWalkForwardCfg,
    tmp: Path,
    fold_id: int,
) -> dict[str, Any]:
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    fit_scores, val_scores, names, fit_meta = _fit_score(fit_rows, val_rows, cfg.ridge_alpha)
    fit_best_scores = np.asarray([x["score"] for x in _best_by_signal(fit_rows, fit_scores)], dtype=float)
    val_best = _best_by_signal(val_rows, val_scores)
    candidates: list[dict[str, Any]] = []
    for q in qs:
        threshold = float(np.quantile(fit_best_scores, q)) if len(fit_best_scores) else 999.0
        for margin in margins:
            pred = tmp / f"fold{fold_id:03d}_val_q{q}_m{margin}.jsonl"
            ps = _write_policy(val_best, str(pred), threshold, margin)
            bt = run_overlay(
                OnlineRiskOverlayConfig(
                    predictions_jsonl=str(pred),
                    market_csv=cfg.market_csv,
                    output=str(tmp / f"fold{fold_id:03d}_val_q{q}_m{margin}.bt.json"),
                    leverage=cfg.leverage,
                    entry_delay_bars=cfg.entry_delay_bars,
                )
            )
            sim = bt["sim"]
            stats = bt["trade_stats"]
            passed, reasons = _passes_validation(sim, stats, cfg)
            score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
            if not passed:
                score -= 1000.0 + 10.0 * len(reasons)
            candidates.append(
                {
                    "q": q,
                    "full_margin": margin,
                    "threshold": threshold,
                    "prediction_summary": ps,
                    "val_sim": sim,
                    "val_trade_stats": stats,
                    "validation_passed": passed,
                    "validation_reject_reasons": reasons,
                    "score": score,
                }
            )
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"validation_passed": False, "validation_reject_reasons": ["no_candidates"], "q": 1.0, "full_margin": 0.0, "score": -9999.0}
    return {"selected": selected, "top5": candidates[:5], "fit_coefficients": fit_meta, "features": {"numeric": len(names[0]), "categorical": len(names[1]), "expanded": fit_meta["expanded_feature_count"]}}


def _trade_test_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    cfg: EventCandidateRidgeWalkForwardCfg,
    fold_dir: Path,
    fold_id: int,
) -> dict[str, Any]:
    pred_path = fold_dir / f"fold{fold_id:03d}_test_predictions.jsonl"
    train_scores, test_scores, _names, fit_meta = _fit_score(train_rows, test_rows, cfg.ridge_alpha)
    train_best_scores = np.asarray([x["score"] for x in _best_by_signal(train_rows, train_scores)], dtype=float)
    threshold = float(np.quantile(train_best_scores, float(selected["q"]))) if len(train_best_scores) else 999.0
    test_best = _best_by_signal(test_rows, test_scores)
    ps = _write_policy(test_best, str(pred_path), threshold, float(selected["full_margin"]))
    bt = run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=str(pred_path),
            market_csv=cfg.market_csv,
            output=str(fold_dir / f"fold{fold_id:03d}_test_backtest.json"),
            leverage=cfg.leverage,
            entry_delay_bars=cfg.entry_delay_bars,
        )
    )
    return {"prediction_path": str(pred_path), "prediction_summary": ps, "test_backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}, "test_threshold": threshold, "fit_coefficients": fit_meta}


def run(cfg: EventCandidateRidgeWalkForwardCfg) -> dict[str, Any]:
    rows = sorted(_load(cfg.input_jsonl), key=lambda r: (_date(r), int(r.get("signal_pos", -1) or -1), str(r.get("side", ""))))
    if not rows:
        raise ValueError(f"no rows loaded from {cfg.input_jsonl}")
    folds = make_folds(_date(rows[0]), _date(rows[-1]), cfg)  # type: ignore[arg-type]
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, Any]] = []
    pred_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rllm_ridge_wf_val_") as tmp_raw:
        tmp = Path(tmp_raw)
        for fold in folds:
            fit = [r for r in rows if _in_half_open(r, fold.fit_start, fold.fit_end)]
            val = [r for r in rows if _in_half_open(r, fold.val_start, fold.val_end)]
            test = [r for r in rows if _in_half_open(r, fold.test_start, fold.test_end)]
            fold_dir = work_dir / f"fold{fold.fold_id:03d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            report: dict[str, Any] = {
                "fold": asdict(fold),
                "rows": {"fit": len(fit), "val": len(val), "test": len(test), "fit_signals": _count_signals(fit), "val_signals": _count_signals(val), "test_signals": _count_signals(test)},
            }
            pred_path = fold_dir / f"fold{fold.fold_id:03d}_test_predictions.jsonl"
            if _count_signals(fit) < int(cfg.min_fit_signals) or _count_signals(test) < int(cfg.min_test_signals) or not val:
                ps = _no_trade_predictions(test, str(pred_path), reason="insufficient_fold_rows")
                report.update({"status": "ABSTAIN", "test_prediction_summary": ps})
                pred_paths.append(str(pred_path))
                fold_reports.append(report)
                continue
            selection = _select_on_validation(fit, val, cfg, tmp, fold.fold_id)
            report["selection"] = selection
            if not bool(selection["selected"].get("validation_passed", False)):
                ps = _no_trade_predictions(test, str(pred_path), reason="validation_gate_failed")
                report.update({"status": "ABSTAIN", "test_prediction_summary": ps})
                pred_paths.append(str(pred_path))
                fold_reports.append(report)
                continue
            test_result = _trade_test_fold(fit + val, test, selection["selected"], cfg, fold_dir, fold.fold_id)
            report.update({"status": "TRADED", "test": test_result})
            pred_paths.append(str(pred_path))
            fold_reports.append(report)
    combined_path = work_dir / "combined_test_predictions.jsonl"
    with combined_path.open("w") as out:
        for path in pred_paths:
            p = Path(path)
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        out.write(line + "\n")
    bt = run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=str(combined_path),
            market_csv=cfg.market_csv,
            output=str(work_dir / "combined_test_backtest.json"),
            leverage=cfg.leverage,
            entry_delay_bars=cfg.entry_delay_bars,
        )
    )
    report = {
        "config": asdict(cfg),
        "rows": {"input": len(rows), "signals": _count_signals(rows), "first_date": _date(rows[0]), "last_date": _date(rows[-1])},
        "folds": fold_reports,
        "aggregate": {"prediction_file": str(combined_path), "prediction_files": pred_paths, "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}},
        "leakage_guard": {"folds_are_time_ordered": True, "validation_selects_policy_before_test": True, "test_rows_never_used_for_fit_threshold_or_gate": True, "test_model_refit_uses_fit_plus_validation_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling validation-gated ridge event candidate walk-forward")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=EventCandidateRidgeWalkForwardCfg.work_dir)
    p.add_argument("--start-date", default=EventCandidateRidgeWalkForwardCfg.start_date)
    p.add_argument("--end-date", default=EventCandidateRidgeWalkForwardCfg.end_date)
    p.add_argument("--fit-months", type=int, default=EventCandidateRidgeWalkForwardCfg.fit_months)
    p.add_argument("--val-months", type=int, default=EventCandidateRidgeWalkForwardCfg.val_months)
    p.add_argument("--test-months", type=int, default=EventCandidateRidgeWalkForwardCfg.test_months)
    p.add_argument("--step-months", type=int, default=EventCandidateRidgeWalkForwardCfg.step_months)
    p.add_argument("--ridge-alpha", type=float, default=EventCandidateRidgeWalkForwardCfg.ridge_alpha)
    p.add_argument("--quantiles", default=EventCandidateRidgeWalkForwardCfg.quantiles)
    p.add_argument("--full-margins", default=EventCandidateRidgeWalkForwardCfg.full_margins)
    p.add_argument("--min-fit-signals", type=int, default=EventCandidateRidgeWalkForwardCfg.min_fit_signals)
    p.add_argument("--min-val-trades", type=int, default=EventCandidateRidgeWalkForwardCfg.min_val_trades)
    p.add_argument("--min-test-signals", type=int, default=EventCandidateRidgeWalkForwardCfg.min_test_signals)
    p.add_argument("--min-val-cagr-pct", type=float, default=EventCandidateRidgeWalkForwardCfg.min_val_cagr_pct)
    p.add_argument("--min-val-ratio", type=float, default=EventCandidateRidgeWalkForwardCfg.min_val_ratio)
    p.add_argument("--max-val-strict-mdd-pct", type=float, default=EventCandidateRidgeWalkForwardCfg.max_val_strict_mdd_pct)
    p.add_argument("--max-val-p-value", type=float, default=EventCandidateRidgeWalkForwardCfg.max_val_p_value)
    p.add_argument("--leverage", type=float, default=EventCandidateRidgeWalkForwardCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=EventCandidateRidgeWalkForwardCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateRidgeWalkForwardCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
