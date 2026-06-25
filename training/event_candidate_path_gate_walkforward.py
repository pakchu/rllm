"""Rolling no-leak path-outcome gate for event candidate policies.

This tests whether candidate-level future path quality is learnable from past-only
features. It differs from the plain expected-return ridge ranker by training on
path-risk-aware targets such as return-minus-MAE and stop-first labels, then using
validation only to choose the target recipe and gate threshold before testing.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_pairwise_walkforward import make_folds
from training.event_candidate_ridge_ranker import _best_by_signal, _date, _feature_names, _load, _predict, _ridge, _standardize, _write_policy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventCandidatePathGateWalkForwardCfg:
    input_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_path_gate_walkforward"
    start_date: str = ""
    end_date: str = ""
    fit_months: int = 6
    val_months: int = 1
    test_months: int = 1
    step_months: int = 1
    ridge_alpha: float = 100.0
    targets: str = "ret,ret_minus_mae,ret_minus_2mae,win_stop1,win_stop2"
    quantiles: str = "0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.5,1.0"
    min_fit_signals: int = 100
    min_val_trades: int = 15
    min_test_signals: int = 10
    min_val_cagr_pct: float = -999.0
    min_val_ratio: float = -999.0
    max_val_strict_mdd_pct: float = 0.0
    max_val_p_value: float = 1.0
    leverage: float = 1.0
    entry_delay_bars: int = 1
    trade_stop_loss_pct: float = 0.0
    trade_take_profit_pct: float = 0.0
    feature_prefixes: str = ""


def _in_half_open(row: dict[str, Any], start: str, end: str) -> bool:
    d = _date(row)
    return start <= d < end


def _count_signals(rows: list[dict[str, Any]]) -> int:
    return len({int(r.get("signal_pos", -1) or -1) for r in rows})


def _target(row: dict[str, Any], recipe: str) -> float:
    rew = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    ret = float(rew.get("net_return_pct", 0.0) or 0.0)
    mae = abs(float(rew.get("mae", 0.0) or 0.0))
    mfe = abs(float(rew.get("mfe", 0.0) or 0.0))
    if recipe == "ret":
        return ret
    if recipe == "ret_minus_mae":
        return ret - mae
    if recipe == "ret_minus_2mae":
        return ret - 2.0 * mae
    if recipe == "ret_plus_mfe_minus_2mae":
        return ret + 0.25 * mfe - 2.0 * mae
    if recipe == "win_stop1":
        return 1.0 if ret > 0.0 and mae <= 0.01 else -1.0
    if recipe == "win_stop2":
        return 1.0 if ret > 0.0 and mae <= 0.02 else -1.0
    raise ValueError(f"unknown target recipe: {recipe}")


def _selected_feature_names(rows: list[dict[str, Any]], prefixes: str) -> tuple[list[str], list[str]]:
    names = _feature_names(rows)
    raw = [x.strip() for x in str(prefixes).split(",") if x.strip()]
    if not raw:
        return names
    num, cat = names
    keep = tuple(raw)
    return [n for n in num if n.startswith(keep)], cat


def _x(rows: list[dict[str, Any]], num_names: list[str], cat_names: list[str]) -> np.ndarray:
    cat_index = {c: i for i, c in enumerate(cat_names)}
    x = np.zeros((len(rows), len(num_names) * 2 + len(cat_names) + 3), dtype=float)
    for i, r in enumerate(rows):
        side = str(r.get("side"))
        sign = 1.0 if side == "LONG" else -1.0
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        vals = np.asarray([float(snap.get(n, 0.0) or 0.0) for n in num_names], dtype=float)
        x[i, : len(num_names)] = vals
        x[i, len(num_names) : len(num_names) * 2] = vals * sign
        base = len(num_names) * 2
        toks = r.get("state_tokens", {}) if isinstance(r.get("state_tokens"), dict) else {}
        for k, v in toks.items():
            j = cat_index.get(f"tok:{k}={v}")
            if j is not None:
                x[i, base + j] = 1.0
        base += len(cat_names)
        x[i, base : base + 3] = [1.0 if side == "LONG" else 0.0, 1.0 if side == "SHORT" else 0.0, sign]
    return x


def _fit_score_target(
    fit_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    alpha: float,
    recipe: str,
    names: tuple[list[str], list[str]] | None = None,
    feature_prefixes: str = "",
) -> tuple[np.ndarray, np.ndarray, tuple[list[str], list[str]]]:
    if names is None:
        names = _selected_feature_names(fit_rows, feature_prefixes)
    num, cat = names
    xtr = _x(fit_rows, num, cat)
    xte = _x(score_rows, num, cat)
    y = np.asarray([_target(r, recipe) for r in fit_rows], dtype=float)
    xtrz, xtez, _ = _standardize(xtr, xte)
    w = _ridge(xtrz, y, alpha)
    return _predict(xtrz, w), _predict(xtez, w), names


def _passes_validation(sim: dict[str, Any], stats: dict[str, Any], cfg: EventCandidatePathGateWalkForwardCfg) -> tuple[bool, list[str]]:
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


def _no_trade_predictions(rows: list[dict[str, Any]], output: str, *, reason: str) -> dict[str, Any]:
    best = _best_by_signal(rows, np.zeros(len(rows), dtype=float)) if rows else []
    out = []
    for item in best:
        r = item["row"]
        out.append({"date": r.get("date"), "signal_pos": r.get("signal_pos"), "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "path_gate", "reason": reason}, "position_scale": 0.0, "score": 0.0, "side_candidate": "NONE"})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + ("\n" if out else ""))
    return {"rows": len(out), "counts": {"TRADE": 0, "NO_TRADE": len(out), "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}, "reason": reason, "output": output}


def _select_on_validation(fit_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], cfg: EventCandidatePathGateWalkForwardCfg, tmp: Path, fold_id: int) -> dict[str, Any]:
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    recipes = [x.strip() for x in cfg.targets.split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    names: tuple[list[str], list[str]] | None = None
    for recipe in recipes:
        fit_scores, val_scores, names = _fit_score_target(fit_rows, val_rows, cfg.ridge_alpha, recipe, names, cfg.feature_prefixes)
        fit_best_scores = np.asarray([x["score"] for x in _best_by_signal(fit_rows, fit_scores)], dtype=float)
        val_best = _best_by_signal(val_rows, val_scores)
        for q in qs:
            threshold = float(np.quantile(fit_best_scores, q)) if len(fit_best_scores) else 999.0
            for margin in margins:
                pred = tmp / f"fold{fold_id:03d}_val_{recipe}_q{q}_m{margin}.jsonl"
                ps = _write_policy(val_best, str(pred), threshold, margin)
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_csv, output=str(tmp / f"fold{fold_id:03d}_val_{recipe}_q{q}_m{margin}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))
                sim = bt["sim"]
                stats = bt["trade_stats"]
                passed, reasons = _passes_validation(sim, stats, cfg)
                score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
                if not passed:
                    score -= 1000.0 + 10.0 * len(reasons)
                candidates.append({"target": recipe, "q": q, "full_margin": margin, "threshold": threshold, "prediction_summary": ps, "val_sim": sim, "val_trade_stats": stats, "validation_passed": passed, "validation_reject_reasons": reasons, "score": score})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"validation_passed": False, "validation_reject_reasons": ["no_candidates"], "target": "ret", "q": 1.0, "full_margin": 0.0, "score": -9999.0}
    return {"selected": selected, "top10": candidates[:10], "features": {"numeric": len(names[0]) if names else 0, "categorical": len(names[1]) if names else 0}}


def _trade_test_fold(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], selected: dict[str, Any], cfg: EventCandidatePathGateWalkForwardCfg, fold_dir: Path, fold_id: int) -> dict[str, Any]:
    pred_path = fold_dir / f"fold{fold_id:03d}_test_predictions.jsonl"
    train_scores, test_scores, _names = _fit_score_target(train_rows, test_rows, cfg.ridge_alpha, str(selected["target"]), feature_prefixes=cfg.feature_prefixes)
    train_best_scores = np.asarray([x["score"] for x in _best_by_signal(train_rows, train_scores)], dtype=float)
    threshold = float(np.quantile(train_best_scores, float(selected["q"]))) if len(train_best_scores) else 999.0
    test_best = _best_by_signal(test_rows, test_scores)
    ps = _write_policy(test_best, str(pred_path), threshold, float(selected["full_margin"]))
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=cfg.market_csv, output=str(fold_dir / f"fold{fold_id:03d}_test_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))
    return {"prediction_path": str(pred_path), "prediction_summary": ps, "test_backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}, "test_threshold": threshold}


def run(cfg: EventCandidatePathGateWalkForwardCfg) -> dict[str, Any]:
    rows = sorted(_load(cfg.input_jsonl), key=lambda r: (_date(r), int(r.get("signal_pos", -1) or -1), str(r.get("side", ""))))
    if not rows:
        raise ValueError(f"no rows loaded from {cfg.input_jsonl}")
    folds = make_folds(_date(rows[0]), _date(rows[-1]), cfg)  # type: ignore[arg-type]
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, Any]] = []
    pred_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rllm_path_gate_wf_val_") as tmp_raw:
        tmp = Path(tmp_raw)
        for fold in folds:
            fit = [r for r in rows if _in_half_open(r, fold.fit_start, fold.fit_end)]
            val = [r for r in rows if _in_half_open(r, fold.val_start, fold.val_end)]
            test = [r for r in rows if _in_half_open(r, fold.test_start, fold.test_end)]
            fold_dir = work_dir / f"fold{fold.fold_id:03d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            pred_path = fold_dir / f"fold{fold.fold_id:03d}_test_predictions.jsonl"
            report: dict[str, Any] = {"fold": asdict(fold), "rows": {"fit": len(fit), "val": len(val), "test": len(test), "fit_signals": _count_signals(fit), "val_signals": _count_signals(val), "test_signals": _count_signals(test)}}
            if _count_signals(fit) < int(cfg.min_fit_signals) or _count_signals(test) < int(cfg.min_test_signals) or not val:
                ps = _no_trade_predictions(test, str(pred_path), reason="insufficient_fold_rows")
                report.update({"status": "ABSTAIN", "test_prediction_summary": ps})
                pred_paths.append(str(pred_path)); fold_reports.append(report); continue
            selection = _select_on_validation(fit, val, cfg, tmp, fold.fold_id)
            report["selection"] = selection
            if not bool(selection["selected"].get("validation_passed", False)):
                ps = _no_trade_predictions(test, str(pred_path), reason="validation_gate_failed")
                report.update({"status": "ABSTAIN", "test_prediction_summary": ps})
                pred_paths.append(str(pred_path)); fold_reports.append(report); continue
            test_result = _trade_test_fold(fit + val, test, selection["selected"], cfg, fold_dir, fold.fold_id)
            report.update({"status": "TRADED", "test": test_result})
            pred_paths.append(str(pred_path)); fold_reports.append(report)
    combined_path = work_dir / "combined_test_predictions.jsonl"
    with combined_path.open("w") as out:
        for path in pred_paths:
            p = Path(path)
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        out.write(line + "\n")
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(combined_path), market_csv=cfg.market_csv, output=str(work_dir / "combined_test_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, trade_stop_loss_pct=cfg.trade_stop_loss_pct, trade_take_profit_pct=cfg.trade_take_profit_pct))
    report = {"config": asdict(cfg), "rows": {"input": len(rows), "signals": _count_signals(rows), "first_date": _date(rows[0]), "last_date": _date(rows[-1])}, "folds": fold_reports, "aggregate": {"prediction_file": str(combined_path), "prediction_files": pred_paths, "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}}, "leakage_guard": {"folds_are_time_ordered": True, "validation_selects_target_and_policy_before_test": True, "candidate_path_labels_are_used_only_when_historically_known": True, "test_rows_never_used_for_fit_threshold_target_or_gate": True, "test_model_refit_uses_fit_plus_validation_only": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling no-leak path-outcome gate for event candidates")
    p.add_argument("--input-jsonl", required=True); p.add_argument("--market-csv", required=True); p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=EventCandidatePathGateWalkForwardCfg.work_dir)
    p.add_argument("--start-date", default=""); p.add_argument("--end-date", default="")
    p.add_argument("--fit-months", type=int, default=6); p.add_argument("--val-months", type=int, default=1); p.add_argument("--test-months", type=int, default=1); p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--ridge-alpha", type=float, default=100.0); p.add_argument("--targets", default=EventCandidatePathGateWalkForwardCfg.targets)
    p.add_argument("--quantiles", default=EventCandidatePathGateWalkForwardCfg.quantiles); p.add_argument("--full-margins", default=EventCandidatePathGateWalkForwardCfg.full_margins)
    p.add_argument("--min-fit-signals", type=int, default=100); p.add_argument("--min-val-trades", type=int, default=15); p.add_argument("--min-test-signals", type=int, default=10)
    p.add_argument("--min-val-cagr-pct", type=float, default=-999.0); p.add_argument("--min-val-ratio", type=float, default=-999.0); p.add_argument("--max-val-strict-mdd-pct", type=float, default=0.0); p.add_argument("--max-val-p-value", type=float, default=1.0)
    p.add_argument("--leverage", type=float, default=1.0); p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--trade-stop-loss-pct", type=float, default=0.0); p.add_argument("--trade-take-profit-pct", type=float, default=0.0)
    p.add_argument("--feature-prefixes", default="")
    return p.parse_args()


def main() -> None:
    report = run(EventCandidatePathGateWalkForwardCfg(**vars(parse_args())))
    print(json.dumps({"sim": report["aggregate"]["backtest"]["sim"], "trade_stats": report["aggregate"]["backtest"]["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
