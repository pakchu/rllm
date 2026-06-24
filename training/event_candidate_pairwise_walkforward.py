"""Rolling validation-gated walk-forward for event candidate pairwise rankers.

Each fold uses only past rows:
- fit window trains a pairwise winner-vs-loser candidate ranker.
- validation window selects quantile/full-margin policy parameters.
- test window is traded only when validation clears configured evidence gates.

The test fold may refit on fit+validation rows because those labels are known before
the next test window starts. No test rows are used for fitting, threshold selection,
or gate decisions.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_pairwise_ranker import EventCandidatePairwiseRankerCfg, _fit_score
from training.event_candidate_ridge_ranker import _best_by_signal, _date, _load, _write_policy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    fit_start: str
    fit_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class EventCandidatePairwiseWalkForwardCfg:
    input_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_pairwise_walkforward"
    start_date: str = ""
    end_date: str = ""
    fit_months: int = 12
    val_months: int = 6
    test_months: int = 6
    step_months: int = 6
    max_pairs_per_signal: int = 8
    min_utility_gap: float = 0.001
    lr: float = 0.2
    l2: float = 10.0
    epochs: int = 200
    quantiles: str = "0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.5,1.0"
    min_fit_signals: int = 100
    min_val_trades: int = 20
    min_test_signals: int = 20
    min_val_cagr_pct: float = 0.0
    min_val_ratio: float = 0.0
    max_val_strict_mdd_pct: float = 25.0
    min_val_t_stat: float = -999.0
    max_val_p_value: float = 1.0
    max_val_power_gap: int = -1
    leverage: float = 1.0
    entry_delay_bars: int = 1
    pair_half_life_days: float = 0.0
    side_min_val_trades: int = 0
    side_min_val_mean_ret_pct: float = -999.0
    side_scale_val_mean_ret_pct: float = 0.0


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _month_floor(value: str) -> datetime:
    dt = _parse_dt(value)
    return datetime(dt.year, dt.month, 1)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _add_months(dt: datetime, months: int) -> datetime:
    month0 = dt.month - 1 + int(months)
    year = dt.year + month0 // 12
    month = month0 % 12 + 1
    return datetime(year, month, 1)


def make_folds(first_date: str, last_date: str, cfg: EventCandidatePairwiseWalkForwardCfg) -> list[WalkForwardFold]:
    start = _month_floor(cfg.start_date or first_date)
    last = _parse_dt(cfg.end_date or last_date)
    folds: list[WalkForwardFold] = []
    fold_id = 0
    cur = start
    while True:
        fit_start = cur
        fit_end = _add_months(fit_start, cfg.fit_months)
        val_end = _add_months(fit_end, cfg.val_months)
        test_end = _add_months(val_end, cfg.test_months)
        if val_end > last:
            break
        bounded_test_end = min(test_end, last)
        if bounded_test_end <= val_end:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                fit_start=_fmt(fit_start),
                fit_end=_fmt(fit_end),
                val_start=_fmt(fit_end),
                val_end=_fmt(val_end),
                test_start=_fmt(val_end),
                test_end=_fmt(bounded_test_end),
            )
        )
        fold_id += 1
        cur = _add_months(cur, cfg.step_months)
    return folds


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
                "prediction": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "pairwise_walkforward", "reason": reason},
                "position_scale": 0.0,
                "score": 0.0,
                "side_candidate": "NONE",
            }
        )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + ("\n" if out else ""))
    return {"rows": len(out), "counts": {"TRADE": 0, "NO_TRADE": len(out), "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}, "reason": reason, "output": output}


def _pairwise_cfg(cfg: EventCandidatePairwiseWalkForwardCfg, train_path: str, eval_path: str) -> EventCandidatePairwiseRankerCfg:
    return EventCandidatePairwiseRankerCfg(
        train_jsonl=train_path,
        eval_jsonl=eval_path,
        market_csv=cfg.market_csv,
        output=cfg.output,
        work_dir=cfg.work_dir,
        max_pairs_per_signal=cfg.max_pairs_per_signal,
        min_utility_gap=cfg.min_utility_gap,
        lr=cfg.lr,
        l2=cfg.l2,
        epochs=cfg.epochs,
        quantiles=cfg.quantiles,
        full_margins=cfg.full_margins,
        min_val_trades=cfg.min_val_trades,
        leverage=cfg.leverage,
        entry_delay_bars=cfg.entry_delay_bars,
        pair_half_life_days=cfg.pair_half_life_days,
    )



def _compact_fit_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k != "scaler"}

def _passes_validation(sim: dict[str, Any], stats: dict[str, Any], cfg: EventCandidatePairwiseWalkForwardCfg) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(sim.get("trade_entries", 0) or 0) < int(cfg.min_val_trades):
        reasons.append("val_trades_below_min")
    if float(sim.get("cagr_pct", -999.0) or -999.0) < float(cfg.min_val_cagr_pct):
        reasons.append("val_cagr_below_min")
    if float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0) < float(cfg.min_val_ratio):
        reasons.append("val_ratio_below_min")
    if float(cfg.max_val_strict_mdd_pct) > 0.0 and float(sim.get("strict_mdd_pct", 999.0) or 999.0) > float(cfg.max_val_strict_mdd_pct):
        reasons.append("val_mdd_above_max")
    if float(stats.get("t_stat_like", -999.0) or -999.0) < float(cfg.min_val_t_stat):
        reasons.append("val_t_stat_below_min")
    if float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0) > float(cfg.max_val_p_value):
        reasons.append("val_p_value_above_max")
    if int(cfg.max_val_power_gap) >= 0 and int(stats.get("n_gap_to_power_rule", 10**9) or 10**9) > int(cfg.max_val_power_gap):
        reasons.append("val_power_gap_above_max")
    return not reasons, reasons



def _side_trade_stats(executed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for side in ("LONG", "SHORT"):
        vals = [float(t.get("trade_ret_pct", 0.0) or 0.0) for t in executed if str(t.get("side")) == side]
        out[side] = {
            "n": float(len(vals)),
            "mean_trade_ret_pct": float(np.mean(vals)) if vals else 0.0,
            "sum_trade_ret_pct": float(np.sum(vals)) if vals else 0.0,
            "win_frac": float(np.mean(np.asarray(vals) > 0.0)) if vals else 0.0,
        }
    return out


def _allowed_sides_from_validation(side_stats: dict[str, dict[str, float]], cfg: EventCandidatePairwiseWalkForwardCfg) -> set[str] | None:
    if int(cfg.side_min_val_trades) <= 0 and float(cfg.side_min_val_mean_ret_pct) <= -998.0:
        return None
    allowed: set[str] = set()
    for side, stats in side_stats.items():
        if int(stats.get("n", 0.0)) < int(cfg.side_min_val_trades):
            continue
        if float(stats.get("mean_trade_ret_pct", 0.0)) < float(cfg.side_min_val_mean_ret_pct):
            continue
        allowed.add(side)
    return allowed


def _side_scales_from_validation(side_stats: dict[str, dict[str, float]], cfg: EventCandidatePairwiseWalkForwardCfg) -> dict[str, float] | None:
    denom = float(cfg.side_scale_val_mean_ret_pct)
    if denom <= 0.0:
        return None
    scales: dict[str, float] = {}
    for side, stats in side_stats.items():
        if int(cfg.side_min_val_trades) > 0 and int(stats.get("n", 0.0)) < int(cfg.side_min_val_trades):
            scales[side] = 0.0
            continue
        mean_ret = float(stats.get("mean_trade_ret_pct", 0.0))
        scales[side] = min(1.0, max(0.0, mean_ret / denom))
    return scales

def _select_on_validation(
    fit_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    cfg: EventCandidatePairwiseWalkForwardCfg,
    tmp: Path,
    fold_id: int,
) -> dict[str, Any]:
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    pw_cfg = _pairwise_cfg(cfg, "", "")
    fit_scores, val_scores, names, fit_meta = _fit_score(fit_rows, val_rows, pw_cfg)
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
            side_stats = _side_trade_stats(bt.get("executed", []))
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
                    "val_side_trade_stats": side_stats,
                    "validation_passed": passed,
                    "validation_reject_reasons": reasons,
                    "score": score,
                }
            )
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0] if candidates else {"validation_passed": False, "validation_reject_reasons": ["no_candidates"], "q": 1.0, "full_margin": 0.0, "score": -9999.0}
    return {"selected": selected, "top5": candidates[:5], "fit_meta": _compact_fit_meta(fit_meta), "features": {"numeric": len(names[0]), "categorical": len(names[1]), "expanded": fit_meta["features"]}}


def _trade_test_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    selected: dict[str, Any],
    cfg: EventCandidatePairwiseWalkForwardCfg,
    fold_dir: Path,
    fold_id: int,
) -> dict[str, Any]:
    pred_path = fold_dir / f"fold{fold_id:03d}_test_predictions.jsonl"
    pw_cfg = _pairwise_cfg(cfg, "", "")
    train_scores, test_scores, _names, fit_meta = _fit_score(train_rows, test_rows, pw_cfg)
    train_best_scores = np.asarray([x["score"] for x in _best_by_signal(train_rows, train_scores)], dtype=float)
    threshold = float(np.quantile(train_best_scores, float(selected["q"]))) if len(train_best_scores) else 999.0
    test_best = _best_by_signal(test_rows, test_scores)
    side_stats = selected.get("val_side_trade_stats", {})
    allowed_sides = _allowed_sides_from_validation(side_stats, cfg)
    side_scales = _side_scales_from_validation(side_stats, cfg)
    ps = _write_policy(test_best, str(pred_path), threshold, float(selected["full_margin"]), allowed_sides=allowed_sides, side_scale_by_side=side_scales)
    bt = run_overlay(
        OnlineRiskOverlayConfig(
            predictions_jsonl=str(pred_path),
            market_csv=cfg.market_csv,
            output=str(fold_dir / f"fold{fold_id:03d}_test_backtest.json"),
            leverage=cfg.leverage,
            entry_delay_bars=cfg.entry_delay_bars,
        )
    )
    return {"prediction_path": str(pred_path), "prediction_summary": ps, "test_backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}, "test_threshold": threshold, "side_allowlist": sorted(allowed_sides) if allowed_sides is not None else None, "side_scale_by_side": side_scales, "fit_meta": _compact_fit_meta(fit_meta)}


def run(cfg: EventCandidatePairwiseWalkForwardCfg) -> dict[str, Any]:
    rows = sorted(_load(cfg.input_jsonl), key=lambda r: (_date(r), int(r.get("signal_pos", -1) or -1), str(r.get("side", ""))))
    if not rows:
        raise ValueError(f"no rows loaded from {cfg.input_jsonl}")
    folds = make_folds(_date(rows[0]), _date(rows[-1]), cfg)
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, Any]] = []
    pred_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rllm_pairwise_wf_val_") as tmp_raw:
        tmp = Path(tmp_raw)
        for fold in folds:
            fit = [r for r in rows if _in_half_open(r, fold.fit_start, fold.fit_end)]
            val = [r for r in rows if _in_half_open(r, fold.val_start, fold.val_end)]
            test = [r for r in rows if _in_half_open(r, fold.test_start, fold.test_end)]
            fold_dir = work_dir / f"fold{fold.fold_id:03d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            row_counts = {"fit": len(fit), "val": len(val), "test": len(test), "fit_signals": _count_signals(fit), "val_signals": _count_signals(val), "test_signals": _count_signals(test)}
            skip_reasons: list[str] = []
            if row_counts["fit_signals"] < cfg.min_fit_signals:
                skip_reasons.append("fit_signals_below_min")
            if row_counts["test_signals"] < cfg.min_test_signals:
                skip_reasons.append("test_signals_below_min")
            if not fit or not val or not test:
                skip_reasons.append("empty_required_split")
            if skip_reasons:
                pred = fold_dir / f"fold{fold.fold_id:03d}_test_predictions.jsonl"
                ps = _no_trade_predictions(test, str(pred), reason=",".join(skip_reasons))
                pred_paths.append(str(pred))
                fold_reports.append({"fold": asdict(fold), "rows": row_counts, "status": "SKIPPED", "skip_reasons": skip_reasons, "test_prediction_summary": ps})
                continue
            selection = _select_on_validation(fit, val, cfg, tmp, fold.fold_id)
            selected = selection["selected"]
            if not bool(selected.get("validation_passed", False)):
                pred = fold_dir / f"fold{fold.fold_id:03d}_test_predictions.jsonl"
                ps = _no_trade_predictions(test, str(pred), reason="validation_gate_failed")
                pred_paths.append(str(pred))
                fold_reports.append({"fold": asdict(fold), "rows": row_counts, "status": "ABSTAIN", "selection": selection, "test_prediction_summary": ps})
                continue
            test_result = _trade_test_fold(fit + val, test, selected, cfg, fold_dir, fold.fold_id)
            pred_paths.append(test_result["prediction_path"])
            fold_reports.append({"fold": asdict(fold), "rows": row_counts, "status": "TRADED", "selection": selection, "test": test_result})
    aggregate: dict[str, Any] | None = None
    if pred_paths:
        agg_path = work_dir / "aggregate_predictions.files.txt"
        agg_path.write_text("\n".join(pred_paths) + "\n")
        bt = run_overlay(
            OnlineRiskOverlayConfig(
                predictions_jsonl=",".join(pred_paths),
                market_csv=cfg.market_csv,
                output=str(work_dir / "aggregate_backtest.json"),
                leverage=cfg.leverage,
                entry_delay_bars=cfg.entry_delay_bars,
            )
        )
        aggregate = {"prediction_files": pred_paths, "backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]}}
    report = {
        "config": asdict(cfg),
        "rows": {"input": len(rows), "signals": _count_signals(rows), "first_date": _date(rows[0]), "last_date": _date(rows[-1])},
        "folds": fold_reports,
        "aggregate": aggregate,
        "leakage_guard": {
            "folds_are_time_ordered": True,
            "validation_selects_policy_before_test": True,
            "test_rows_never_used_for_fit_threshold_or_gate": True,
            "test_model_refit_uses_fit_plus_validation_only": True,
            "aggregate_backtest_reads_only_fold_test_predictions": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling validation-gated pairwise event candidate walk-forward")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=EventCandidatePairwiseWalkForwardCfg.work_dir)
    p.add_argument("--start-date", default=EventCandidatePairwiseWalkForwardCfg.start_date)
    p.add_argument("--end-date", default=EventCandidatePairwiseWalkForwardCfg.end_date)
    p.add_argument("--fit-months", type=int, default=EventCandidatePairwiseWalkForwardCfg.fit_months)
    p.add_argument("--val-months", type=int, default=EventCandidatePairwiseWalkForwardCfg.val_months)
    p.add_argument("--test-months", type=int, default=EventCandidatePairwiseWalkForwardCfg.test_months)
    p.add_argument("--step-months", type=int, default=EventCandidatePairwiseWalkForwardCfg.step_months)
    p.add_argument("--max-pairs-per-signal", type=int, default=EventCandidatePairwiseWalkForwardCfg.max_pairs_per_signal)
    p.add_argument("--min-utility-gap", type=float, default=EventCandidatePairwiseWalkForwardCfg.min_utility_gap)
    p.add_argument("--lr", type=float, default=EventCandidatePairwiseWalkForwardCfg.lr)
    p.add_argument("--l2", type=float, default=EventCandidatePairwiseWalkForwardCfg.l2)
    p.add_argument("--epochs", type=int, default=EventCandidatePairwiseWalkForwardCfg.epochs)
    p.add_argument("--quantiles", default=EventCandidatePairwiseWalkForwardCfg.quantiles)
    p.add_argument("--full-margins", default=EventCandidatePairwiseWalkForwardCfg.full_margins)
    p.add_argument("--min-fit-signals", type=int, default=EventCandidatePairwiseWalkForwardCfg.min_fit_signals)
    p.add_argument("--min-val-trades", type=int, default=EventCandidatePairwiseWalkForwardCfg.min_val_trades)
    p.add_argument("--min-test-signals", type=int, default=EventCandidatePairwiseWalkForwardCfg.min_test_signals)
    p.add_argument("--min-val-cagr-pct", type=float, default=EventCandidatePairwiseWalkForwardCfg.min_val_cagr_pct)
    p.add_argument("--min-val-ratio", type=float, default=EventCandidatePairwiseWalkForwardCfg.min_val_ratio)
    p.add_argument("--max-val-strict-mdd-pct", type=float, default=EventCandidatePairwiseWalkForwardCfg.max_val_strict_mdd_pct)
    p.add_argument("--min-val-t-stat", type=float, default=EventCandidatePairwiseWalkForwardCfg.min_val_t_stat)
    p.add_argument("--max-val-p-value", type=float, default=EventCandidatePairwiseWalkForwardCfg.max_val_p_value)
    p.add_argument("--max-val-power-gap", type=int, default=EventCandidatePairwiseWalkForwardCfg.max_val_power_gap)
    p.add_argument("--leverage", type=float, default=EventCandidatePairwiseWalkForwardCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=EventCandidatePairwiseWalkForwardCfg.entry_delay_bars)
    p.add_argument("--pair-half-life-days", type=float, default=EventCandidatePairwiseWalkForwardCfg.pair_half_life_days)
    p.add_argument("--side-min-val-trades", type=int, default=EventCandidatePairwiseWalkForwardCfg.side_min_val_trades)
    p.add_argument("--side-min-val-mean-ret-pct", type=float, default=EventCandidatePairwiseWalkForwardCfg.side_min_val_mean_ret_pct)
    p.add_argument("--side-scale-val-mean-ret-pct", type=float, default=EventCandidatePairwiseWalkForwardCfg.side_scale_val_mean_ret_pct)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidatePairwiseWalkForwardCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
