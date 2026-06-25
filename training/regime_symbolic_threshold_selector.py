"""No-leak threshold selector for regime-conditioned symbolic policies.

Protocol:
1. Split a historical candidate file into fit-history rows and validation rows.
2. For each target/threshold candidate, run monthly rolling symbolic prediction on
   validation using only rows before each validation month.
3. Select target/threshold on validation backtest only.
4. Re-run the selected fixed config on final eval using all historical rows before
   eval start. Final eval is not used for selection.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.regime_conditioned_symbolic_policy import rolling_predict
from training.symbolic_action_ridge import load_jsonl, write_jsonl


@dataclass(frozen=True)
class SymbolicThresholdSelectorCfg:
    history_jsonl: str
    final_eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/regime_symbolic_threshold_selector"
    validation_start: str = "2025-01-01"
    validation_end: str = "2026-01-01"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    targets: str = "net_return,risk_adjusted,tail_risk,distributional_safety"
    thresholds: str = "-0.0154,-0.0134,-0.0125,-0.0108,-0.0101,-0.0094,-0.0002,0.0,0.0002"
    alpha: float = 10000.0
    min_gap: float = 0.0
    expert_margin: float = 0.0
    min_feature_count: int = 5
    min_train_rows: int = 2000
    leverage: float = 1.0
    min_val_trades: int = 100
    max_val_mdd_pct: float = 20.0
    min_val_cagr_pct: float = 0.0
    max_val_p_value: float = 0.5


def _date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def _split_history(history_jsonl: str, validation_start: str, validation_end: str, work_dir: Path) -> tuple[str, str, dict[str, int]]:
    rows = load_jsonl(history_jsonl)
    fit = [r for r in rows if _date(r) < validation_start]
    val = [r for r in rows if validation_start <= _date(r) < validation_end]
    fit_path = work_dir / "selector_fit_history.jsonl"
    val_path = work_dir / "selector_validation.jsonl"
    write_jsonl(fit_path, fit)
    write_jsonl(val_path, val)
    return str(fit_path), str(val_path), {"history": len(rows), "fit": len(fit), "validation": len(val)}


def _parse_csv(raw: str, typ=str) -> list[Any]:
    return [typ(x.strip()) for x in str(raw).split(",") if x.strip()]


def _candidate_score(bt: dict[str, Any], cfg: SymbolicThresholdSelectorCfg) -> tuple[float, list[str]]:
    sim = bt["sim"]
    stats = bt["trade_stats"]
    reasons: list[str] = []
    if int(sim.get("trade_entries", 0) or 0) < int(cfg.min_val_trades):
        reasons.append("val_trades_below_min")
    if float(sim.get("cagr_pct", -999.0) or -999.0) < float(cfg.min_val_cagr_pct):
        reasons.append("val_cagr_below_min")
    if float(sim.get("strict_mdd_pct", 999.0) or 999.0) > float(cfg.max_val_mdd_pct):
        reasons.append("val_mdd_above_max")
    if float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0) > float(cfg.max_val_p_value):
        reasons.append("val_p_value_above_max")
    score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    if reasons:
        score -= 1000.0 + 10.0 * len(reasons)
    return score, reasons


def _run_one(*, history_jsonl: str, eval_jsonl: str, start: str, end: str, target: str, threshold: float, cfg: SymbolicThresholdSelectorCfg, out_dir: Path, tag: str) -> dict[str, Any]:
    pred = out_dir / f"{tag}_predictions.jsonl"
    summ = out_dir / f"{tag}_summary.json"
    bt_path = out_dir / f"{tag}_backtest.json"
    summary = rolling_predict(
        history_jsonl=history_jsonl,
        eval_jsonl=eval_jsonl,
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
    return {"target": target, "threshold": float(threshold), "predictions": str(pred), "summary": summary, "backtest": {"period": bt["period"], "sim": bt["sim"], "trade_stats": bt["trade_stats"]}}


def run(cfg: SymbolicThresholdSelectorCfg) -> dict[str, Any]:
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fit_path, val_path, split_counts = _split_history(cfg.history_jsonl, cfg.validation_start, cfg.validation_end, work)
    candidates: list[dict[str, Any]] = []
    for target in _parse_csv(cfg.targets, str):
        for threshold in _parse_csv(cfg.thresholds, float):
            tag = f"val_{target}_th{str(threshold).replace('-', 'm').replace('.', 'p')}"
            row = _run_one(history_jsonl=fit_path, eval_jsonl=val_path, start=cfg.validation_start, end=cfg.validation_end, target=target, threshold=float(threshold), cfg=cfg, out_dir=work, tag=tag)
            score, reject = _candidate_score(row["backtest"], cfg)
            row["selection_score"] = score
            row["validation_passed"] = not reject
            row["validation_reject_reasons"] = reject
            candidates.append(row)
    candidates.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    selected = candidates[0] if candidates else None
    eval_result = None
    if selected is not None and bool(selected.get("validation_passed", False)):
        eval_result = _run_one(
            history_jsonl=cfg.history_jsonl,
            eval_jsonl=cfg.final_eval_jsonl,
            start=cfg.eval_start,
            end=cfg.eval_end,
            target=str(selected["target"]),
            threshold=float(selected["threshold"]),
            cfg=cfg,
            out_dir=work,
            tag=f"eval_selected_{selected['target']}_th{str(selected['threshold']).replace('-', 'm').replace('.', 'p')}",
        )
    report = {
        "config": asdict(cfg),
        "split_counts": split_counts,
        "top_validation": candidates[:20],
        "selected": selected,
        "eval_result": eval_result,
        "leakage_guard": {
            "validation_rows_split_from_history_before_eval": True,
            "validation_selects_target_threshold_only_before_final_eval": True,
            "final_eval_not_used_for_selection": True,
            "monthly_fits_use_rows_before_month_start_only": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-leak symbolic threshold selector")
    p.add_argument("--history-jsonl", required=True)
    p.add_argument("--final-eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=SymbolicThresholdSelectorCfg.work_dir)
    p.add_argument("--validation-start", default="2025-01-01")
    p.add_argument("--validation-end", default="2026-01-01")
    p.add_argument("--eval-start", default="2026-01-01")
    p.add_argument("--eval-end", default="2026-06-01")
    p.add_argument("--targets", default=SymbolicThresholdSelectorCfg.targets)
    p.add_argument("--thresholds", default=SymbolicThresholdSelectorCfg.thresholds)
    p.add_argument("--alpha", type=float, default=10000.0)
    p.add_argument("--min-gap", type=float, default=0.0)
    p.add_argument("--expert-margin", type=float, default=0.0)
    p.add_argument("--min-feature-count", type=int, default=5)
    p.add_argument("--min-train-rows", type=int, default=2000)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--min-val-trades", type=int, default=100)
    p.add_argument("--max-val-mdd-pct", type=float, default=20.0)
    p.add_argument("--min-val-cagr-pct", type=float, default=0.0)
    p.add_argument("--max-val-p-value", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    report = run(SymbolicThresholdSelectorCfg(**vars(parse_args())))
    sel = report.get("selected") or {}
    ev = report.get("eval_result") or {}
    print(json.dumps({"selected": {"target": sel.get("target"), "threshold": sel.get("threshold"), "validation_passed": sel.get("validation_passed"), "score": sel.get("selection_score"), "val_sim": (sel.get("backtest") or {}).get("sim"), "val_stats": (sel.get("backtest") or {}).get("trade_stats")}, "eval_sim": (ev.get("backtest") or {}).get("sim"), "eval_stats": (ev.get("backtest") or {}).get("trade_stats")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
