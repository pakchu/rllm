"""Expanding no-leak walk-forward IC ranker over event candidates.

For each test year Y:
- fit IC feature weights on rows before Y-1
- select q/full_margin on validation year Y-1
- evaluate on Y only
The test year is never used for feature selection, threshold, or sizing.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_ic_ranker import _best_by_signal, _date, _feature_names, _fit_ic, _load, _score, _write_policy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class ICWalkForwardCfg:
    train_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_ic_walkforward"
    test_years: str = "2022,2023,2024,2025,2026"
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.25,0.5,1.0"
    min_abs_ic: float = 0.02
    min_sign_consistency: float = 0.75
    min_val_trades: int = 30
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _year(row: dict[str, Any]) -> int:
    return int(_date(row)[:4])


def _select_policy(
    fit: list[dict[str, Any]],
    val: list[dict[str, Any]],
    cfg: ICWalkForwardCfg,
    tmp: Path,
) -> dict[str, Any]:
    model = _fit_ic(fit, _feature_names(fit), cfg.min_abs_ic, cfg.min_sign_consistency)
    fit_best_scores = np.asarray([x["score"] for x in _best_by_signal(fit, _score(fit, model))], dtype=float)
    val_best = _best_by_signal(val, _score(val, model))
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    for q in qs:
        thr = float(np.quantile(fit_best_scores, q)) if len(fit_best_scores) else 999.0
        for margin in margins:
            pred = tmp / f"val_q{q}_m{margin}.jsonl"
            ps = _write_policy(val_best, str(pred), thr, margin)
            bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_csv, output=str(tmp / f"val_q{q}_m{margin}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
            sim = bt["sim"]
            score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
            if int(sim.get("trade_entries", 0) or 0) < cfg.min_val_trades:
                score -= 1000.0
            candidates.append({"q": q, "full_margin": margin, "threshold": thr, "prediction_summary": ps, "val_sim": sim, "val_trade_stats": bt["trade_stats"], "score": score})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    return {"model": model, "selected": candidates[0], "top5_val": candidates[:5]}


def run(cfg: ICWalkForwardCfg) -> dict[str, Any]:
    rows = _load(cfg.train_jsonl) + _load(cfg.eval_jsonl)
    years = [int(x) for x in cfg.test_years.split(",") if x.strip()]
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    year_reports: list[dict[str, Any]] = []
    combined_predictions: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rllm_ic_wf_") as tmp_raw:
        tmp_root = Path(tmp_raw)
        for y in years:
            fit = [r for r in rows if _year(r) < y - 1]
            val = [r for r in rows if _year(r) == y - 1]
            test = [r for r in rows if _year(r) == y]
            fit_years = sorted({_year(r) for r in fit})
            if not fit or not val or not test or len(fit_years) < 2:
                year_reports.append({"year": y, "skipped": True, "reason": "requires non-empty fit/val/test and at least two fit years for IC stability", "rows": {"fit": len(fit), "val": len(val), "test": len(test)}, "fit_years": fit_years})
                continue
            selected = _select_policy(fit, val, cfg, tmp_root / str(y))
            model = selected["model"]
            if not model.get("selected"):
                year_reports.append({"year": y, "skipped": True, "reason": "no sign-stable IC features selected", "rows": {"fit": len(fit), "val": len(val), "test": len(test)}, "fit_years": fit_years})
                continue
            sel = selected["selected"]
            test_best = _best_by_signal(test, _score(test, model))
            pred_path = Path(cfg.work_dir) / f"test_{y}_predictions.jsonl"
            ps = _write_policy(test_best, str(pred_path), float(sel["threshold"]), float(sel["full_margin"]))
            bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=cfg.market_csv, output=str(Path(cfg.work_dir) / f"test_{y}_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
            combined_predictions.extend([line for line in pred_path.read_text().splitlines() if line.strip()])
            year_reports.append({
                "year": y,
                "rows": {"fit": len(fit), "val": len(val), "test": len(test)},
                "features": model["selected"],
                "weights": model["weights"].tolist(),
                "selected": sel,
                "test_prediction_summary": ps,
                "test_backtest": {"sim": bt["sim"], "trade_stats": bt["trade_stats"]},
                "top5_val": selected["top5_val"],
            })
    combined_path = Path(cfg.work_dir) / "combined_test_predictions.jsonl"
    combined_path.write_text("\n".join(combined_predictions) + ("\n" if combined_predictions else ""))
    combined_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(combined_path), market_csv=cfg.market_csv, output=str(Path(cfg.work_dir) / "combined_test_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars)) if combined_predictions else {"sim": {}, "trade_stats": {}}
    report = {
        "config": cfg.__dict__,
        "selection_rule": "for each test year Y, fit rows before Y-1, select q/full_margin on Y-1 only, test on Y only",
        "years": year_reports,
        "combined_prediction_path": str(combined_path),
        "combined_backtest": {"sim": combined_bt["sim"], "trade_stats": combined_bt["trade_stats"]},
        "leakage_guard": {"test_year_not_used_for_fit_or_selection": True, "expanding_fit_only_uses_past_rows": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward IC event candidate ranker")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=ICWalkForwardCfg.work_dir)
    p.add_argument("--test-years", default=ICWalkForwardCfg.test_years)
    p.add_argument("--quantiles", default=ICWalkForwardCfg.quantiles)
    p.add_argument("--full-margins", default=ICWalkForwardCfg.full_margins)
    p.add_argument("--min-abs-ic", type=float, default=ICWalkForwardCfg.min_abs_ic)
    p.add_argument("--min-sign-consistency", type=float, default=ICWalkForwardCfg.min_sign_consistency)
    p.add_argument("--min-val-trades", type=int, default=ICWalkForwardCfg.min_val_trades)
    p.add_argument("--leverage", type=float, default=ICWalkForwardCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=ICWalkForwardCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ICWalkForwardCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
