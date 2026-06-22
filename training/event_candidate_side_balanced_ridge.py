"""Side-balanced no-leak ridge ranker for event candidate policies.

This variant fits independent LONG and SHORT ridge models and applies
side-specific train-score quantile thresholds. It is meant to test whether the
single-model event ranker failed because a train-period LONG prior suppressed
SHORT candidates rather than because the candidate surface is unusable.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_ridge_ranker import _date, _feature_names, _load, _predict, _ridge, _standardize, _xy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

SIDES = ("LONG", "SHORT")


@dataclass(frozen=True)
class SideBalancedRidgeCfg:
    train_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_side_balanced_ridge"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    ridge_alpha: float = 100.0
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.25,0.5,1.0"
    min_val_trades: int = 80
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _fit_score_by_side(
    fit_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    alpha: float,
    names: tuple[list[str], list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[list[str], list[str]]]:
    if names is None:
        names = _feature_names(fit_rows)
    num, cat = names
    fit_scores = np.full(len(fit_rows), np.nan, dtype=float)
    score_scores = np.full(len(score_rows), np.nan, dtype=float)
    fit_side_scores: dict[str, np.ndarray] = {}
    for side in SIDES:
        fit_idx = [i for i, r in enumerate(fit_rows) if str(r.get("side")) == side]
        score_idx = [i for i, r in enumerate(score_rows) if str(r.get("side")) == side]
        side_fit = [fit_rows[i] for i in fit_idx]
        side_score = [score_rows[i] for i in score_idx]
        if not side_fit:
            fit_side_scores[side] = np.asarray([], dtype=float)
            continue
        xtr, ytr = _xy(side_fit, num, cat)
        xte, _ = _xy(side_score, num, cat) if side_score else (np.zeros((0, xtr.shape[1])), np.zeros(0))
        xtrz, xtez, _ = _standardize(xtr, xte)
        w = _ridge(xtrz, ytr, alpha)
        sfit = _predict(xtrz, w)
        fit_side_scores[side] = np.asarray(sfit, dtype=float)
        for i, sc in zip(fit_idx, sfit):
            fit_scores[i] = float(sc)
        if side_score:
            ste = _predict(xtez, w)
            for i, sc in zip(score_idx, ste):
                score_scores[i] = float(sc)
    return fit_scores, score_scores, fit_side_scores, names


def _best_by_signal_side_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for r, sc in zip(rows, scores):
        if not np.isfinite(sc):
            continue
        side = str(r.get("side"))
        if side not in thresholds:
            continue
        excess = float(sc) - float(thresholds[side])
        pos = int(r.get("signal_pos"))
        cur = best.get(pos)
        if cur is None or excess > float(cur["excess"]):
            best[pos] = {"row": r, "score": float(sc), "threshold": float(thresholds[side]), "excess": excess}
    return [best[k] for k in sorted(best)]


def _write_policy(best_rows: list[dict[str, Any]], output: str, full_margin: float, small_scale: float = 0.5) -> dict[str, Any]:
    out = []
    counts = {"TRADE": 0, "NO_TRADE": 0, "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}
    for item in best_rows:
        r = item["row"]
        side = str(r.get("side"))
        excess = float(item["excess"])
        score = float(item["score"])
        threshold = float(item["threshold"])
        hold = int(r.get("candidate", {}).get("hold_bars", 288) or 288)
        if excess >= 0.0 and side in SIDES:
            scale = 1.0 if excess >= float(full_margin) else float(small_scale)
            pred = {"gate": "TRADE", "side": side, "hold_bars": hold, "confidence": "HIGH", "family": "event_candidate_side_balanced_ridge"}
            counts["TRADE"] += 1
            counts[side] += 1
            counts["FULL" if scale >= 1.0 else "SMALL"] += 1
        else:
            scale = 0.0
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "event_candidate_side_balanced_ridge"}
            counts["NO_TRADE"] += 1
        out.append({
            "date": r.get("date"),
            "signal_pos": r.get("signal_pos"),
            "prediction": pred,
            "position_scale": scale,
            "score": score,
            "threshold": threshold,
            "excess": excess,
            "side_candidate": side,
        })
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"rows": len(out), "counts": counts, "full_margin": full_margin, "output": output}


def _thresholds(side_scores: dict[str, np.ndarray], q: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for side in SIDES:
        scores = side_scores.get(side, np.asarray([], dtype=float))
        scores = scores[np.isfinite(scores)]
        out[side] = float(np.quantile(scores, q)) if len(scores) else 999.0
    return out


def run(cfg: SideBalancedRidgeCfg) -> dict[str, Any]:
    train_all = _load(cfg.train_jsonl)
    eval_rows = _load(cfg.eval_jsonl)
    fit = [r for r in train_all if _date(r) < cfg.validation_start]
    val = [r for r in train_all if cfg.validation_start <= _date(r) <= cfg.validation_end]
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]

    _, val_scores, fit_side_scores, names = _fit_score_by_side(fit, val, cfg.ridge_alpha)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rllm_side_balanced_val_") as tmp_raw:
        tmp = Path(tmp_raw)
        for q in qs:
            th = _thresholds(fit_side_scores, q)
            val_best = _best_by_signal_side_threshold(val, val_scores, th)
            for margin in margins:
                pred = tmp / f"val_q{q}_m{margin}.jsonl"
                ps = _write_policy(val_best, str(pred), margin)
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_csv, output=str(tmp / f"val_q{q}_m{margin}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
                sim = bt["sim"]
                stats = bt["trade_stats"]
                score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
                if int(sim.get("trade_entries", 0) or 0) < cfg.min_val_trades:
                    score -= 1000.0
                candidates.append({"q": q, "full_margin": margin, "thresholds": th, "prediction_summary": ps, "val_sim": sim, "val_trade_stats": stats, "score": score})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0]

    _, eval_scores, train_side_scores, _ = _fit_score_by_side(train_all, eval_rows, cfg.ridge_alpha, names)
    eval_thresholds = _thresholds(train_side_scores, float(selected["q"]))
    eval_best = _best_by_signal_side_threshold(eval_rows, eval_scores, eval_thresholds)
    eval_pred = str(Path(cfg.work_dir) / "selected_eval_predictions.jsonl")
    eval_ps = _write_policy(eval_best, eval_pred, float(selected["full_margin"]))
    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir) / "selected_eval_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))

    report = {
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "train_all": len(train_all), "eval": len(eval_rows)},
        "features": {"numeric": len(names[0]), "categorical": len(names[1])},
        "selection_rule": "fit separate LONG/SHORT ridge models; select shared q/full_margin on validation only; eval refits on all train and uses side-specific train quantiles",
        "top10_val": candidates[:10],
        "selected": selected,
        "eval_thresholds": eval_thresholds,
        "eval_prediction_summary": eval_ps,
        "eval_backtest": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": {"validation_only_selects_policy": True, "eval_not_used_for_fit_or_threshold_selection": True, "eval_thresholds_use_train_side_score_quantiles": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-balanced event candidate ridge ranker")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=SideBalancedRidgeCfg.work_dir)
    p.add_argument("--validation-start", default=SideBalancedRidgeCfg.validation_start)
    p.add_argument("--validation-end", default=SideBalancedRidgeCfg.validation_end)
    p.add_argument("--ridge-alpha", type=float, default=SideBalancedRidgeCfg.ridge_alpha)
    p.add_argument("--quantiles", default=SideBalancedRidgeCfg.quantiles)
    p.add_argument("--full-margins", default=SideBalancedRidgeCfg.full_margins)
    p.add_argument("--min-val-trades", type=int, default=SideBalancedRidgeCfg.min_val_trades)
    p.add_argument("--leverage", type=float, default=SideBalancedRidgeCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=SideBalancedRidgeCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(SideBalancedRidgeCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
