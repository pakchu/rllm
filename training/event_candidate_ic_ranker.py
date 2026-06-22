"""No-leak IC-weighted event candidate ranker.

This ranker deliberately avoids high-capacity fitting. It selects numeric
side-signed features whose yearly Spearman IC is sign-stable on the fit period,
weights them by fit-period median IC, then selects execution threshold on a
validation period only.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_ridge_ranker import _date, _load
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class ICRankerCfg:
    train_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_ic_ranker"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    quantiles: str = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.25,0.5,1.0"
    min_abs_ic: float = 0.02
    min_sign_consistency: float = 0.75
    min_val_trades: int = 80
    leverage: float = 1.0
    entry_delay_bars: int = 1


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 50 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    base = sorted({k for r in rows for k in (r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}).keys()})
    names: list[str] = []
    for k in base:
        names.append(f"signed:{k}")
        names.append(f"raw:{k}")
    return names


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(names)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    for i, r in enumerate(rows):
        sign = 1.0 if str(r.get("side")) == "LONG" else -1.0
        snap = r.get("feature_snapshot", {}) if isinstance(r.get("feature_snapshot"), dict) else {}
        for j, name in enumerate(names):
            kind, key = name.split(":", 1)
            val = float(snap.get(key, 0.0) or 0.0)
            x[i, j] = val * sign if kind == "signed" else val
        y[i] = float(r.get("reward", {}).get("net_return_pct", 0.0))
    return x, y


def _fit_ic(rows: list[dict[str, Any]], names: list[str], min_abs_ic: float, min_sign_consistency: float) -> dict[str, Any]:
    x, y = _matrix(rows, names)
    years = sorted({_date(r)[:4] for r in rows})
    selected: list[str] = []
    weights: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        ics: list[tuple[str, float]] = []
        for year in years:
            idx = np.asarray([_date(r).startswith(year) for r in rows], dtype=bool)
            ic = _spearman(x[idx, j], y[idx])
            if ic is not None:
                ics.append((year, ic))
        if len(ics) < 2:
            continue
        vals = np.asarray([ic for _, ic in ics], dtype=float)
        med = float(np.median(vals))
        if abs(med) < min_abs_ic:
            continue
        cons = float(np.mean(vals * med > 0.0))
        diagnostics.append({"feature": name, "median_ic": med, "sign_consistency": cons, "yearly_ic": ics})
        if cons >= min_sign_consistency:
            selected.append(name)
            weights.append(med)
    if not selected:
        # Conservative fallback: keep the strongest signed features by median IC.
        diagnostics.sort(key=lambda d: abs(float(d["median_ic"])), reverse=True)
        for d in diagnostics[:8]:
            selected.append(str(d["feature"]))
            weights.append(float(d["median_ic"]))
    idx = [names.index(n) for n in selected]
    xs = x[:, idx] if idx else np.zeros((len(rows), 0), dtype=float)
    mu = xs.mean(axis=0) if xs.size else np.zeros(0)
    sd = xs.std(axis=0) if xs.size else np.ones(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    w = np.asarray(weights, dtype=float)
    norm = np.sum(np.abs(w))
    if norm > 0:
        w = w / norm
    return {"selected": selected, "weights": w, "mu": mu, "sd": sd, "diagnostics": sorted(diagnostics, key=lambda d: abs(float(d["median_ic"])), reverse=True)}


def _score(rows: list[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    names = list(model["selected"])
    if not names:
        return np.zeros(len(rows), dtype=float)
    x, _ = _matrix(rows, names)
    return ((x - model["mu"]) / model["sd"]) @ model["weights"]


def _best_by_signal(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for r, sc in zip(rows, scores):
        pos = int(r.get("signal_pos"))
        cur = best.get(pos)
        if cur is None or float(sc) > float(cur["score"]):
            best[pos] = {"row": r, "score": float(sc)}
    return [best[k] for k in sorted(best)]


def _write_policy(best_rows: list[dict[str, Any]], output: str, threshold: float, full_margin: float, small_scale: float = 0.5) -> dict[str, Any]:
    out = []
    counts = {"TRADE": 0, "NO_TRADE": 0, "LONG": 0, "SHORT": 0, "FULL": 0, "SMALL": 0}
    for item in best_rows:
        r = item["row"]
        score = float(item["score"])
        side = str(r.get("side"))
        hold = int(r.get("candidate", {}).get("hold_bars", 288) or 288)
        if score >= threshold and side in {"LONG", "SHORT"}:
            scale = 1.0 if score >= threshold + float(full_margin) else float(small_scale)
            pred = {"gate": "TRADE", "side": side, "hold_bars": hold, "confidence": "HIGH", "family": "event_candidate_ic_ranker"}
            counts["TRADE"] += 1
            counts[side] += 1
            counts["FULL" if scale >= 1.0 else "SMALL"] += 1
        else:
            scale = 0.0
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "event_candidate_ic_ranker"}
            counts["NO_TRADE"] += 1
        out.append({"date": r.get("date"), "signal_pos": r.get("signal_pos"), "prediction": pred, "position_scale": scale, "score": score, "side_candidate": side})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"rows": len(out), "counts": counts, "threshold": threshold, "full_margin": full_margin, "output": output}


def run(cfg: ICRankerCfg) -> dict[str, Any]:
    train_all = _load(cfg.train_jsonl)
    eval_rows = _load(cfg.eval_jsonl)
    fit = [r for r in train_all if _date(r) < cfg.validation_start]
    val = [r for r in train_all if cfg.validation_start <= _date(r) <= cfg.validation_end]
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    names = _feature_names(fit)
    model = _fit_ic(fit, names, cfg.min_abs_ic, cfg.min_sign_consistency)
    fit_best_scores = np.asarray([x["score"] for x in _best_by_signal(fit, _score(fit, model))], dtype=float)
    val_best = _best_by_signal(val, _score(val, model))
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rllm_ic_ranker_val_") as tmp_raw:
        tmp = Path(tmp_raw)
        for q in qs:
            thr = float(np.quantile(fit_best_scores, q)) if len(fit_best_scores) else 999.0
            for margin in margins:
                pred = tmp / f"val_q{q}_m{margin}.jsonl"
                ps = _write_policy(val_best, str(pred), thr, margin)
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_csv, output=str(tmp / f"val_q{q}_m{margin}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
                sim = bt["sim"]
                stats = bt["trade_stats"]
                score = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
                if int(sim.get("trade_entries", 0) or 0) < cfg.min_val_trades:
                    score -= 1000.0
                candidates.append({"q": q, "full_margin": margin, "threshold": thr, "prediction_summary": ps, "val_sim": sim, "val_trade_stats": stats, "score": score})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = candidates[0]

    final_model = _fit_ic(train_all, _feature_names(train_all), cfg.min_abs_ic, cfg.min_sign_consistency)
    train_best_scores = np.asarray([x["score"] for x in _best_by_signal(train_all, _score(train_all, final_model))], dtype=float)
    eval_thr = float(np.quantile(train_best_scores, float(selected["q"]))) if len(train_best_scores) else 999.0
    eval_best = _best_by_signal(eval_rows, _score(eval_rows, final_model))
    eval_pred = str(Path(cfg.work_dir) / "selected_eval_predictions.jsonl")
    eval_ps = _write_policy(eval_best, eval_pred, eval_thr, float(selected["full_margin"]))
    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir) / "selected_eval_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))

    report = {
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "train_all": len(train_all), "eval": len(eval_rows)},
        "selection_rule": "fit-period sign-stable IC weighted numeric features; select q/full_margin on validation only; eval refits IC weights on all train",
        "fit_model": {"selected": model["selected"], "weights": model["weights"].tolist(), "top_diagnostics": model["diagnostics"][:20]},
        "final_model": {"selected": final_model["selected"], "weights": final_model["weights"].tolist(), "top_diagnostics": final_model["diagnostics"][:20]},
        "top10_val": candidates[:10],
        "selected": selected,
        "eval_threshold": eval_thr,
        "eval_prediction_summary": eval_ps,
        "eval_backtest": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": {"validation_only_selects_policy": True, "eval_not_used_for_fit_or_threshold_selection": True, "eval_threshold_uses_train_score_quantile": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IC-weighted event candidate ranker")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=ICRankerCfg.work_dir)
    p.add_argument("--validation-start", default=ICRankerCfg.validation_start)
    p.add_argument("--validation-end", default=ICRankerCfg.validation_end)
    p.add_argument("--quantiles", default=ICRankerCfg.quantiles)
    p.add_argument("--full-margins", default=ICRankerCfg.full_margins)
    p.add_argument("--min-abs-ic", type=float, default=ICRankerCfg.min_abs_ic)
    p.add_argument("--min-sign-consistency", type=float, default=ICRankerCfg.min_sign_consistency)
    p.add_argument("--min-val-trades", type=int, default=ICRankerCfg.min_val_trades)
    p.add_argument("--leverage", type=float, default=ICRankerCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=ICRankerCfg.entry_delay_bars)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ICRankerCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
