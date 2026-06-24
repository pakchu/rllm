"""No-leak per-signal pairwise ranker for event candidate rows."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.event_candidate_ridge_ranker import _best_by_signal, _date, _feature_names, _load, _standardize, _write_policy, _xy
from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay


@dataclass(frozen=True)
class EventCandidatePairwiseRankerCfg:
    train_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    work_dir: str = "results/event_candidate_pairwise_ranker"
    validation_start: str = "2024-01-01"
    validation_end: str = "2025-12-31 23:59:59"
    max_pairs_per_signal: int = 8
    min_utility_gap: float = 0.001
    lr: float = 0.2
    l2: float = 10.0
    epochs: int = 200
    quantiles: str = "0.70,0.80,0.85,0.90,0.95"
    full_margins: str = "0,0.25,0.5"
    min_val_trades: int = 50
    leverage: float = 1.0
    entry_delay_bars: int = 1
    pair_half_life_days: float = 0.0
    ranker_drop_prefixes: str = ""


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[int]]:
    out: dict[tuple[str, int], list[int]] = {}
    for i, row in enumerate(rows):
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1))
        out.setdefault(key, []).append(i)
    return out


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("rank_utility", reward.get("net_return_pct", 0.0)) or 0.0)


def build_pairs(rows: list[dict[str, Any]], *, max_pairs_per_signal: int, min_utility_gap: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for idxs in _groups(rows).values():
        ordered = sorted(idxs, key=lambda i: _utility(rows[i]), reverse=True)
        if len(ordered) < 2:
            continue
        best = ordered[0]
        best_u = _utility(rows[best])
        count = 0
        for other in ordered[1:]:
            if best_u - _utility(rows[other]) < float(min_utility_gap):
                continue
            pairs.append((best, other))
            count += 1
            if count >= int(max_pairs_per_signal):
                break
    return pairs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _fit_pairwise(x: np.ndarray, pairs: list[tuple[int, int]], *, lr: float, l2: float, epochs: int, pair_weights: np.ndarray | None = None) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    if not pairs or x.shape[1] == 0:
        return w
    win = np.asarray([a for a, _ in pairs], dtype=np.int64)
    lose = np.asarray([b for _, b in pairs], dtype=np.int64)
    diff = x[win] - x[lose]
    if pair_weights is None:
        weights = np.ones(len(pairs), dtype=np.float64)
    else:
        weights = np.asarray(pair_weights, dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        weights = weights / max(1e-12, float(np.mean(weights)))
    n = float(np.sum(weights))
    for _ in range(int(epochs)):
        p = _sigmoid(diff @ w)
        grad = -(diff.T @ (weights * (1.0 - p))) / n + float(l2) * w / n
        w -= float(lr) * grad
    return w



def _pair_time_weights(rows: list[dict[str, Any]], pairs: list[tuple[int, int]], half_life_days: float) -> np.ndarray | None:
    if float(half_life_days) <= 0.0 or not pairs:
        return None
    dates = [np.datetime64(str(r.get("date"))) for r in rows]
    max_date = max(dates)
    half_life_seconds = float(half_life_days) * 86400.0
    weights = []
    for a, _ in pairs:
        age_seconds = float((max_date - dates[a]) / np.timedelta64(1, "s"))
        weights.append(0.5 ** (max(0.0, age_seconds) / half_life_seconds))
    return np.asarray(weights, dtype=np.float64)

def _fit_score(
    fit_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    cfg: EventCandidatePairwiseRankerCfg,
    names: tuple[list[str], list[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[list[str], list[str]], dict[str, Any]]:
    if names is None:
        names = _feature_names(fit_rows, tuple(x.strip() for x in cfg.ranker_drop_prefixes.split(",") if x.strip()))
    num, cat = names
    x_fit, _ = _xy(fit_rows, num, cat)
    x_score, _ = _xy(score_rows, num, cat)
    x_fit_z, x_score_z, scaler = _standardize(x_fit, x_score)
    pairs = build_pairs(fit_rows, max_pairs_per_signal=cfg.max_pairs_per_signal, min_utility_gap=cfg.min_utility_gap)
    pair_weights = _pair_time_weights(fit_rows, pairs, cfg.pair_half_life_days)
    w = _fit_pairwise(x_fit_z, pairs, lr=cfg.lr, l2=cfg.l2, epochs=cfg.epochs, pair_weights=pair_weights)
    meta = {"pairs": len(pairs), "features": x_fit_z.shape[1], "scaler": scaler, "weight_l2": float(np.sqrt(np.sum(w * w))), "pair_half_life_days": float(cfg.pair_half_life_days), "pair_weight_min": float(np.min(pair_weights)) if pair_weights is not None and len(pair_weights) else 1.0, "pair_weight_max": float(np.max(pair_weights)) if pair_weights is not None and len(pair_weights) else 1.0}
    return x_fit_z @ w, x_score_z @ w, names, meta


def run(cfg: EventCandidatePairwiseRankerCfg) -> dict[str, Any]:
    train_all = _load(cfg.train_jsonl)
    eval_rows = _load(cfg.eval_jsonl)
    fit = [r for r in train_all if _date(r) < cfg.validation_start]
    val = [r for r in train_all if cfg.validation_start <= _date(r) <= cfg.validation_end]
    qs = [float(x) for x in cfg.quantiles.split(",") if x.strip()]
    margins = [float(x) for x in cfg.full_margins.split(",") if x.strip()]
    fit_scores, val_scores, names, fit_meta = _fit_score(fit, val, cfg)
    fit_best_scores = np.asarray([x["score"] for x in _best_by_signal(fit, fit_scores)], dtype=float)
    val_best = _best_by_signal(val, val_scores)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rllm_event_pairwise_val_") as tmp_raw:
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
    train_scores, eval_scores, names2, final_meta = _fit_score(train_all, eval_rows, cfg, names)
    train_best_scores = np.asarray([x["score"] for x in _best_by_signal(train_all, train_scores)], dtype=float)
    eval_thr = float(np.quantile(train_best_scores, float(selected["q"]))) if len(train_best_scores) else 999.0
    eval_best = _best_by_signal(eval_rows, eval_scores)
    eval_pred = str(Path(cfg.work_dir) / "selected_eval_predictions.jsonl")
    eval_ps = _write_policy(eval_best, eval_pred, eval_thr, float(selected["full_margin"]))
    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir) / "selected_eval_backtest.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars))
    report = {
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "train_all": len(train_all), "eval": len(eval_rows)},
        "features": {"numeric": len(names[0]), "categorical": len(names[1]), "expanded": fit_meta["features"]},
        "fit_meta": fit_meta,
        "final_meta": final_meta,
        "selection_rule": "fit pairwise winner>loser within each signal; select q/full_margin on validation only; eval refits on all train",
        "top10_val": candidates[:10],
        "selected": selected,
        "eval_threshold": eval_thr,
        "eval_prediction_summary": eval_ps,
        "eval_backtest": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": {"pairs_use_train_reward_only": True, "validation_only_selects_policy": True, "eval_not_used_for_fit_or_threshold_selection": True, "eval_threshold_uses_train_score_quantile": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise event candidate ranker")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=EventCandidatePairwiseRankerCfg.work_dir)
    p.add_argument("--validation-start", default=EventCandidatePairwiseRankerCfg.validation_start)
    p.add_argument("--validation-end", default=EventCandidatePairwiseRankerCfg.validation_end)
    p.add_argument("--max-pairs-per-signal", type=int, default=EventCandidatePairwiseRankerCfg.max_pairs_per_signal)
    p.add_argument("--min-utility-gap", type=float, default=EventCandidatePairwiseRankerCfg.min_utility_gap)
    p.add_argument("--lr", type=float, default=EventCandidatePairwiseRankerCfg.lr)
    p.add_argument("--l2", type=float, default=EventCandidatePairwiseRankerCfg.l2)
    p.add_argument("--epochs", type=int, default=EventCandidatePairwiseRankerCfg.epochs)
    p.add_argument("--quantiles", default=EventCandidatePairwiseRankerCfg.quantiles)
    p.add_argument("--full-margins", default=EventCandidatePairwiseRankerCfg.full_margins)
    p.add_argument("--min-val-trades", type=int, default=EventCandidatePairwiseRankerCfg.min_val_trades)
    p.add_argument("--leverage", type=float, default=EventCandidatePairwiseRankerCfg.leverage)
    p.add_argument("--entry-delay-bars", type=int, default=EventCandidatePairwiseRankerCfg.entry_delay_bars)
    p.add_argument("--pair-half-life-days", type=float, default=EventCandidatePairwiseRankerCfg.pair_half_life_days)
    p.add_argument("--ranker-drop-prefixes", default=EventCandidatePairwiseRankerCfg.ranker_drop_prefixes)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidatePairwiseRankerCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
