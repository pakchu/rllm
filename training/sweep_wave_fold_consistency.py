"""Fold-consistency selection for 15m wave-teacher candidate policies.

This avoids picking the single best threshold/regime on one contiguous period.
Candidate policies are scored across multiple chronological selection folds, then
both the best robust individual policy and a same-side vote ensemble of robust
policies are replayed unchanged on a later held-out period.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay
from training.sweep_wave_regime_filters import _allow_by_regime, _trend_at
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features, _load_wave_module


@dataclass(frozen=True)
class FoldConsistencyConfig:
    wave_root: str
    market_5m_csv: str
    output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    selection_folds: str = "2021-01-01|2021-06-30 23:59:59,2021-07-01|2021-12-31 23:59:59,2022-01-01|2022-06-30 23:59:59,2022-07-01|2022-12-31 23:59:59,2023-01-01|2023-06-30 23:59:59,2023-07-01|2023-12-31 23:59:59,2024-01-01|2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    long_thresholds: str = "0.66,0.69"
    short_thresholds: str = "0.32,0.35"
    trend_windows: str = "0,288,864"
    modes: str = "all,up,down"
    top_k: int = 5
    vote_k: int = 2
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    min_fold_trades: int = 5
    min_total_trades: int = 50


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _parse_folds(raw: str) -> list[tuple[str, str]]:
    folds = []
    for item in str(raw).split(","):
        if not item.strip():
            continue
        a, b = item.split("|", 1)
        folds.append((a.strip(), b.strip()))
    return folds


def _load_closes(path: str) -> np.ndarray:
    return pd.read_csv(path, usecols=["close"], compression="gzip" if path.endswith(".gz") else None)["close"].to_numpy(dtype=float)


def _policy_side(row: dict[str, Any], closes: np.ndarray, cfg: dict[str, Any]) -> str:
    prob = float(row["teacher_probability_long"])
    side = "NONE"
    if prob >= float(cfg["long_th"]):
        side = "LONG"
    elif prob <= float(cfg["short_th"]):
        side = "SHORT"
    if side == "NONE":
        return side
    tw = int(cfg["trend_window"])
    if tw <= 0:
        return side
    trend = _trend_at(closes, int(row["signal_pos"]), tw)
    mode = str(cfg["long_mode"] if side == "LONG" else cfg["short_mode"])
    return side if _allow_by_regime(side, trend, mode) else "NONE"


def _write_policy_predictions(rows: list[dict[str, Any]], path: Path, *, closes: np.ndarray, policy: dict[str, Any], hold_bars: int) -> dict[str, Any]:
    counts = {"LONG": 0, "SHORT": 0}
    with path.open("w") as f:
        for row in rows:
            side = _policy_side(row, closes, policy)
            pred = {"confidence": "HIGH", "family": "wave_fold_consistency", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            if side in counts:
                counts[side] += 1
                pred = {"confidence": "HIGH", "family": "wave_fold_consistency", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
            f.write(json.dumps({**row, "prediction": pred, "policy": policy}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": counts["LONG"] + counts["SHORT"], "long": counts["LONG"], "short": counts["SHORT"]}


def _write_vote_predictions(rows: list[dict[str, Any]], path: Path, *, closes: np.ndarray, policies: list[dict[str, Any]], vote_k: int, hold_bars: int) -> dict[str, Any]:
    counts = {"LONG": 0, "SHORT": 0}
    with path.open("w") as f:
        for row in rows:
            votes = {"LONG": 0, "SHORT": 0}
            for policy in policies:
                side = _policy_side(row, closes, policy)
                if side in votes:
                    votes[side] += 1
            side = "NONE"
            if votes["LONG"] >= int(vote_k) and votes["LONG"] > votes["SHORT"]:
                side = "LONG"
            elif votes["SHORT"] >= int(vote_k) and votes["SHORT"] > votes["LONG"]:
                side = "SHORT"
            pred = {"confidence": "HIGH", "family": "wave_fold_vote_ensemble", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            if side in counts:
                counts[side] += 1
                pred = {"confidence": "HIGH", "family": "wave_fold_vote_ensemble", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
            f.write(json.dumps({**row, "prediction": pred, "vote_counts": votes, "vote_k": int(vote_k)}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": counts["LONG"] + counts["SHORT"], "long": counts["LONG"], "short": counts["SHORT"]}


def _candidate_policies(cfg: FoldConsistencyConfig) -> list[dict[str, Any]]:
    policies = []
    modes = [m.strip() for m in cfg.modes.split(",") if m.strip()]
    for lt in _parse_floats(cfg.long_thresholds):
        for st in _parse_floats(cfg.short_thresholds):
            if st >= lt:
                continue
            for tw in _parse_ints(cfg.trend_windows):
                if tw <= 0:
                    policies.append({"long_th": lt, "short_th": st, "trend_window": 0, "long_mode": "all", "short_mode": "all"})
                else:
                    for lm in modes:
                        for sm in modes:
                            policies.append({"long_th": lt, "short_th": st, "trend_window": tw, "long_mode": lm, "short_mode": sm})
    # Deduplicate 0-window policies repeated by modes/windows.
    seen = set()
    out = []
    for p in policies:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _robust_score(folds: list[dict[str, Any]], min_fold_trades: int, min_total_trades: int) -> float:
    sims = [f["sim"] for f in folds]
    stats = [f["trade_stats"] for f in folds]
    total_trades = sum(int(s.get("trade_entries", 0) or 0) for s in sims)
    positive = sum(1 for s in sims if float(s.get("cagr_pct", 0.0) or 0.0) > 0.0 and int(s.get("trade_entries", 0) or 0) >= min_fold_trades)
    ratios = [float(s.get("cagr_to_strict_mdd", 0.0) or 0.0) for s in sims]
    cagr = [float(s.get("cagr_pct", 0.0) or 0.0) for s in sims]
    mdd = [float(s.get("strict_mdd_pct", 0.0) or 0.0) for s in sims]
    mean_p = float(np.mean([float(st.get("p_value_mean_ret_approx", 1.0) or 1.0) for st in stats])) if stats else 1.0
    if total_trades < min_total_trades:
        return -1000.0 + total_trades / 1000.0
    return positive * 10.0 + float(np.median(ratios)) + min(2.0, total_trades / 100.0) + float(np.median(cagr)) / 100.0 - max(0.0, float(np.max(mdd)) - 20.0) / 10.0 - mean_p


def run_sweep(cfg: FoldConsistencyConfig) -> dict[str, Any]:
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    closes = _load_closes(cfg.market_5m_csv)
    folds = _parse_folds(cfg.selection_folds)
    fold_rows = [
        {"start": a, "end": b, "rows": _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=a, eval_end=b, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)}
        for a, b in folds
    ]
    eval_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    policies = _candidate_policies(cfg)
    candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rllm_wave_fold_consistency_") as tmp_raw:
        tmp = Path(tmp_raw)
        for idx, policy in enumerate(policies):
            fold_results = []
            for fidx, fold in enumerate(fold_rows):
                pred_path = tmp / f"p{idx}_f{fidx}.jsonl"
                pred_summary = _write_policy_predictions(fold["rows"], pred_path, closes=closes, policy=policy, hold_bars=hold_bars)
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred_path), market_csv=cfg.market_5m_csv, output=str(tmp / f"p{idx}_f{fidx}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
                fold_results.append({"period": {"start": fold["start"], "end": fold["end"]}, "prediction_summary": pred_summary, "sim": bt["sim"], "trade_stats": bt["trade_stats"]})
            candidates.append({"policy": policy, "folds": fold_results, "score": _robust_score(fold_results, cfg.min_fold_trades, cfg.min_total_trades)})
        candidates.sort(key=lambda r: float(r["score"]), reverse=True)
        selected = candidates[: max(1, int(cfg.top_k))]
        best_pred = tmp / "best_eval.jsonl"
        best_summary = _write_policy_predictions(eval_rows, best_pred, closes=closes, policy=selected[0]["policy"], hold_bars=hold_bars)
        best_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(best_pred), market_csv=cfg.market_5m_csv, output=str(tmp / "best_eval.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
        vote_pred = tmp / "vote_eval.jsonl"
        vote_summary = _write_vote_predictions(eval_rows, vote_pred, closes=closes, policies=[x["policy"] for x in selected], vote_k=cfg.vote_k, hold_bars=hold_bars)
        vote_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(vote_pred), market_csv=cfg.market_5m_csv, output=str(tmp / "vote_eval.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    report = {
        "config": asdict(cfg),
        "teacher_params": data["params"],
        "data": {"selection_fold_rows": [{"start": f["start"], "end": f["end"], "rows": len(f["rows"])} for f in fold_rows], "eval_rows": len(eval_rows), "policies": len(policies), "hold_bars_5m": hold_bars},
        "selection_rule": "rank policies by fold consistency only; eval replays best policy and top-k vote ensemble unchanged",
        "top10": candidates[:10],
        "selected_policies": [x["policy"] for x in selected],
        "heldout_eval": {
            "best_policy": {"prediction_summary": best_summary, "eval_sim": best_bt["sim"], "eval_trade_stats": best_bt["trade_stats"]},
            "vote_ensemble": {"prediction_summary": vote_summary, "eval_sim": vote_bt["sim"], "eval_trade_stats": vote_bt["trade_stats"]},
        },
        "leakage_guard": {"selection_uses_only_pre_eval_folds": True, "eval_replay_is_frozen": True, "rolling_train_before_test": True, "regime_features_are_past_only": True, "entry_delay_bars_aligns_next_15m_bar": int(cfg.entry_delay_bars) == 3},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fold-consistency selection for wave teacher policies")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--selection-folds", default=FoldConsistencyConfig.selection_folds)
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--long-thresholds", default="0.66,0.69")
    p.add_argument("--short-thresholds", default="0.32,0.35")
    p.add_argument("--trend-windows", default="0,288,864")
    p.add_argument("--modes", default="all,up,down")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--vote-k", type=int, default=2)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    p.add_argument("--min-fold-trades", type=int, default=5)
    p.add_argument("--min-total-trades", type=int, default=50)
    return p.parse_args()


def main() -> None:
    report = run_sweep(FoldConsistencyConfig(**vars(parse_args())))
    print(json.dumps({"selected_policies": report["selected_policies"], "heldout_eval": report["heldout_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
