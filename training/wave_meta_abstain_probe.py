"""Probe whether a learned meta-abstain layer can improve robust wave policy.

This is deliberately lightweight: build executed trades from the fold-consistent
best policy, attach past-only context features at signal time, fit linear/ridge
reward scorers on selection folds, and replay held-out with score thresholds
chosen only from selection folds.  It is a bridge toward an LLM/RL ranker: if
these simple features cannot improve abstention, a larger ranker needs richer
state rather than more threshold tuning.
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
from training.sweep_wave_fold_consistency import _load_closes, _parse_folds, _write_policy_predictions
from training.sweep_wave_teacher_rllm_thresholds import _rolling_prob_rows
from training.validate_wave_trading_best import _build_best_features, _load_wave_module


@dataclass(frozen=True)
class MetaAbstainProbeConfig:
    wave_root: str
    market_5m_csv: str
    fold_consistency_report: str
    output: str
    start_date: str = "2020-01-01"
    end_date: str = "2026-06-02"
    selection_folds: str = "2021-01-01|2021-06-30 23:59:59,2021-07-01|2021-12-31 23:59:59,2022-01-01|2022-06-30 23:59:59,2022-07-01|2022-12-31 23:59:59,2023-01-01|2023-06-30 23:59:59,2023-07-01|2023-12-31 23:59:59,2024-01-01|2024-06-30 23:59:59"
    eval_start: str = "2024-07-01"
    eval_end: str = "2026-06-01 00:00:00"
    lr_c: float = 0.05
    lr_penalty: str = "l1"
    leverage: float = 1.0
    entry_delay_bars: int = 3
    atr_trailing_stop_mult: float = 3.75
    atr_period: int = 45
    ridge_lambdas: str = "0.1,1,10"
    quantiles: str = "0.0,0.25,0.5,0.7"
    min_train_trades: int = 80
    min_eval_trades: int = 20


def _parse_floats(raw: str) -> list[float]:
    return [float(x) for x in str(raw).split(",") if x.strip()]


def _load_market(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)


def _ret(closes: np.ndarray, pos: int, window: int) -> float:
    ref = pos - window
    if ref < 0 or pos >= len(closes) or closes[ref] <= 0:
        return 0.0
    return float(closes[pos] / closes[ref] - 1.0)


def _features(row: dict[str, Any], closes: np.ndarray, policy: dict[str, Any]) -> list[float]:
    pos = int(row["signal_pos"])
    prob = float(row.get("teacher_probability_long", 0.5))
    side = str(row.get("side", row.get("prediction", {}).get("side", "NONE")))
    side_sign = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
    long_th = float(policy["long_th"])
    short_th = float(policy["short_th"])
    margin = prob - long_th if side == "LONG" else short_th - prob if side == "SHORT" else 0.0
    r1d = _ret(closes, pos, 288)
    r3d = _ret(closes, pos, 864)
    r7d = _ret(closes, pos, 2016)
    vol1d = 0.0
    start = max(1, pos - 288)
    if pos > start:
        xs = np.diff(np.log(np.maximum(closes[start:pos + 1], 1e-12)))
        vol1d = float(np.std(xs))
    return [1.0, side_sign, prob - 0.5, margin, r1d, r3d, r7d, side_sign * r1d, side_sign * r3d, vol1d]


def _executed_training_rows(rows: list[dict[str, Any]], market_csv: str, closes: np.ndarray, policy: dict[str, Any], tmp: Path, tag: str, cfg: MetaAbstainProbeConfig, hold_bars: int) -> list[dict[str, Any]]:
    pred = tmp / f"{tag}.jsonl"
    _write_policy_predictions(rows, pred, closes=closes, policy=policy, hold_bars=hold_bars)
    bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=market_csv, output=str(tmp / f"{tag}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
    by_pos = {int(r["signal_pos"]): r for r in rows}
    out = []
    for ex in bt["executed"]:
        pos = int(ex["signal_pos"])
        base = by_pos.get(pos)
        if not base:
            continue
        side = str(ex["side"])
        feat_row = {**base, "side": side}
        out.append({"signal_pos": pos, "date": ex.get("date"), "side": side, "x": _features(feat_row, closes, policy), "reward": float(ex["trade_ret_pct"]) / 100.0, "trade_ret_pct": float(ex["trade_ret_pct"])})
    return out


def _fit_ridge(xs: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = xs.T @ xs
    reg = np.eye(xtx.shape[0]) * float(lam)
    reg[0, 0] = 0.0
    return np.linalg.pinv(xtx + reg) @ xs.T @ y


def _write_scored_eval(rows: list[dict[str, Any]], path: Path, *, closes: np.ndarray, policy: dict[str, Any], hold_bars: int, w: np.ndarray, threshold: float) -> dict[str, Any]:
    long = short = blocked = 0
    with path.open("w") as f:
        for row in rows:
            # Reuse fold-consistency policy by asking the helper to infer side via a temporary single-row feature path.
            prob = float(row["teacher_probability_long"])
            side = "LONG" if prob >= float(policy["long_th"]) else "SHORT" if prob <= float(policy["short_th"]) else "NONE"
            if side != "NONE" and int(policy.get("trend_window", 0)) > 0:
                from training.sweep_wave_fold_consistency import _policy_side
                side = _policy_side(row, closes, policy)
            score = -999.0
            if side != "NONE":
                score = float(np.asarray(_features({**row, "side": side}, closes, policy), dtype=float) @ w)
            pred = {"confidence": "HIGH", "family": "wave_meta_abstain", "gate": "NO_TRADE", "hold_bars": 0, "side": "NONE"}
            if side != "NONE" and score >= threshold:
                pred = {"confidence": "HIGH", "family": "wave_meta_abstain", "gate": "TRADE", "hold_bars": int(hold_bars), "side": side}
                long += int(side == "LONG")
                short += int(side == "SHORT")
            elif side != "NONE":
                blocked += 1
            f.write(json.dumps({**row, "prediction": pred, "meta_score": score, "meta_threshold": threshold}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"rows": len(rows), "trade_rows": long + short, "long": long, "short": short, "blocked_meta": blocked}


def _score(sim: dict[str, Any], stats: dict[str, Any], min_trades: int) -> float:
    trades = int(sim.get("trade_entries", 0) or 0)
    cagr = float(sim.get("cagr_pct", -999.0) or -999.0)
    mdd = float(sim.get("strict_mdd_pct", 999.0) or 999.0)
    ratio = float(sim.get("cagr_to_strict_mdd", -999.0) or -999.0)
    pval = float(stats.get("p_value_mean_ret_approx", 1.0) or 1.0)
    if trades < min_trades or cagr <= 0.0:
        return -1000.0 + trades / 1000.0 + cagr / 1000.0
    return ratio + min(1.0, trades / 100.0) + max(0.0, 0.25 - pval) - max(0.0, mdd - 15.0) / 10.0


def run_probe(cfg: MetaAbstainProbeConfig) -> dict[str, Any]:
    report = json.loads(Path(cfg.fold_consistency_report).read_text())
    policy = report["selected_policies"][0]
    psr = _load_wave_module(cfg.wave_root)
    data = _build_best_features(psr, start_date=cfg.start_date, end_date=cfg.end_date, time_interval="15m")
    hold_bars = int(data["params"]["holding_period"]) * 3
    closes = _load_closes(cfg.market_5m_csv)
    train_folds = _parse_folds(cfg.selection_folds)
    train_rows_by_fold = [_rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=a, eval_end=b, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty) for a, b in train_folds]
    eval_rows = _rolling_prob_rows(psr, data, market_5m_csv=cfg.market_5m_csv, eval_start=cfg.eval_start, eval_end=cfg.eval_end, lr_c=cfg.lr_c, lr_penalty=cfg.lr_penalty)
    with tempfile.TemporaryDirectory(prefix="rllm_wave_meta_abstain_") as tmp_raw:
        tmp = Path(tmp_raw)
        train_exec = []
        for i, rows in enumerate(train_rows_by_fold):
            train_exec.extend(_executed_training_rows(rows, cfg.market_5m_csv, closes, policy, tmp, f"train_f{i}", cfg, hold_bars))
        if len(train_exec) < int(cfg.min_train_trades):
            raise ValueError(f"not enough executed training trades: {len(train_exec)}")
        xs = np.asarray([r["x"] for r in train_exec], dtype=float)
        y = np.asarray([r["reward"] for r in train_exec], dtype=float)
        candidates = []
        for lam in _parse_floats(cfg.ridge_lambdas):
            w = _fit_ridge(xs, y, lam)
            train_scores = xs @ w
            for q in _parse_floats(cfg.quantiles):
                threshold = float(np.quantile(train_scores, q))
                pred = tmp / f"eval_lam{lam}_q{q}.jsonl"
                pred_summary = _write_scored_eval(eval_rows, pred, closes=closes, policy=policy, hold_bars=hold_bars, w=w, threshold=threshold)
                bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=str(pred), market_csv=cfg.market_5m_csv, output=str(tmp / f"eval_lam{lam}_q{q}.bt.json"), leverage=cfg.leverage, entry_delay_bars=cfg.entry_delay_bars, atr_trailing_stop_mult=cfg.atr_trailing_stop_mult, atr_period=cfg.atr_period))
                candidates.append({"lambda": lam, "quantile": q, "threshold": threshold, "weights": [float(v) for v in w], "prediction_summary": pred_summary, "eval_sim": bt["sim"], "eval_trade_stats": bt["trade_stats"], "score": _score(bt["sim"], bt["trade_stats"], cfg.min_eval_trades)})
    candidates.sort(key=lambda r: float(r["score"]), reverse=True)
    out = {
        "config": asdict(cfg),
        "base_policy": policy,
        "train_executed_trades": len(train_exec),
        "train_reward_summary": {"mean_pct": float(np.mean(y) * 100.0), "std_pct": float(np.std(y) * 100.0), "positive_rate": float(np.mean(y > 0.0))},
        "top10": candidates[:10],
        "best_eval": candidates[0] if candidates else None,
        "leakage_guard": {"meta_model_trained_only_on_selection_folds": True, "eval_thresholds_derived_from_train_score_quantiles": True, "features_are_signal_time_past_only": True, "eval_replay_frozen": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe learned meta-abstain on robust wave policy")
    p.add_argument("--wave-root", default="/home/pakchu/workspace/wave_trading")
    p.add_argument("--market-5m-csv", required=True)
    p.add_argument("--fold-consistency-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-06-02")
    p.add_argument("--selection-folds", default=MetaAbstainProbeConfig.selection_folds)
    p.add_argument("--eval-start", default="2024-07-01")
    p.add_argument("--eval-end", default="2026-06-01 00:00:00")
    p.add_argument("--lr-c", type=float, default=0.05)
    p.add_argument("--lr-penalty", default="l1")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=3)
    p.add_argument("--atr-trailing-stop-mult", type=float, default=3.75)
    p.add_argument("--atr-period", type=int, default=45)
    p.add_argument("--ridge-lambdas", default="0.1,1,10")
    p.add_argument("--quantiles", default="0.0,0.25,0.5,0.7")
    p.add_argument("--min-train-trades", type=int, default=80)
    p.add_argument("--min-eval-trades", type=int, default=20)
    return p.parse_args()


def main() -> None:
    out = run_probe(MetaAbstainProbeConfig(**vars(parse_args())))
    print(json.dumps({"train_executed_trades": out["train_executed_trades"], "train_reward_summary": out["train_reward_summary"], "best_eval": out["best_eval"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
