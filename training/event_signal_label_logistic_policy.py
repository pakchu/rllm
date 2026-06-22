"""No-leak logistic baseline for signal-level action label datasets.

Train/validation/eval chronology is explicit:
- fit: train rows before validation_start
- validation: train rows inside validation window, used only for threshold selection
- eval: separate eval file, never used for fitting or threshold choice
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.online_risk_overlay_backtest import OnlineRiskOverlayConfig, run_overlay

LABELS = ["NO_TRADE", "LONG", "SHORT"]

@dataclass(frozen=True)
class Cfg:
    train_labels: str
    eval_labels: str
    output: str
    work_dir: str = "results/event_signal_label_logistic"
    validation_start: str = "2023-01-01"
    validation_end: str = "2024-12-31 23:59:59"
    market_csv: str = "data/2020-01-01_2026-06-01_btcusdt_futures_5m.csv.gz"
    min_trade_conf: str = "0.34,0.38,0.42,0.46,0.5,0.55,0.6"
    min_val_trades: int = 50
    feature_allowlist: str = ""
    use_state_tokens: int = 1
    epochs: int = 500
    lr: float = 0.08
    l2: float = 0.001


def load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def date(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))


def label(row: dict[str, Any]) -> str:
    tgt = row.get("target", {}) if isinstance(row.get("target"), dict) else {}
    act = str(tgt.get("action", "NO_TRADE"))
    return act if act in LABELS else "NO_TRADE"


def feature_names(rows: list[dict[str, Any]], allowlist: str, use_state_tokens: bool) -> tuple[list[str], list[str]]:
    allowed = {x.strip() for x in str(allowlist).split(",") if x.strip()}
    nums: set[str] = set()
    cats: set[str] = set()
    for row in rows:
        snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
        cur = {str(k) for k in snap.keys()}
        nums.update((cur & allowed) if allowed else cur)
        if use_state_tokens and not allowed:
            toks = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
            for k, v in toks.items():
                cats.add(f"{k}={v}")
    return sorted(nums), sorted(cats)


def xy(rows: list[dict[str, Any]], nums: list[str], cats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cat_index = {c: i for i, c in enumerate(cats)}
    X = np.zeros((len(rows), len(nums)+len(cats)), dtype=float)
    y = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows):
        snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
        X[i, :len(nums)] = [float(snap.get(n, 0.0) or 0.0) for n in nums]
        base = len(nums)
        toks = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
        for k, v in toks.items():
            j = cat_index.get(f"{k}={v}")
            if j is not None:
                X[i, base+j] = 1.0
        y[i] = LABELS.index(label(row))
    return X, y


def standardize_fit_apply(fit: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mu = fit.mean(axis=0)
    sd = fit.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (fit-mu)/sd, (other-mu)/sd, {"mean": mu.tolist(), "std": sd.tolist()}


def train_softmax(X: np.ndarray, y: np.ndarray, *, lr: float, epochs: int, l2: float) -> np.ndarray:
    n = X.shape[0]
    k = len(LABELS)
    Xb = np.c_[np.ones(n), X]
    W = np.zeros((Xb.shape[1], k), dtype=float)
    Y = np.eye(k)[y]
    for _ in range(epochs):
        z = Xb @ W
        z -= z.max(axis=1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(axis=1, keepdims=True)
        grad = Xb.T @ (P-Y) / n + l2 * W
        grad[0] = Xb.T[0] @ (P-Y) / n
        W -= lr * grad
    return W


def predict_proba(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.c_[np.ones(len(X)), X]
    z = Xb @ W
    z -= z.max(axis=1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(axis=1, keepdims=True)
    return P


def metrics(y: np.ndarray, P: np.ndarray) -> dict[str, Any]:
    pred = P.argmax(axis=1)
    conf = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for a, b in zip(y, pred):
        conf[int(a), int(b)] += 1
    trade_mask = y != 0
    pred_trade_mask = pred != 0
    side_mask = trade_mask & pred_trade_mask
    return {
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "trade_recall": float((pred_trade_mask & trade_mask).sum()/max(1, trade_mask.sum())),
        "trade_precision": float((pred_trade_mask & trade_mask).sum()/max(1, pred_trade_mask.sum())),
        "side_accuracy_when_both_trade": float((pred[side_mask] == y[side_mask]).mean()) if side_mask.any() else 0.0,
        "pred_counts": dict(zip(LABELS, np.bincount(pred, minlength=len(LABELS)).astype(int).tolist())),
        "label_counts": dict(zip(LABELS, np.bincount(y, minlength=len(LABELS)).astype(int).tolist())),
        "confusion_rows_actual_cols_pred": {LABELS[i]: dict(zip(LABELS, conf[i].astype(int).tolist())) for i in range(len(LABELS))},
    }


def write_predictions(rows: list[dict[str, Any]], P: np.ndarray, path: str, threshold: float) -> dict[str, Any]:
    out=[]
    counts=Counter()
    for row, p in zip(rows, P):
        j=int(np.argmax(p)); lab=LABELS[j]; score=float(p[j])
        if lab != "NO_TRADE" and score >= threshold:
            pred = {"gate":"TRADE", "side": lab, "hold_bars": 288, "confidence":"HIGH", "family":"event_signal_label_logistic"}
            scale=0.5
            counts["TRADE"] += 1; counts[lab] += 1
        else:
            pred = {"gate":"NO_TRADE", "side":"NONE", "hold_bars":0, "confidence":"LOW", "family":"event_signal_label_logistic"}
            scale=0.0
            counts["NO_TRADE"] += 1
        out.append({"date":row["date"], "signal_pos":row["signal_pos"], "prediction":pred, "position_scale":scale, "probs":dict(zip(LABELS, map(float, p)))})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, sort_keys=True) for r in out)+"\n")
    return {"rows": len(out), "counts": dict(counts), "threshold": threshold, "output": path}


def run(cfg: Cfg) -> dict[str, Any]:
    train_rows = load(cfg.train_labels)
    eval_rows = load(cfg.eval_labels)
    fit = [r for r in train_rows if date(r) < cfg.validation_start]
    val = [r for r in train_rows if cfg.validation_start <= date(r) <= cfg.validation_end]
    nums, cats = feature_names(fit, cfg.feature_allowlist, bool(int(cfg.use_state_tokens)))
    Xf, yf = xy(fit, nums, cats)
    Xv, yv = xy(val, nums, cats)
    Xe, ye = xy(eval_rows, nums, cats)
    Xfz, Xvz, scaler = standardize_fit_apply(Xf, Xv)
    _, Xez, _ = standardize_fit_apply(Xf, Xe)
    W = train_softmax(Xfz, yf, lr=float(cfg.lr), epochs=int(cfg.epochs), l2=float(cfg.l2))
    Pv = predict_proba(Xvz, W)
    Pe = predict_proba(Xez, W)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    val_results=[]
    for threshold in [float(x) for x in str(cfg.min_trade_conf).split(",") if x.strip()]:
        pred_path = str(Path(cfg.work_dir)/f"val_t{threshold}.jsonl")
        pred_summary = write_predictions(val, Pv, pred_path, threshold)
        bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=pred_path, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir)/f"val_t{threshold}.bt.json"), leverage=1.0, entry_delay_bars=1))
        score = float(bt["sim"]["cagr_to_strict_mdd"])
        if int(bt["sim"]["trade_entries"]) < int(cfg.min_val_trades):
            score -= 1000.0
        val_results.append({"threshold": threshold, "prediction_summary": pred_summary, "val_sim": bt["sim"], "val_trade_stats": bt["trade_stats"], "score": score})
    val_results.sort(key=lambda x: x["score"], reverse=True)
    selected = val_results[0]
    eval_pred_path = str(Path(cfg.work_dir)/"selected_eval_predictions.jsonl")
    eval_pred_summary = write_predictions(eval_rows, Pe, eval_pred_path, float(selected["threshold"]))
    eval_bt = run_overlay(OnlineRiskOverlayConfig(predictions_jsonl=eval_pred_path, market_csv=cfg.market_csv, output=str(Path(cfg.work_dir)/"selected_eval_backtest.json"), leverage=1.0, entry_delay_bars=1))
    report = {
        "config": cfg.__dict__,
        "rows": {"fit": len(fit), "val": len(val), "eval": len(eval_rows)},
        "features": {"numeric": len(nums), "categorical": len(cats), "names": nums, "categorical_count": len(cats)},
        "fit_metrics": metrics(yf, predict_proba(Xfz, W)),
        "val_metrics": metrics(yv, Pv),
        "eval_metrics": metrics(ye, Pe),
        "top_val": val_results,
        "selected": selected,
        "eval_prediction_summary": eval_pred_summary,
        "eval_backtest": {"sim": eval_bt["sim"], "trade_stats": eval_bt["trade_stats"]},
        "leakage_guard": "fit uses only rows before validation_start; validation selects threshold; eval file is untouched until final selected threshold",
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument("--train-labels", required=True)
    p.add_argument("--eval-labels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", default=Cfg.work_dir)
    p.add_argument("--validation-start", default=Cfg.validation_start)
    p.add_argument("--validation-end", default=Cfg.validation_end)
    p.add_argument("--market-csv", default=Cfg.market_csv)
    p.add_argument("--min-trade-conf", default=Cfg.min_trade_conf)
    p.add_argument("--min-val-trades", type=int, default=Cfg.min_val_trades)
    p.add_argument("--feature-allowlist", default=Cfg.feature_allowlist)
    p.add_argument("--use-state-tokens", type=int, default=Cfg.use_state_tokens)
    p.add_argument("--epochs", type=int, default=Cfg.epochs)
    p.add_argument("--lr", type=float, default=Cfg.lr)
    p.add_argument("--l2", type=float, default=Cfg.l2)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
