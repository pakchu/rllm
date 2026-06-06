"""Leakage-safe direct utility baseline for economic value rows.

This is intentionally simple: parse past-only analyzer summaries plus candidate
action, fit a ridge regressor on train utility labels, then pick the highest
predicted utility per signal for strict backtesting.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import run_economic_action_backtest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _summary_obj(prompt: str) -> dict[str, Any]:
    marker = "Past-only analyzer summary:"
    text = str(prompt).split(marker, 1)[1].strip() if marker in str(prompt) else str(prompt)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _action_obj(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except Exception:
        obj = {}
    return {"gate": str(obj.get("gate", "NO_TRADE")), "side": str(obj.get("side", "NONE")), "hold_bars": int(obj.get("hold_bars", 0) or 0)}


def row_features(row: dict[str, Any]) -> dict[str, float]:
    s = _summary_obj(str(row.get("prompt", "")))
    action = _action_obj(str(row.get("action", "{}")))
    feats: dict[str, float] = {"bias": 1.0}
    for prefix, mapping in (("evidence", s.get("evidence", {})), ("seq", s.get("sequence_stats", {}))):
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                try:
                    feats[f"{prefix}:{k}"] = float(v)
                except Exception:
                    pass
    # Top-level and symbolic categoricals.
    cat_items: list[tuple[str, Any]] = []
    for k in ("candle_pattern", "location", "momentum", "oscillator", "regime", "risk_state", "trend_alignment", "trend_strength", "volatility_level", "volume_state"):
        cat_items.append((k, s.get(k)))
    sym = s.get("symbolic_features", {})
    if isinstance(sym, dict):
        for k, v in sym.items():
            cat_items.append((f"sym:{k}", v))
    for tag in s.get("context_tags", []) if isinstance(s.get("context_tags"), list) else []:
        cat_items.append(("tag", tag))
    for k, v in cat_items:
        if v is not None:
            feats[f"cat:{k}={v}"] = 1.0
    # Action features and interactions with compact regime descriptors.
    gate = action["gate"]
    side = action["side"]
    hold = int(action["hold_bars"])
    feats[f"action:gate={gate}"] = 1.0
    feats[f"action:side={side}"] = 1.0
    feats[f"action:hold={hold}"] = 1.0
    feats["action:hold_scaled"] = hold / 432.0
    feats["action:is_trade"] = 1.0 if gate == "TRADE" else 0.0
    for k in ("regime", "risk_state", "trend_alignment", "location", "momentum"):
        v = s.get(k)
        if v is not None:
            feats[f"inter:{k}={v}|side={side}"] = 1.0
            feats[f"inter:{k}={v}|hold={hold}"] = 1.0
    return feats


class FeatureSpace:
    def __init__(self, min_count: int = 2):
        self.min_count = int(min_count)
        self.names: list[str] = []
        self.index: dict[str, int] = {}
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(row_features(row).keys())
        self.names = [k for k, c in sorted(counts.items()) if c >= self.min_count or k == "bias"]
        self.index = {k: i for i, k in enumerate(self.names)}

    def matrix(self, rows: list[dict[str, Any]], *, fit_scale: bool = False) -> np.ndarray:
        x = np.zeros((len(rows), len(self.names)), dtype=np.float64)
        for r, row in enumerate(rows):
            for k, v in row_features(row).items():
                i = self.index.get(k)
                if i is not None:
                    x[r, i] = float(v)
        if fit_scale:
            self.mean = x.mean(axis=0)
            self.std = x.std(axis=0)
            self.std[self.std < 1e-9] = 1.0
            if "bias" in self.index:
                bi = self.index["bias"]
                self.mean[bi] = 0.0
                self.std[bi] = 1.0
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return x


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    return np.linalg.solve(xtx + reg, x.T @ y)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def predict_rows(rows: list[dict[str, Any]], fs: FeatureSpace, w: np.ndarray) -> list[dict[str, Any]]:
    x = fs.matrix(rows)
    preds = x @ w
    out = []
    for row, pred in zip(rows, preds):
        action = _action_obj(str(row.get("action", "{}")))
        out.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "prediction": action, "predicted_utility": float(pred), "actual_utility": float(row.get("utility", 0.0)), "action": action})
    return out


def choose_best_per_signal(pred_rows: list[dict[str, Any]], *, threshold: float = 0.0) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        grouped[(str(row.get("date")), int(row.get("signal_pos", -1)))] .append(row)
    chosen: list[dict[str, Any]] = []
    for key in sorted(grouped):
        best = max(grouped[key], key=lambda r: float(r["predicted_utility"]))
        pred = dict(best["prediction"])
        if float(best["predicted_utility"]) < float(threshold):
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        chosen.append({"date": best["date"], "signal_pos": best["signal_pos"], "prediction": pred, "predicted_utility": best["predicted_utility"], "actual_utility": best["actual_utility"]})
    return chosen


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def run_value_baseline(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    alpha: float = 10.0,
    min_feature_count: int = 2,
    threshold: float = 0.0,
) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    ev = load_jsonl(eval_jsonl)
    fs = FeatureSpace(min_count=min_feature_count)
    fs.fit(train)
    x_train = fs.matrix(train, fit_scale=True)
    y_train = np.array([float(r.get("utility", 0.0)) for r in train], dtype=np.float64)
    w = fit_ridge(x_train, y_train, alpha=alpha)
    train_pred = x_train @ w
    eval_pred_rows = predict_rows(ev, fs, w)
    chosen = choose_best_per_signal(eval_pred_rows, threshold=threshold)
    write_jsonl(predictions_output, chosen)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(f"{r['prediction']['gate']}/{r['prediction']['side']}/{r['prediction'].get('hold_bars',0)}" for r in chosen)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "market_csv": market_csv,
        "config": {"alpha": alpha, "min_feature_count": min_feature_count, "threshold": threshold, "features": len(fs.names)},
        "fit": {"train_rows": len(train), "eval_rows": len(ev), "train_corr": _corr(train_pred, y_train), "train_rmse_pct": math.sqrt(float(np.mean((train_pred - y_train) ** 2))) * 100.0},
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"feature_space_fit_on_train_only": True, "eval_utility_not_used_for_fit_or_selection": True, "strict_backtest_uses_predicted_actions": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate direct economic value ridge baseline")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--min-feature-count", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_value_baseline(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
