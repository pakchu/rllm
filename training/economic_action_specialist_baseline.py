"""Action-specialist opportunity detectors for economic value rows.

Train one event detector per concrete action: "would this action's utility exceed
threshold?".  At evaluation, choose the action with the highest specialist
probability above a threshold.  This is live-safe and tests whether direction /
horizon selection can improve over the majority LONG/432 baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import run_economic_action_backtest
from training.economic_opportunity_baseline import fit_logistic, _metrics
from training.economic_value_baseline import FeatureSpace, load_jsonl, write_jsonl


def _signal_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1)))


def _action_obj(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(str(text))
    except Exception:
        obj = {}
    return {"gate": str(obj.get("gate", "NO_TRADE")), "side": str(obj.get("side", "NONE")), "hold_bars": int(obj.get("hold_bars", 0) or 0)}


def _action_key_from_obj(obj: dict[str, Any]) -> str:
    return f"{obj.get('gate','NO_TRADE')}/{obj.get('side','NONE')}/{int(obj.get('hold_bars',0) or 0)}"


def _action_key(row: dict[str, Any]) -> str:
    return _action_key_from_obj(_action_obj(str(row.get("action", "{}"))))


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[_signal_key(row)][_action_key(row)] = row
    return grouped


def signal_rows_for_action(rows: list[dict[str, Any]], action_key: str) -> list[dict[str, Any]]:
    out = []
    for _key, actions in sorted(group_rows(rows).items()):
        row = actions.get(action_key)
        if row is not None:
            # Keep prompt but neutralize action so detector learns signal context,
            # not the action token (one detector per action already encodes action).
            out.append({**row, "action": json.dumps({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, sort_keys=True)})
    return out


def fit_specialists(
    train_rows: list[dict[str, Any]],
    *,
    utility_threshold: float,
    min_positive: int = 20,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 400,
) -> tuple[FeatureSpace, dict[str, np.ndarray], dict[str, Any]]:
    trade_actions = sorted(k for k in {_action_key(r) for r in train_rows} if not k.startswith("NO_TRADE"))
    # Shared signal-only feature space fit on all action-neutralized rows.
    neutral_rows = []
    for action in trade_actions:
        neutral_rows.extend(signal_rows_for_action(train_rows, action))
    fs = FeatureSpace(min_count=2)
    fs.fit(neutral_rows)
    models: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}
    for action in trade_actions:
        rows = signal_rows_for_action(train_rows, action)
        y = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in rows], dtype=np.float64)
        positives = int(y.sum())
        stats[action] = {"rows": len(rows), "positives": positives, "positive_rate": float(y.mean()) if len(y) else 0.0}
        if positives < int(min_positive):
            continue
        x = fs.matrix(rows, fit_scale=(fs.mean is None)) if fs.mean is None else fs.matrix(rows)
        models[action] = fit_logistic(x, y, lr=lr, l2=l2, epochs=epochs)
        scores = 1.0 / (1.0 + np.exp(-np.clip(x @ models[action], -40, 40)))
        stats[action]["train_metrics_at_0p5"] = _metrics(y, scores, 0.5)
    return fs, models, stats


def predict_specialists(
    eval_rows: list[dict[str, Any]],
    fs: FeatureSpace,
    models: dict[str, np.ndarray],
    *,
    score_threshold: float,
) -> list[dict[str, Any]]:
    grouped = group_rows(eval_rows)
    preds: list[dict[str, Any]] = []
    # Precompute score per action/signal.
    action_rows = {a: signal_rows_for_action(eval_rows, a) for a in models}
    action_scores: dict[str, dict[tuple[str, int], float]] = {}
    for action, rows in action_rows.items():
        if not rows:
            action_scores[action] = {}
            continue
        x = fs.matrix(rows)
        scores = 1.0 / (1.0 + np.exp(-np.clip(x @ models[action], -40, 40)))
        action_scores[action] = {_signal_key(r): float(s) for r, s in zip(rows, scores)}
    for key in sorted(grouped):
        best_action = ""
        best_score = float("-inf")
        for action in models:
            score = action_scores.get(action, {}).get(key, 0.0)
            if score > best_score:
                best_score = score
                best_action = action
        if best_score < float(score_threshold) or not best_action:
            action_obj = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        else:
            row = grouped[key][best_action]
            action_obj = _action_obj(str(row.get("action", "{}")))
        preds.append({"date": key[0], "signal_pos": key[1], "prediction": action_obj, "specialist_score": best_score, "specialist_action": best_action})
    return preds


def run_action_specialist_baseline(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    utility_threshold: float = 0.003,
    score_threshold: float = 0.7,
    min_positive: int = 20,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 400,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_jsonl)
    eval_rows = load_jsonl(eval_jsonl)
    fs, models, specialist_stats = fit_specialists(
        train_rows,
        utility_threshold=utility_threshold,
        min_positive=min_positive,
        lr=lr,
        l2=l2,
        epochs=epochs,
    )
    preds = predict_specialists(eval_rows, fs, models, score_threshold=score_threshold)
    write_jsonl(predictions_output, preds)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(_action_key_from_obj(p["prediction"]) for p in preds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": {"utility_threshold": utility_threshold, "score_threshold": score_threshold, "min_positive": min_positive, "lr": lr, "l2": l2, "epochs": epochs, "features": len(fs.names), "specialists": sorted(models)},
        "specialist_stats": specialist_stats,
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"specialists_fit_on_train_only": True, "eval_utility_not_used_for_selection": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--utility-threshold", type=float, default=0.003)
    p.add_argument("--score-threshold", type=float, default=0.7)
    p.add_argument("--min-positive", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=400)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_action_specialist_baseline(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
