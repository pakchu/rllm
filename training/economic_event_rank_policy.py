"""Combine opportunity detection with pairwise action ranking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.economic_action_backtest import run_economic_action_backtest
from training.economic_opportunity_baseline import best_rows_by_signal, fit_logistic, signal_feature_row, _metrics
from training.economic_pairwise_baseline import build_pair_indices, fit_pairwise_logistic, group_by_signal
from training.economic_value_baseline import FeatureSpace, load_jsonl, write_jsonl


def _action_obj(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except Exception:
        obj = {}
    return {"gate": str(obj.get("gate", "NO_TRADE")), "side": str(obj.get("side", "NONE")), "hold_bars": int(obj.get("hold_bars", 0) or 0)}


def run_event_rank_policy(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    utility_threshold: float = 0.003,
    event_threshold: float = 0.7,
    min_utility_gap: float = 0.005,
    event_l2: float = 1.0,
    rank_l2: float = 1.0,
    epochs: int = 400,
    lr: float = 0.2,
) -> dict[str, Any]:
    train_rows = load_jsonl(train_jsonl)
    eval_rows = load_jsonl(eval_jsonl)

    # Event detector on signal-only features.
    train_best = best_rows_by_signal(train_rows)
    eval_best = best_rows_by_signal(eval_rows)
    train_signal = [signal_feature_row(r) for r in train_best]
    eval_signal = [signal_feature_row(r) for r in eval_best]
    fs_event = FeatureSpace(min_count=2)
    fs_event.fit(train_signal)
    x_event_train = fs_event.matrix(train_signal, fit_scale=True)
    x_event_eval = fs_event.matrix(eval_signal)
    y_event_train = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in train_best], dtype=np.float64)
    y_event_eval = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in eval_best], dtype=np.float64)
    w_event = fit_logistic(x_event_train, y_event_train, lr=lr, l2=event_l2, epochs=epochs)
    event_train_scores = 1.0 / (1.0 + np.exp(-np.clip(x_event_train @ w_event, -40, 40)))
    event_eval_scores = 1.0 / (1.0 + np.exp(-np.clip(x_event_eval @ w_event, -40, 40)))
    event_by_key = {(str(r.get("date")), int(r.get("signal_pos", -1))): float(s) for r, s in zip(eval_best, event_eval_scores)}

    # Pairwise action ranker on action-conditioned features.
    fs_rank = FeatureSpace(min_count=2)
    fs_rank.fit(train_rows)
    x_rank_train = fs_rank.matrix(train_rows, fit_scale=True)
    x_rank_eval = fs_rank.matrix(eval_rows)
    pairs = build_pair_indices(train_rows, max_pairs_per_signal=10, min_utility_gap=min_utility_gap)
    w_rank = fit_pairwise_logistic(x_rank_train, pairs, lr=lr, l2=rank_l2, epochs=epochs)
    rank_scores = x_rank_eval @ w_rank

    grouped = group_by_signal(eval_rows)
    row_pos = {id(r): i for i, r in enumerate(eval_rows)}
    preds = []
    for key in sorted(grouped):
        ev_score = event_by_key.get(key, 0.0)
        if ev_score < event_threshold:
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
            rank_score = 0.0
            actual_u = 0.0
        else:
            best = max(grouped[key], key=lambda r: float(rank_scores[row_pos[id(r)]]))
            action = _action_obj(str(best.get("action", "{}")))
            rank_score = float(rank_scores[row_pos[id(best)]])
            actual_u = float(best.get("utility", 0.0))
        preds.append({"date": key[0], "signal_pos": key[1], "prediction": action, "event_score": ev_score, "rank_score": rank_score, "actual_utility": actual_u})
    write_jsonl(predictions_output, preds)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(f"{p['prediction']['gate']}/{p['prediction']['side']}/{p['prediction'].get('hold_bars',0)}" for p in preds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": {"utility_threshold": utility_threshold, "event_threshold": event_threshold, "min_utility_gap": min_utility_gap, "event_l2": event_l2, "rank_l2": rank_l2, "epochs": epochs, "lr": lr, "rank_pairs": len(pairs)},
        "train_event_metrics": _metrics(y_event_train, event_train_scores, event_threshold),
        "eval_event_metrics": _metrics(y_event_eval, event_eval_scores, event_threshold),
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"event_and_rank_fit_on_train_only": True, "eval_utility_not_used_for_selection": True},
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
    p.add_argument("--event-threshold", type=float, default=0.7)
    p.add_argument("--min-utility-gap", type=float, default=0.005)
    p.add_argument("--event-l2", type=float, default=1.0)
    p.add_argument("--rank-l2", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_event_rank_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
