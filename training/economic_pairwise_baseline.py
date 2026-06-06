"""Pairwise/ranking learnability baseline for economic value rows.

For each signal, compare candidate action pairs and learn whether action A has
higher strict utility than action B.  Evaluation ranks all actions per signal by
Bradley-Terry-style aggregate pairwise wins and strict-backtests the selected
action.  This checks whether the action ordering is learnable at all before
training another LLM head.
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
from training.economic_value_baseline import FeatureSpace, load_jsonl, row_features, write_jsonl


def _signal_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1)))


def _action_key(row: dict[str, Any]) -> str:
    try:
        obj = json.loads(str(row.get("action", "{}")))
    except Exception:
        obj = {}
    return f"{obj.get('gate','NO_TRADE')}/{obj.get('side','NONE')}/{int(obj.get('hold_bars',0) or 0)}"


def group_by_signal(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_signal_key(row)].append(row)
    return grouped


def build_pair_indices(rows: list[dict[str, Any]], *, max_pairs_per_signal: int = 10, min_utility_gap: float = 0.001) -> list[tuple[int, int, int]]:
    """Return (winner_idx, loser_idx, label=1) pairs.

    Pairs are deterministic: compare the best action for a signal against up to
    max_pairs_per_signal lower-utility alternatives whose gap is large enough.
    """

    index_by_id = {id(row): i for i, row in enumerate(rows)}
    pairs: list[tuple[int, int, int]] = []
    for group in group_by_signal(rows).values():
        ordered = sorted(group, key=lambda r: float(r.get("utility", 0.0)), reverse=True)
        if len(ordered) < 2:
            continue
        best = ordered[0]
        best_u = float(best.get("utility", 0.0))
        count = 0
        for other in ordered[1:]:
            if best_u - float(other.get("utility", 0.0)) < float(min_utility_gap):
                continue
            pairs.append((index_by_id[id(best)], index_by_id[id(other)], 1))
            count += 1
            if count >= int(max_pairs_per_signal):
                break
    return pairs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def fit_pairwise_logistic(
    x: np.ndarray,
    pairs: list[tuple[int, int, int]],
    *,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 200,
) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    if not pairs:
        return w
    a_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    b_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    d = x[a_idx] - x[b_idx]
    n = float(len(pairs))
    step = float(lr)
    for _ in range(int(epochs)):
        margin = d @ w
        prob = _sigmoid(margin)
        grad = -(d.T @ (1.0 - prob)) / n + float(l2) * w / n
        w -= step * grad
    return w


def rank_actions(rows: list[dict[str, Any]], scores: np.ndarray, *, threshold: float = 0.0) -> list[dict[str, Any]]:
    by_key = group_by_signal(rows)
    row_pos = {id(row): i for i, row in enumerate(rows)}
    out: list[dict[str, Any]] = []
    for key in sorted(by_key):
        group = by_key[key]
        best = max(group, key=lambda r: float(scores[row_pos[id(r)]]))
        score = float(scores[row_pos[id(best)]])
        try:
            action = json.loads(str(best.get("action", "{}")))
        except Exception:
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        if score < float(threshold):
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        out.append({"date": best.get("date"), "signal_pos": best.get("signal_pos"), "prediction": action, "rank_score": score, "actual_utility": float(best.get("utility", 0.0)), "action_key": _action_key(best)})
    return out


def _top1_oracle_metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    grouped = group_by_signal(rows)
    row_pos = {id(row): i for i, row in enumerate(rows)}
    exact = 0
    uplift = []
    selected_util = []
    oracle_util = []
    for group in grouped.values():
        pred = max(group, key=lambda r: float(scores[row_pos[id(r)]]))
        oracle = max(group, key=lambda r: float(r.get("utility", 0.0)))
        if _action_key(pred) == _action_key(oracle):
            exact += 1
        pu = float(pred.get("utility", 0.0))
        ou = float(oracle.get("utility", 0.0))
        selected_util.append(pu)
        oracle_util.append(ou)
        uplift.append(pu)
    n = max(1, len(grouped))
    return {
        "signals": len(grouped),
        "top1_exact_best_action": exact / n,
        "selected_utility_mean_pct": sum(selected_util) / n * 100.0,
        "oracle_utility_mean_pct": sum(oracle_util) / n * 100.0,
        "oracle_capture_ratio": (sum(selected_util) / sum(oracle_util)) if abs(sum(oracle_util)) > 1e-12 else 0.0,
    }


def run_pairwise_baseline(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    max_pairs_per_signal: int = 10,
    min_utility_gap: float = 0.001,
    min_feature_count: int = 2,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 200,
    threshold: float = 0.0,
) -> dict[str, Any]:
    train = load_jsonl(train_jsonl)
    ev = load_jsonl(eval_jsonl)
    fs = FeatureSpace(min_count=min_feature_count)
    fs.fit(train)
    x_train = fs.matrix(train, fit_scale=True)
    x_eval = fs.matrix(ev)
    pairs = build_pair_indices(train, max_pairs_per_signal=max_pairs_per_signal, min_utility_gap=min_utility_gap)
    w = fit_pairwise_logistic(x_train, pairs, lr=lr, l2=l2, epochs=epochs)
    train_scores = x_train @ w
    eval_scores = x_eval @ w
    chosen = rank_actions(ev, eval_scores, threshold=threshold)
    write_jsonl(predictions_output, chosen)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(f"{r['prediction']['gate']}/{r['prediction']['side']}/{r['prediction'].get('hold_bars',0)}" for r in chosen)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "market_csv": market_csv,
        "config": {
            "max_pairs_per_signal": max_pairs_per_signal,
            "min_utility_gap": min_utility_gap,
            "min_feature_count": min_feature_count,
            "lr": lr,
            "l2": l2,
            "epochs": epochs,
            "threshold": threshold,
            "features": len(fs.names),
            "train_pairs": len(pairs),
        },
        "train_ranking": _top1_oracle_metrics(train, train_scores),
        "eval_ranking": _top1_oracle_metrics(ev, eval_scores),
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"feature_space_fit_on_train_only": True, "eval_utility_not_used_for_fit_or_selection": True, "strict_backtest_uses_predicted_actions": True},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate pairwise economic ranking baseline")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-output", required=True)
    p.add_argument("--max-pairs-per-signal", type=int, default=10)
    p.add_argument("--min-utility-gap", type=float, default=0.001)
    p.add_argument("--min-feature-count", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_pairwise_baseline(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
