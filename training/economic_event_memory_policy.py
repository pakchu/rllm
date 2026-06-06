"""Event-conditioned train-memory action policy.

After a live-safe opportunity detector fires, choose the most common future-best
action among train positive events with matching past-only regime keys.  This is
live-safe for validation because eval labels are not used for action selection.
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
from training.economic_opportunity_baseline import best_rows_by_signal, fit_logistic, signal_feature_row, _metrics
from training.economic_value_baseline import FeatureSpace, _summary_obj, load_jsonl, write_jsonl


def _action_obj(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(str(text))
    except Exception:
        obj = {}
    return {"gate": str(obj.get("gate", "NO_TRADE")), "side": str(obj.get("side", "NONE")), "hold_bars": int(obj.get("hold_bars", 0) or 0)}


def _action_text(action: dict[str, Any]) -> str:
    return json.dumps({"gate": action["gate"], "side": action["side"], "hold_bars": int(action.get("hold_bars", 0) or 0)}, sort_keys=True, separators=(",", ":"))


def regime_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    s = _summary_obj(str(row.get("prompt", "")))
    sym = s.get("symbolic_features", {}) if isinstance(s.get("symbolic_features"), dict) else {}
    vals = []
    for f in fields:
        vals.append(str(s.get(f, sym.get(f, "NA"))))
    return tuple(vals)


def build_action_memory(train_best: list[dict[str, Any]], *, utility_threshold: float, fields: tuple[str, ...], min_bucket: int = 5) -> tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in train_best:
        if float(row.get("utility", 0.0)) < float(utility_threshold):
            continue
        action = _action_text(_action_obj(str(row.get("action", "{}"))))
        buckets[regime_key(row, fields)][action] += 1
        global_counts[action] += 1
    fallback = _action_obj(global_counts.most_common(1)[0][0]) if global_counts else {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    memory: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, counts in buckets.items():
        if sum(counts.values()) >= int(min_bucket):
            memory[key] = _action_obj(counts.most_common(1)[0][0])
    return memory, fallback


def run_event_memory_policy(
    *,
    train_jsonl: str,
    eval_jsonl: str,
    market_csv: str,
    output: str,
    predictions_output: str,
    utility_threshold: float = 0.003,
    event_threshold: float = 0.7,
    fields: str = "regime,trend_alignment,location,momentum",
    min_bucket: int = 5,
    lr: float = 0.2,
    l2: float = 1.0,
    epochs: int = 500,
) -> dict[str, Any]:
    train_best = best_rows_by_signal(load_jsonl(train_jsonl))
    eval_best = best_rows_by_signal(load_jsonl(eval_jsonl))
    field_tuple = tuple(x.strip() for x in fields.split(",") if x.strip())

    train_signal = [signal_feature_row(r) for r in train_best]
    eval_signal = [signal_feature_row(r) for r in eval_best]
    fs = FeatureSpace(min_count=2)
    fs.fit(train_signal)
    x_train = fs.matrix(train_signal, fit_scale=True)
    x_eval = fs.matrix(eval_signal)
    y_train = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in train_best], dtype=np.float64)
    y_eval = np.array([float(r.get("utility", 0.0)) >= utility_threshold for r in eval_best], dtype=np.float64)
    w = fit_logistic(x_train, y_train, lr=lr, l2=l2, epochs=epochs)
    train_scores = 1.0 / (1.0 + np.exp(-np.clip(x_train @ w, -40, 40)))
    eval_scores = 1.0 / (1.0 + np.exp(-np.clip(x_eval @ w, -40, 40)))

    memory, fallback = build_action_memory(train_best, utility_threshold=utility_threshold, fields=field_tuple, min_bucket=min_bucket)
    preds = []
    for row, score in zip(eval_best, eval_scores):
        if float(score) < float(event_threshold):
            action = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
        else:
            action = memory.get(regime_key(row, field_tuple), fallback)
        preds.append({"date": row.get("date"), "signal_pos": row.get("signal_pos"), "prediction": action, "event_score": float(score), "memory_key": list(regime_key(row, field_tuple)), "oracle_utility": float(row.get("utility", 0.0))})
    write_jsonl(predictions_output, preds)
    bt_path = str(Path(output).with_suffix(".strict_backtest.json"))
    bt = run_economic_action_backtest(predictions_jsonl=predictions_output, market_csv=market_csv, output=bt_path)
    counts = Counter(f"{p['prediction']['gate']}/{p['prediction']['side']}/{p['prediction'].get('hold_bars',0)}" for p in preds)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": {"utility_threshold": utility_threshold, "event_threshold": event_threshold, "fields": list(field_tuple), "min_bucket": min_bucket, "memory_buckets": len(memory), "fallback": fallback},
        "train_event_metrics": _metrics(y_train, train_scores, event_threshold),
        "eval_event_metrics": _metrics(y_eval, eval_scores, event_threshold),
        "prediction_counts": dict(sorted(counts.items())),
        "strict_backtest_path": bt_path,
        "strict_backtest": bt["backtest"],
        "leakage_guard": {"event_detector_fit_on_train_only": True, "action_memory_fit_on_train_positive_events_only": True, "eval_utility_not_used_for_selection": True},
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
    p.add_argument("--fields", default="regime,trend_alignment,location,momentum")
    p.add_argument("--min-bucket", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=500)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_event_memory_policy(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
