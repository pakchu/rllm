"""Leakage-safe online analyzer-state memory diagnostic.

Fixed symbolic buckets failed because train-selected state/action mappings did not
hold OOS.  This module tests the next structural idea: use a recency-aware memory
of similar past analyzer states.  At each decision, only examples whose future
path would already be realized are eligible for memory:

    memory_example.signal_pos + hold_bars <= current.signal_pos

The diagnostic still uses future path diagnostics for offline evaluation, but it
never uses a row's own target/action side to choose the prediction side.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.analyzer_state_edge_report import _opposite, _path_mae, _path_net, _stats_to_json, _trend_side, summarize_values
from training.decision_feature_learnability import load_jsonl, record_features
from training.edge_decay_analyzer_data import write_jsonl

ACTIONS = ("TREND", "FADE")


@dataclass(frozen=True)
class MemoryExample:
    signal_pos: int
    features: dict[str, str]
    returns: dict[str, float]
    maes: dict[str, float]


def parse_fields(raw: str) -> tuple[str, ...]:
    fields = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not fields:
        raise ValueError("at least one similarity field is required")
    return fields


def example_from_row(row: dict[str, Any], fields: tuple[str, ...]) -> MemoryExample | None:
    if _trend_side(row) == "NONE":
        return None
    returns: dict[str, float] = {}
    maes: dict[str, float] = {}
    for action in ACTIONS:
        ret = _path_net(row, action)
        mae = _path_mae(row, action)
        if ret is None or mae is None:
            return None
        returns[action] = float(ret)
        maes[action] = float(mae)
    feats = record_features(row)
    selected = {field: str(feats.get(field, "<missing>")) for field in fields}
    return MemoryExample(signal_pos=int(row.get("signal_pos", 0)), features=selected, returns=returns, maes=maes)


def _similarity(row_features: dict[str, str], memory_features: dict[str, str], fields: tuple[str, ...]) -> float:
    if not fields:
        return 0.0
    matches = sum(1 for field in fields if row_features.get(field) == memory_features.get(field))
    return matches / len(fields)


def choose_action(
    row: dict[str, Any],
    memory: list[MemoryExample],
    fields: tuple[str, ...],
    *,
    top_k: int,
    min_similarity: float,
    min_neighbors: int,
    min_mean_return: float,
    mae_penalty: float,
    recency_halflife_bars: float,
) -> tuple[str, dict[str, Any]]:
    if _trend_side(row) == "NONE":
        return "SKIP", {"reason": "no_trend_side"}
    feats_all = record_features(row)
    feats = {field: str(feats_all.get(field, "<missing>")) for field in fields}
    current_pos = int(row.get("signal_pos", 0))
    scored: list[tuple[float, MemoryExample]] = []
    for ex in memory:
        sim = _similarity(feats, ex.features, fields)
        if sim < float(min_similarity):
            continue
        age = max(0, current_pos - int(ex.signal_pos))
        if recency_halflife_bars > 0:
            recency = 0.5 ** (age / float(recency_halflife_bars))
        else:
            recency = 1.0
        weight = max(1e-9, sim * recency)
        scored.append((weight, ex))
    scored.sort(key=lambda x: x[0], reverse=True)
    neighbors = scored[: max(1, int(top_k))]
    action_reports: dict[str, Any] = {}
    best_action = "SKIP"
    best_score = float("-inf")
    for action in ACTIONS:
        weights = [w for w, _ in neighbors]
        if len(weights) < int(min_neighbors):
            action_reports[action] = {"neighbors": len(weights), "mean_return_pct": 0.0, "score": float("-inf")}
            continue
        denom = sum(weights)
        mean_ret = sum(w * ex.returns[action] for w, ex in neighbors) / denom if denom else 0.0
        mean_mae = sum(w * ex.maes[action] for w, ex in neighbors) / denom if denom else 0.0
        score = mean_ret - float(mae_penalty) * mean_mae
        action_reports[action] = {
            "neighbors": len(weights),
            "mean_return_pct": mean_ret * 100.0,
            "mean_mae_pct": mean_mae * 100.0,
            "score_pct": score * 100.0,
        }
        if mean_ret >= float(min_mean_return) and score > best_score:
            best_score = score
            best_action = action
    return best_action, {"neighbors_considered": len(neighbors), "actions": action_reports}


def _decision_side(row: dict[str, Any], action: str) -> tuple[str, str]:
    trend_side = _trend_side(row)
    if action == "TREND" and trend_side in {"LONG", "SHORT"}:
        return "TRADE_TREND", trend_side
    if action == "FADE" and trend_side in {"LONG", "SHORT"}:
        return "FADE_TREND", _opposite(trend_side)
    return "ABSTAIN", "NONE"


def prediction_record(row: dict[str, Any], action: str) -> dict[str, Any]:
    decision, side = _decision_side(row, action)
    pred = {
        "decision": decision,
        "action_side": side,
        "confidence": "LOW",
        "rationale_class": "ONLINE_STATE_MEMORY",
    }
    return {
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "prediction": json.dumps(pred, sort_keys=True, separators=(",", ":")),
        "target": row.get("target"),
        "task": "online_state_memory_policy",
    }


def run_online(
    split_rows: dict[str, list[dict[str, Any]]],
    fields: tuple[str, ...],
    *,
    hold_bars: int,
    top_k: int,
    min_similarity: float,
    min_neighbors: int,
    min_mean_return: float,
    mae_penalty: float,
    recency_halflife_bars: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    tagged: list[tuple[str, dict[str, Any]]] = []
    for split, rows in split_rows.items():
        tagged.extend((split, row) for row in rows)
    tagged.sort(key=lambda item: (int(item[1].get("signal_pos", 0)), str(item[1].get("date", ""))))

    pending: deque[MemoryExample] = deque()
    memory: list[MemoryExample] = []
    returns: dict[str, list[float]] = {split: [] for split in split_rows}
    maes: dict[str, list[float]] = {split: [] for split in split_rows}
    action_counts: dict[str, Counter[str]] = {split: Counter({"TREND": 0, "FADE": 0, "SKIP": 0}) for split in split_rows}
    predictions: dict[str, list[dict[str, Any]]] = {split: [] for split in split_rows}
    debug_counts: dict[str, Counter[str]] = {split: Counter() for split in split_rows}

    for split, row in tagged:
        current_pos = int(row.get("signal_pos", 0))
        while pending and pending[0].signal_pos + int(hold_bars) <= current_pos:
            memory.append(pending.popleft())
        action, dbg = choose_action(
            row,
            memory,
            fields,
            top_k=top_k,
            min_similarity=min_similarity,
            min_neighbors=min_neighbors,
            min_mean_return=min_mean_return,
            mae_penalty=mae_penalty,
            recency_halflife_bars=recency_halflife_bars,
        )
        debug_counts[split]["memory_size_sum"] += len(memory)
        debug_counts[split]["decisions"] += 1
        if action == "SKIP" and dbg.get("reason"):
            debug_counts[split][str(dbg["reason"])] += 1
        action_counts[split][action] += 1
        predictions[split].append(prediction_record(row, action))
        if action != "SKIP":
            ret = _path_net(row, action)
            mae = _path_mae(row, action)
            if ret is not None and mae is not None:
                returns[split].append(ret)
                maes[split].append(mae)
        ex = example_from_row(row, fields)
        if ex is not None:
            pending.append(ex)

    report: dict[str, Any] = {}
    for split, rows in split_rows.items():
        stats = summarize_values(returns[split], maes[split])
        decisions = max(1, debug_counts[split]["decisions"])
        report[split] = {
            "samples": len(rows),
            "trades": stats.n,
            "action_counts": dict(action_counts[split]),
            "trade_stats": _stats_to_json(stats),
            "avg_memory_size_before_decision": debug_counts[split]["memory_size_sum"] / decisions,
            "skip_reasons": {k: v for k, v in debug_counts[split].items() if k not in {"memory_size_sum", "decisions"}},
        }
    return report, predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe online analyzer-state memory report")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", default="")
    p.add_argument("--output", default="results/online_state_memory_report.json")
    p.add_argument("--prediction-output-dir", default="")
    p.add_argument(
        "--similarity-fields",
        default="regime,trend_alignment,location,volatility_level,risk_state,sequence_stats.wide_or_extreme,sequence_stats.rally_or_up,sequence_stats.drop_or_down",
    )
    p.add_argument("--hold-bars", type=int, default=432)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--min-similarity", type=float, default=0.625)
    p.add_argument("--min-neighbors", type=int, default=20)
    p.add_argument("--min-mean-return", type=float, default=0.001)
    p.add_argument("--mae-penalty", type=float, default=0.0)
    p.add_argument("--recency-halflife-bars", type=float, default=8640.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fields = parse_fields(args.similarity_fields)
    splits = {"train": load_jsonl(args.train_jsonl), "val": load_jsonl(args.val_jsonl)}
    if args.oos_jsonl:
        splits["oos"] = load_jsonl(args.oos_jsonl)
    split_report, predictions = run_online(
        splits,
        fields,
        hold_bars=int(args.hold_bars),
        top_k=int(args.top_k),
        min_similarity=float(args.min_similarity),
        min_neighbors=int(args.min_neighbors),
        min_mean_return=float(args.min_mean_return),
        mae_penalty=float(args.mae_penalty),
        recency_halflife_bars=float(args.recency_halflife_bars),
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"train_jsonl": args.train_jsonl, "val_jsonl": args.val_jsonl, "oos_jsonl": args.oos_jsonl},
        "similarity_fields": list(fields),
        "config": {
            "hold_bars": int(args.hold_bars),
            "top_k": int(args.top_k),
            "min_similarity": float(args.min_similarity),
            "min_neighbors": int(args.min_neighbors),
            "min_mean_return_pct": float(args.min_mean_return) * 100.0,
            "mae_penalty": float(args.mae_penalty),
            "recency_halflife_bars": float(args.recency_halflife_bars),
        },
        "splits": split_report,
        "leakage_guard": {
            "memory_update_requires_signal_pos_plus_hold_bars_before_current_pos": True,
            "prediction_side_source": "past source_edge_target.trend_side; never target.action_side",
            "future_path_used_only_after_maturity_for_offline_memory": True,
            "parameters_not_selected_on_val_or_oos": True,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.prediction_output_dir:
        root = Path(args.prediction_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for split, rows in predictions.items():
            write_jsonl(root / f"{split}_predictions.jsonl", rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
