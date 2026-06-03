"""Train-selected analyzer-state edge report for LLM/RL target redesign.

This diagnostic stops asking the LLM to predict future-path oracle classes
 directly.  Instead, it asks whether past-only analyzer states carry stable
 economic edge when actions are selected on train and then frozen for val/OOS.

Input rows are decision/edge analyzer records that include:
- a past-only analyzer summary in ``past_summary`` or ``prompt``;
- a past trend side in ``source_edge_target.trend_side``;
- future path diagnostics in ``path_diagnostics.long_same`` and
  ``path_diagnostics.long_opposite``.

The future path diagnostics are used only for offline train/test/eval analysis;
 exported prediction rows use the train-selected bucket policy plus past trend
 side, never target action_side.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from training.decision_feature_learnability import load_jsonl, parse_jsonish, record_features
from training.edge_decay_analyzer_data import write_jsonl

ACTIONS = ("TREND", "FADE")


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    return "NONE"


def _trend_side(row: dict[str, Any]) -> str:
    source = parse_jsonish(row.get("source_edge_target"))
    side = str(source.get("trend_side", "NONE"))
    return side if side in {"LONG", "SHORT"} else "NONE"


def _path_net(row: dict[str, Any], action: str) -> float | None:
    diag = row.get("path_diagnostics") or {}
    key = "long_same" if action == "TREND" else "long_opposite"
    block = diag.get(key) or {}
    if "net_return" not in block:
        return None
    return float(block.get("net_return") or 0.0)


def _path_mae(row: dict[str, Any], action: str) -> float | None:
    diag = row.get("path_diagnostics") or {}
    key = "long_same" if action == "TREND" else "long_opposite"
    block = diag.get(key) or {}
    if "mae" not in block:
        return None
    return float(block.get("mae") or 0.0)


def bucket_key(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    feats = record_features(row)
    parts = []
    for field in fields:
        parts.append(f"{field}={feats.get(field, '<missing>')}")
    return "|".join(parts)


@dataclass(frozen=True)
class ActionStats:
    n: int
    mean_return: float
    win_rate: float
    mean_mae: float
    ci95_low: float
    ci95_high: float


def summarize_values(returns: list[float], maes: list[float]) -> ActionStats:
    n = len(returns)
    mean = sum(returns) / n if n else 0.0
    wins = sum(1 for x in returns if x > 0.0)
    mean_mae = sum(maes) / len(maes) if maes else 0.0
    if n > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in returns) / (n - 1))
        se = std / math.sqrt(n)
    else:
        se = 0.0
    return ActionStats(
        n=n,
        mean_return=mean,
        win_rate=wins / n if n else 0.0,
        mean_mae=mean_mae,
        ci95_low=mean - 1.96 * se,
        ci95_high=mean + 1.96 * se,
    )


def _stats_to_json(stats: ActionStats) -> dict[str, Any]:
    return {
        "n": stats.n,
        "mean_return_pct": stats.mean_return * 100.0,
        "win_rate": stats.win_rate,
        "mean_mae_pct": stats.mean_mae * 100.0,
        "ci95_mean_return_pct": [stats.ci95_low * 100.0, stats.ci95_high * 100.0],
    }


def bucket_action_stats(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, dict[str, ActionStats]]:
    returns: dict[str, dict[str, list[float]]] = defaultdict(lambda: {a: [] for a in ACTIONS})
    maes: dict[str, dict[str, list[float]]] = defaultdict(lambda: {a: [] for a in ACTIONS})
    for row in rows:
        if _trend_side(row) == "NONE":
            continue
        key = bucket_key(row, fields)
        for action in ACTIONS:
            ret = _path_net(row, action)
            mae = _path_mae(row, action)
            if ret is None or mae is None:
                continue
            returns[key][action].append(ret)
            maes[key][action].append(mae)
    out: dict[str, dict[str, ActionStats]] = {}
    for key, action_returns in returns.items():
        out[key] = {action: summarize_values(action_returns[action], maes[key][action]) for action in ACTIONS}
    return out


def select_bucket_policy(
    train_rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    min_train_count: int,
    min_mean_return: float,
    require_positive_ci: bool,
    max_buckets: int,
) -> dict[str, str]:
    stats = bucket_action_stats(train_rows, fields)
    candidates: list[tuple[float, str, str]] = []
    for key, by_action in stats.items():
        for action, action_stats in by_action.items():
            if action_stats.n < int(min_train_count):
                continue
            if action_stats.mean_return < float(min_mean_return):
                continue
            if require_positive_ci and action_stats.ci95_low <= 0.0:
                continue
            score = action_stats.mean_return / max(action_stats.mean_mae, 1e-6)
            candidates.append((score, key, action))
    candidates.sort(reverse=True)
    policy: dict[str, str] = {}
    for _, key, action in candidates:
        if key in policy:
            continue
        policy[key] = action
        if max_buckets and len(policy) >= int(max_buckets):
            break
    return policy


def evaluate_policy(rows: list[dict[str, Any]], fields: tuple[str, ...], policy: dict[str, str]) -> dict[str, Any]:
    returns: list[float] = []
    maes: list[float] = []
    action_counts: Counter[str] = Counter({"TREND": 0, "FADE": 0, "SKIP": 0})
    matched_buckets: Counter[str] = Counter()
    for row in rows:
        action = policy.get(bucket_key(row, fields), "SKIP")
        action_counts[action] += 1
        if action == "SKIP":
            continue
        ret = _path_net(row, action)
        mae = _path_mae(row, action)
        if ret is None or mae is None or _trend_side(row) == "NONE":
            continue
        returns.append(ret)
        maes.append(mae)
        matched_buckets[bucket_key(row, fields)] += 1
    stats = summarize_values(returns, maes)
    return {
        "samples": len(rows),
        "trades": stats.n,
        "action_counts": dict(action_counts),
        "trade_stats": _stats_to_json(stats),
        "matched_bucket_count": len(matched_buckets),
        "top_matched_buckets": dict(matched_buckets.most_common(20)),
    }


def prediction_record(row: dict[str, Any], fields: tuple[str, ...], policy: dict[str, str]) -> dict[str, Any]:
    action = policy.get(bucket_key(row, fields), "SKIP")
    trend_side = _trend_side(row)
    if action == "TREND" and trend_side in {"LONG", "SHORT"}:
        decision = "TRADE_TREND"
        side = trend_side
    elif action == "FADE" and trend_side in {"LONG", "SHORT"}:
        decision = "FADE_TREND"
        side = _opposite(trend_side)
    else:
        decision = "ABSTAIN"
        side = "NONE"
    pred = {
        "decision": decision,
        "action_side": side,
        "confidence": "LOW",
        "rationale_class": "TRAIN_SELECTED_STATE_EDGE",
    }
    return {
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "prediction": json.dumps(pred, sort_keys=True, separators=(",", ":")),
        "target": row.get("target"),
        "task": "analyzer_state_edge_policy",
    }


def policy_details(train_rows: list[dict[str, Any]], fields: tuple[str, ...], policy: dict[str, str]) -> list[dict[str, Any]]:
    stats = bucket_action_stats(train_rows, fields)
    details = []
    for key, action in policy.items():
        details.append({"bucket": key, "action": action, "train_stats": _stats_to_json(stats[key][action])})
    details.sort(key=lambda x: x["train_stats"]["mean_return_pct"], reverse=True)
    return details


def parse_fields(raw: str) -> tuple[str, ...]:
    fields = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not fields:
        raise ValueError("at least one bucket field is required")
    return fields


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-selected past-state edge stability report")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--oos-jsonl", default="")
    p.add_argument("--output", default="results/analyzer_state_edge_report.json")
    p.add_argument("--prediction-output-dir", default="")
    p.add_argument(
        "--bucket-fields",
        default="regime,trend_alignment,location,volatility_level,risk_state,sequence_stats.wide_or_extreme",
    )
    p.add_argument("--min-train-count", type=int, default=20)
    p.add_argument("--min-mean-return", type=float, default=0.001)
    p.add_argument("--require-positive-ci", action="store_true")
    p.add_argument("--max-buckets", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fields = parse_fields(args.bucket_fields)
    splits = {"train": load_jsonl(args.train_jsonl), "val": load_jsonl(args.val_jsonl)}
    if args.oos_jsonl:
        splits["oos"] = load_jsonl(args.oos_jsonl)
    policy = select_bucket_policy(
        splits["train"],
        fields,
        min_train_count=int(args.min_train_count),
        min_mean_return=float(args.min_mean_return),
        require_positive_ci=bool(args.require_positive_ci),
        max_buckets=int(args.max_buckets),
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"train_jsonl": args.train_jsonl, "val_jsonl": args.val_jsonl, "oos_jsonl": args.oos_jsonl},
        "bucket_fields": list(fields),
        "selection": {
            "selected_on": "train_only",
            "min_train_count": int(args.min_train_count),
            "min_mean_return_pct": float(args.min_mean_return) * 100.0,
            "require_positive_ci": bool(args.require_positive_ci),
            "max_buckets": int(args.max_buckets),
            "selected_buckets": len(policy),
        },
        "splits": {name: evaluate_policy(rows, fields, policy) for name, rows in splits.items()},
        "policy": policy_details(splits["train"], fields, policy),
        "leakage_guard": {
            "bucket_features_are_past_only": True,
            "policy_selected_on_train_only": True,
            "val_oos_do_not_select_parameters": True,
            "prediction_side_source": "past source_edge_target.trend_side; never target.action_side",
            "future_path_used_for_offline_report_only": True,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if args.prediction_output_dir:
        root = Path(args.prediction_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            write_jsonl(root / f"{name}_predictions.jsonl", [prediction_record(row, fields, policy) for row in rows])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
