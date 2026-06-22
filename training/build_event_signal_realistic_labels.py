"""Build signal-level realistic action labels from event candidates.

This keeps inputs causal: prompts/features contain only signal-time snapshots and state
buckets. Future rewards are used only to create supervised labels for offline
experiments.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABELS = ("NO_TRADE", "LONG", "SHORT")

@dataclass(frozen=True)
class Cfg:
    train_candidates: str
    eval_candidates: str
    train_output: str
    eval_output: str
    min_net_pct: float = 0.5
    min_utility: float = 0.2
    max_mae_pct: float = 6.0
    mae_free_pct: float = 2.0
    mae_penalty: float = 0.35
    feature_decimals: int = 5


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def group_by_signal(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["signal_pos"])].append(row)
    return [groups[k] for k in sorted(groups)]


def reward(row: dict[str, Any]) -> dict[str, float]:
    rw = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    return {
        "net_return_pct": float(rw.get("net_return_pct", 0.0) or 0.0),
        "utility": float(rw.get("utility", rw.get("net_return_pct", 0.0)) or 0.0),
        "mae_pct": float(rw.get("mae_pct", 0.0) or 0.0),
        "mfe_pct": float(rw.get("mfe_pct", 0.0) or 0.0),
    }


def side_row(group: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    for row in group:
        if str(row.get("side")) == side:
            return row
    return None


def side_score(row: dict[str, Any], cfg: Cfg) -> tuple[float, bool, dict[str, float]]:
    rw = reward(row)
    passes = (
        rw["net_return_pct"] >= cfg.min_net_pct
        and rw["utility"] >= cfg.min_utility
        and rw["mae_pct"] <= cfg.max_mae_pct
    )
    score = rw["net_return_pct"] - cfg.mae_penalty * max(0.0, rw["mae_pct"] - cfg.mae_free_pct)
    return score, passes, rw


def choose_label(group: list[dict[str, Any]], cfg: Cfg) -> tuple[str, dict[str, Any]]:
    candidates=[]
    reward_summary={}
    for side in ("LONG", "SHORT"):
        row=side_row(group, side)
        if row is None:
            continue
        score, passes, rw = side_score(row, cfg)
        reward_summary[side] = {"score": score, "passes": passes, **rw}
        if passes:
            candidates.append((score, side))
    if not candidates:
        return "NO_TRADE", {"reason":"no_side_meets_net_utility_mae", "side_rewards": reward_summary}
    candidates.sort(reverse=True)
    return candidates[0][1], {"reason":"best_side_meets_realistic_trade_filter", "side_rewards": reward_summary}


def compact_features(row: dict[str, Any], decimals: int) -> dict[str, float]:
    snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
    out={}
    for k,v in sorted(snap.items()):
        try:
            out[str(k)] = round(float(v or 0.0), decimals)
        except Exception:
            out[str(k)] = 0.0
    return out


def build_prompt(row: dict[str, Any], features: dict[str, float]) -> str:
    tokens = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    triggers = row.get("event_triggers", []) if isinstance(row.get("event_triggers"), list) else []
    state_bits = ", ".join(f"{k}={v}" for k, v in sorted(tokens.items()))
    feature_bits = ", ".join(f"{k}={v:+.5f}" for k, v in features.items())
    trigger_bits = ", ".join(map(str, triggers))
    return (
        "You are a BTCUSDT futures signal trader. Use only signal-time evidence. "
        "Choose exactly one action: NO_TRADE, LONG, or SHORT.\n"
        f"date: {row.get('date')}\n"
        f"event_triggers: {trigger_bits}\n"
        f"state_tokens: {state_bits}\n"
        f"numeric_features: {feature_bits}\n"
        "answer:"
    )


def convert(groups: list[list[dict[str, Any]]], cfg: Cfg) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows=[]
    labels=Counter()
    for group in groups:
        base=group[0]
        label, label_meta = choose_label(group, cfg)
        features = compact_features(base, cfg.feature_decimals)
        prompt = build_prompt(base, features)
        target = {"action": label, "schema_version": "event_signal_realistic_label_v1"}
        out = {
            "task": "event_signal_realistic_action",
            "date": base.get("date"),
            "signal_pos": int(base.get("signal_pos")),
            "event_triggers": base.get("event_triggers", []),
            "state_tokens": base.get("state_tokens", {}),
            "feature_snapshot": features,
            "prompt": prompt,
            "target": target,
            "messages": [
                {"role":"user", "content": prompt},
                {"role":"assistant", "content": json.dumps(target, sort_keys=True)},
            ],
            "label_meta": label_meta,
            "leakage_guard": {
                "features_signal_time_or_prior": True,
                "prompt_uses_future_reward": False,
                "target_uses_future_reward_for_training_only": True,
            },
        }
        rows.append(out)
        labels[label]+=1
    return rows, {"rows": len(rows), "label_counts": dict(labels)}


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)+"\n")


def run(cfg: Cfg) -> dict[str, Any]:
    train_rows, train_summary = convert(group_by_signal(load_jsonl(cfg.train_candidates)), cfg)
    eval_rows, eval_summary = convert(group_by_signal(load_jsonl(cfg.eval_candidates)), cfg)
    write_jsonl(train_rows, cfg.train_output)
    write_jsonl(eval_rows, cfg.eval_output)
    return {
        "config": cfg.__dict__,
        "train": {**train_summary, "output": cfg.train_output},
        "eval": {**eval_summary, "output": cfg.eval_output},
        "label_contract": "Future rewards create offline labels only; prompt/features are signal-time only.",
    }


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument("--train-candidates", required=True)
    p.add_argument("--eval-candidates", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--min-net-pct", type=float, default=Cfg.min_net_pct)
    p.add_argument("--min-utility", type=float, default=Cfg.min_utility)
    p.add_argument("--max-mae-pct", type=float, default=Cfg.max_mae_pct)
    p.add_argument("--mae-free-pct", type=float, default=Cfg.mae_free_pct)
    p.add_argument("--mae-penalty", type=float, default=Cfg.mae_penalty)
    p.add_argument("--feature-decimals", type=int, default=Cfg.feature_decimals)
    return Cfg(**vars(p.parse_args()))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
