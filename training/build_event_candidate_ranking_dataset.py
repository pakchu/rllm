"""Build TAKE/ABSTAIN candidate-ranking rows from event-trigger decisions.

Each event timestamp becomes side-specific candidates.  The prompt contains only
signal-time state and the candidate side.  The target decision is derived from
that side's future reward for supervised/RL training, never as input.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventCandidateRankingCfg:
    input_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    sides: str = "LONG,SHORT"
    full_net_return_pct: float = 1.2
    small_net_return_pct: float = 0.25
    max_full_mae_pct: float = 5.0
    max_small_mae_pct: float = 7.5
    min_full_utility_pct: float = 0.5
    min_small_utility_pct: float = 0.0


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _target(side_reward: dict[str, Any], cfg: EventCandidateRankingCfg) -> dict[str, str]:
    net = float(side_reward.get("net_return_pct", 0.0))
    mae = float(side_reward.get("mae_pct", 999.0))
    utility = float(side_reward.get("utility", net))
    if net >= cfg.full_net_return_pct and mae <= cfg.max_full_mae_pct and utility >= cfg.min_full_utility_pct:
        return {"decision": "TAKE_FULL", "risk_reason": "reward_strong_after_path_risk"}
    if net >= cfg.small_net_return_pct and mae <= cfg.max_small_mae_pct and utility >= cfg.min_small_utility_pct:
        return {"decision": "TAKE_SMALL", "risk_reason": "reward_positive_but_thin"}
    return {"decision": "ABSTAIN", "risk_reason": "reward_not_worth_path_risk"}


def _prompt(row: dict[str, Any], side: str) -> str:
    tokens = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
    triggers = row.get("event_triggers", []) if isinstance(row.get("event_triggers"), list) else []
    lines = [
        "Task: decide position size for a BTCUSDT futures candidate.",
        "Use only signal-time state below. Do not infer future reward.",
        "Return exactly one decision: TAKE_FULL, TAKE_SMALL, or ABSTAIN.",
        f"Candidate side: {side}",
        f"Date: {row.get('date')}",
        f"Hold bars: {int(row.get('candidate', {}).get('hold_bars', 288))}",
        "Event triggers: " + (", ".join(map(str, triggers)) if triggers else "none"),
        "State buckets:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    # Compact numeric evidence: enough for LLM/rule probes without bloating rows.
    keep = ["trend_24", "trend_96", "range_pos", "rsi_norm", "mfi_norm", "range_vol", "window_drawdown", "volume_zscore", "taker_imbalance", "dxy_momentum", "kimchi_premium_zscore", "usdkrw_momentum"]
    if snap:
        lines.append("Numeric evidence:")
        for k in keep:
            if k in snap:
                try:
                    lines.append(f"- {k}: {float(snap[k]):+.4f}")
                except Exception:
                    pass
    return "\n".join(lines)


def _candidate_row(row: dict[str, Any], side: str, cfg: EventCandidateRankingCfg) -> dict[str, Any] | None:
    reward = row.get("reward_audit", {}).get(side) if isinstance(row.get("reward_audit"), dict) else None
    if not isinstance(reward, dict):
        return None
    return {
        "task": "event_candidate_ranking",
        "split": row.get("split"),
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "side": side,
        "prompt": _prompt(row, side),
        "target": _target(reward, cfg),
        "reward": reward,
        "candidate": {"hold_bars": int(row.get("candidate", {}).get("hold_bars", 288)), "side": side},
        "state_tokens": row.get("state_tokens", {}),
        "feature_snapshot": row.get("feature_snapshot", {}),
        "event_triggers": row.get("event_triggers", []),
        "leakage_guard": {"prompt_uses_future_reward": False, "target_uses_future_reward_for_training_only": True, "features_signal_time_or_prior": True},
    }


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([float(r["reward"].get("net_return_pct", 0.0)) for r in rows], dtype=float) if rows else np.asarray([])
    dec = Counter(str(r["target"].get("decision")) for r in rows)
    side = Counter(str(r["side"]) for r in rows)
    lens = [len(str(r["prompt"])) for r in rows]
    return {
        "rows": len(rows),
        "decisions": dict(sorted(dec.items())),
        "sides": dict(sorted(side.items())),
        "net_return_pct": {"mean": float(np.mean(rewards)) if len(rewards) else 0.0, "std": float(np.std(rewards)) if len(rewards) else 0.0, "positive_rate": float(np.mean(rewards > 0.0)) if len(rewards) else 0.0},
        "prompt_chars": {"min": min(lens) if lens else 0, "max": max(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens))},
    }


def run(cfg: EventCandidateRankingCfg) -> dict[str, Any]:
    sides = [s.strip().upper() for s in cfg.sides.split(",") if s.strip()]
    out: list[dict[str, Any]] = []
    for row in _load(cfg.input_jsonl):
        for side in sides:
            cand = _candidate_row(row, side, cfg)
            if cand:
                out.append(cand)
    train = [r for r in out if r.get("split") == "train"]
    eval_rows = [r for r in out if r.get("split") == "eval"]
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report = {"config": cfg.__dict__, "outputs": {"train": cfg.train_output, "eval": cfg.eval_output}, "train": _summary(train), "eval": _summary(eval_rows), "contract": "side-specific candidate ranking; prompt is signal-time only; target/reward are label-only"}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event candidate-ranking dataset")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--sides", default=EventCandidateRankingCfg.sides)
    p.add_argument("--full-net-return-pct", type=float, default=EventCandidateRankingCfg.full_net_return_pct)
    p.add_argument("--small-net-return-pct", type=float, default=EventCandidateRankingCfg.small_net_return_pct)
    p.add_argument("--max-full-mae-pct", type=float, default=EventCandidateRankingCfg.max_full_mae_pct)
    p.add_argument("--max-small-mae-pct", type=float, default=EventCandidateRankingCfg.max_small_mae_pct)
    p.add_argument("--min-full-utility-pct", type=float, default=EventCandidateRankingCfg.min_full_utility_pct)
    p.add_argument("--min-small-utility-pct", type=float, default=EventCandidateRankingCfg.min_small_utility_pct)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateRankingCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
