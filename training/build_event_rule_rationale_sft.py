"""Build Gemma-style rule-rationale SFT rows from event candidates.

The prompt contains only signal-time features. The completion asks the model to
emit a compact analyzer JSON (feature-derived, no future) plus a final trading
decision label (supervised by future reward for training only).
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
class RuleRationaleCfg:
    train_jsonl: str
    eval_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    include_abstain_fraction: float = 0.35
    seed: int = 13


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _side_sign(side: str) -> float:
    return 1.0 if side == "LONG" else -1.0


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float((row.get("feature_snapshot", {}) or {}).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _bucket(v: float, lo: float, hi: float) -> str:
    if v <= lo:
        return "low"
    if v >= hi:
        return "high"
    return "neutral"


def _analyzer(row: dict[str, Any]) -> dict[str, Any]:
    side = str(row.get("side"))
    sign = _side_sign(side)
    signed_trend96 = _f(row, "trend_96") * sign
    signed_4h = _f(row, "htf_4h_return_4") * sign
    signed_range = _f(row, "range_pos") * sign
    signed_rsi = _f(row, "rsi_norm") * sign
    signed_bb = _f(row, "bb_z") * sign
    dxy = _f(row, "dxy_zscore")
    kimchi = _f(row, "kimchi_premium_zscore")
    usdkrw = _f(row, "usdkrw_zscore")
    mr_score = -0.50 * signed_trend96 - 0.50 * signed_4h - 0.20 * signed_range - 0.15 * signed_rsi - 0.10 * signed_bb
    pressure_conflict = abs(dxy) > 2.0 or abs(usdkrw) > 2.0 or abs(kimchi) > 2.0
    return {
        "candidate_side": side,
        "setup_family": "mean_reversion_fade" if mr_score > 0 else "trend_continuation_or_none",
        "trend_alignment": _bucket(-signed_trend96, -0.002, 0.002),
        "htf_4h_alignment": _bucket(-signed_4h, -0.002, 0.002),
        "range_location_for_side": _bucket(-signed_range, -0.25, 0.25),
        "oscillator_for_side": _bucket(-signed_rsi, -0.15, 0.15),
        "external_pressure": "conflict" if pressure_conflict else "normal",
        "risk_path_hint": "avoid_if_external_conflict" if pressure_conflict else "normal_path_risk",
        "rule_score": round(float(mr_score), 6),
    }


def _prompt(row: dict[str, Any]) -> str:
    side = str(row.get("side"))
    snap = row.get("feature_snapshot", {}) if isinstance(row.get("feature_snapshot"), dict) else {}
    toks = row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}
    keep = [
        "trend_96", "htf_4h_return_4", "range_pos", "rsi_norm", "bb_z", "return_zscore_48",
        "range_vol", "window_drawdown", "taker_buy_ratio", "dxy_zscore", "dxy_momentum",
        "kimchi_premium_zscore", "kimchi_premium_change", "usdkrw_zscore", "usdkrw_momentum",
    ]
    lines = [
        "You are a BTCUSDT futures analyzer-trader.",
        "Use only signal-time evidence. Do not infer or mention future reward.",
        "First classify the setup, then choose one decision: TAKE_FULL, TAKE_SMALL, ABSTAIN.",
        f"date: {row.get('date')}",
        f"candidate_side: {side}",
        f"event_triggers: {', '.join(map(str, row.get('event_triggers', []))) if row.get('event_triggers') else 'none'}",
        "state_tokens:",
    ]
    for k in sorted(toks):
        lines.append(f"- {k}: {toks[k]}")
    lines.append("numeric_features:")
    for k in keep:
        if k in snap:
            lines.append(f"- {k}: {float(snap.get(k, 0.0) or 0.0):+.6f}")
    lines.append('Return JSON with keys: analyzer, decision.')
    return "\n".join(lines)


def _completion(row: dict[str, Any]) -> str:
    return json.dumps({"analyzer": _analyzer(row), "decision": row.get("target", {}).get("decision", "ABSTAIN")}, ensure_ascii=False, sort_keys=True)


def _convert(rows: list[dict[str, Any]], abstain_fraction: float, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        decision = str(row.get("target", {}).get("decision"))
        if decision == "ABSTAIN" and float(rng.random()) > abstain_fraction:
            continue
        prompt = _prompt(row)
        target = _completion(row)
        out.append({
            "task": "event_rule_rationale_sft",
            "prompt": prompt,
            "target": target,
            "messages": [
                {"role": "system", "content": "You emit compact no-leak trading analysis JSON for BTCUSDT futures."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "metadata": {
                "date": row.get("date"),
                "side": row.get("side"),
                "decision": decision,
                "rule_analyzer": _analyzer(row),
                "leakage_guard": "prompt/analyzer feature-derived only; decision label uses future reward for supervised training only",
            },
        })
    return out


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = Counter(r["metadata"]["decision"] for r in rows)
    sides = Counter(r["metadata"]["side"] for r in rows)
    setups = Counter(r["metadata"]["rule_analyzer"]["setup_family"] for r in rows)
    chars = [len(str(r.get("prompt", ""))) + len(str(r.get("target", ""))) for r in rows]
    return {"rows": len(rows), "decisions": dict(decisions), "sides": dict(sides), "setups": dict(setups), "chars": {"min": min(chars) if chars else 0, "max": max(chars) if chars else 0, "mean": sum(chars)/max(1,len(chars))}}


def run(cfg: RuleRationaleCfg) -> dict[str, Any]:
    train = _convert(_load(cfg.train_jsonl), cfg.include_abstain_fraction, cfg.seed)
    eval_rows = _convert(_load(cfg.eval_jsonl), cfg.include_abstain_fraction, cfg.seed + 1)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report = {"config": cfg.__dict__, "train": _summary(train), "eval": _summary(eval_rows), "contract": "Gemma SFT rows with feature-derived analyzer JSON and reward-derived decision label"}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event rule-rationale SFT rows")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--include-abstain-fraction", type=float, default=RuleRationaleCfg.include_abstain_fraction)
    p.add_argument("--seed", type=int, default=RuleRationaleCfg.seed)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RuleRationaleCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
