"""Convert causal focus-label scores into single-policy prediction rows."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.single_policy_sft_data import exit_profile_for_hold

TARGET_PATH = "CLEAN_WIN_PATH"
TARGET_UTILITY = "UTILITY_HIGH"
PATH_OPTIONS = ["CLEAN_WIN_PATH", "HIGH_ADVERSE_PATH", "FAILED_FOLLOW_THROUGH", "LOW_EDGE_PATH", "MIXED_PATH"]
UTILITY_OPTIONS = ["UTILITY_LOW", "UTILITY_MID", "UTILITY_HIGH"]


@dataclass(frozen=True)
class FocusScorePolicyCfg:
    focus_predictions_jsonl: str
    output_jsonl: str
    summary_json: str
    min_clean_prob: float = 0.0
    min_high_prob: float = 0.0
    min_clean_margin: float = 0.0
    min_high_margin: float = 0.0


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _softmax(scores: dict[str, float], options: list[str]) -> dict[str, float]:
    valid = {opt: float(scores[opt]) for opt in options if opt in scores}
    if not valid:
        return {}
    max_score = max(valid.values())
    exps = {k: math.exp(v - max_score) for k, v in valid.items()}
    denom = sum(exps.values()) or 1.0
    return {k: v / denom for k, v in exps.items()}


def _margin(scores: dict[str, float], target: str) -> float:
    if target not in scores or len(scores) < 2:
        return float("-inf")
    other = max(float(v) for k, v in scores.items() if k != target)
    return float(scores[target]) - other


def _best(probs: dict[str, float]) -> str:
    if not probs:
        return ""
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _policy_for_row(row: dict[str, Any], cfg: FocusScorePolicyCfg) -> tuple[dict[str, str], dict[str, Any]]:
    scores = row.get("focus_scores") or row.get("scores") or {}
    path_scores = {k: float(v) for k, v in dict(scores.get("path_shape") or {}).items()}
    utility_scores = {k: float(v) for k, v in dict(scores.get("utility_bucket") or {}).items()}
    path_probs = _softmax(path_scores, PATH_OPTIONS)
    utility_probs = _softmax(utility_scores, UTILITY_OPTIONS)
    path_pred = _best(path_probs)
    utility_pred = _best(utility_probs)
    clean_prob = float(path_probs.get(TARGET_PATH, 0.0))
    high_prob = float(utility_probs.get(TARGET_UTILITY, 0.0))
    clean_margin = _margin(path_scores, TARGET_PATH)
    high_margin = _margin(utility_scores, TARGET_UTILITY)

    cand = dict(row.get("candidate") or {})
    side = str(cand.get("side", "")).upper()
    horizon = int(cand.get("horizon", 288) or 288)
    trade = (
        path_pred == TARGET_PATH
        and utility_pred == TARGET_UTILITY
        and clean_prob >= float(cfg.min_clean_prob)
        and high_prob >= float(cfg.min_high_prob)
        and clean_margin >= float(cfg.min_clean_margin)
        and high_margin >= float(cfg.min_high_margin)
        and side in {"LONG", "SHORT"}
    )
    diagnostic = {
        "path_prediction": path_pred,
        "utility_prediction": utility_pred,
        "clean_prob": clean_prob,
        "high_prob": high_prob,
        "clean_margin": clean_margin if math.isfinite(clean_margin) else None,
        "high_margin": high_margin if math.isfinite(high_margin) else None,
        "has_scores": bool(path_scores and utility_scores),
    }
    if not trade:
        return (
            {
                "regime": "RANGE",
                "edge_quality": "NONE",
                "risk": "LOW",
                "action": "NO_TRADE",
                "exit_profile": "AVOID",
                "confidence": "LOW",
            },
            diagnostic,
        )
    return (
        {
            "regime": "TREND_UP" if side == "LONG" else "TREND_DOWN",
            "edge_quality": "STRONG",
            "risk": "LOW",
            "action": side,
            "exit_profile": exit_profile_for_hold(horizon),
            "confidence": "HIGH",
        },
        diagnostic,
    )


def run(cfg: FocusScorePolicyCfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.focus_predictions_jsonl)
    out_rows = []
    actions = Counter()
    diagnostics = []
    for row in rows:
        policy, diag = _policy_for_row(row, cfg)
        actions[policy["action"]] += 1
        diagnostics.append(diag)
        out_rows.append(
            {
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "candidate": row.get("candidate") or {},
                "policy_prediction": policy,
                "focus_score_policy": diag,
                # Audit-only fields; downstream action selection must not read these.
                "focus_prediction": row.get("focus_prediction") or row.get("prediction") or {},
                "focus_target": row.get("focus_target") or row.get("target") or {},
                "target_audit": row.get("target_audit") or {},
            }
        )

    scored = sum(1 for d in diagnostics if d["has_scores"])
    traded = len(rows) - actions.get("NO_TRADE", 0)
    summary = {
        "config": asdict(cfg),
        "rows": len(rows),
        "rows_with_scores": scored,
        "actions": dict(actions),
        "trade_rate": traded / max(1, len(rows)),
    }
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out_rows) + "\n")
    Path(cfg.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--focus-predictions-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-json", required=True)
    p.add_argument("--min-clean-prob", type=float, default=FocusScorePolicyCfg.min_clean_prob)
    p.add_argument("--min-high-prob", type=float, default=FocusScorePolicyCfg.min_high_prob)
    p.add_argument("--min-clean-margin", type=float, default=FocusScorePolicyCfg.min_clean_margin)
    p.add_argument("--min-high-margin", type=float, default=FocusScorePolicyCfg.min_high_margin)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(FocusScorePolicyCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
