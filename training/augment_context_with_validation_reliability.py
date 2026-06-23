"""Augment LLM context rows with causal prior-validation reliability tokens.

The source rolling summary contains, for each target month, a selected score that
was computed from data before that month.  This script converts that score into
coarse text buckets and injects them into both `state_tokens` and the prompt.
No target-month returns are read here.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationReliabilityAugmentCfg:
    input_jsonl: str
    rolling_summary_json: str
    output_jsonl: str
    summary_output: str = ""
    reliable_threshold: float = 0.5
    inverse_threshold: float = -500.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def _month_scores(summary: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in summary.get("months", []):
        month = str(m.get("month"))
        try:
            out[month] = float((m.get("selected") or {}).get("score", float("-inf")))
        except Exception:
            out[month] = float("-inf")
    return out


def reliability_bucket(score: float | None, *, reliable_threshold: float = 0.5, inverse_threshold: float = -500.0) -> tuple[str, str]:
    """Return `(side_map_reliability, prior_validation_health)` text tokens."""
    if score is None:
        return "unknown_pre_roll", "unknown"
    if float(score) >= float(reliable_threshold):
        return "reliable_normal", "positive"
    if float(score) < float(inverse_threshold):
        return "inverse_candidate", "severe_decay"
    return "weak_or_decaying", "nonpositive"


def _inject_prompt_tokens(prompt: str, additions: dict[str, str]) -> str:
    lines = str(prompt).splitlines()
    rendered = [f"- {k}: {v}" for k, v in sorted(additions.items())]
    for i, line in enumerate(lines):
        if line.startswith("Policy intent:"):
            return "\n".join(lines[:i] + rendered + lines[i:])
    if lines and lines[-1].strip():
        return "\n".join(lines + rendered)
    return "\n".join(lines + rendered)


def augment(cfg: ValidationReliabilityAugmentCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    summary = json.loads(Path(cfg.rolling_summary_json).read_text())
    scores = _month_scores(summary)
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    month_counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        month = _month(row)
        score = scores.get(month)
        side_map, health = reliability_bucket(score, reliable_threshold=cfg.reliable_threshold, inverse_threshold=cfg.inverse_threshold)
        additions = {
            "prior_validation_health": health,
            "side_map_reliability": side_map,
        }
        nr = dict(row)
        toks = dict(nr.get("state_tokens", {}) if isinstance(nr.get("state_tokens"), dict) else {})
        toks.update(additions)
        nr["state_tokens"] = toks
        if isinstance(nr.get("prompt"), str):
            nr["prompt"] = _inject_prompt_tokens(nr["prompt"], additions)
        nr["validation_reliability_score_available"] = score is not None
        nr["validation_reliability_leakage_guard"] = {
            "uses_target_month_outcomes": False,
            "uses_prior_rolling_summary_selected_score": True,
            "score_bucketed_not_raw_numeric_prompted": True,
        }
        counts[side_map] = counts.get(side_map, 0) + 1
        m = month_counts.setdefault(month, {"rows": 0, "score": score, "side_map_reliability": side_map, "prior_validation_health": health})
        m["rows"] = int(m["rows"]) + 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(out),
        "counts": counts,
        "months": month_counts,
        "leakage_guard": {
            "input_context_prompts_remain_causal": True,
            "rolling_scores_are_assumed_prior_to_target_month": True,
            "raw_numeric_scores_not_inserted_into_prompt": True,
        },
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add prior-validation reliability text tokens to LLM context rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--rolling-summary-json", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--reliable-threshold", type=float, default=ValidationReliabilityAugmentCfg.reliable_threshold)
    p.add_argument("--inverse-threshold", type=float, default=ValidationReliabilityAugmentCfg.inverse_threshold)
    return p.parse_args()


def main() -> None:
    report = augment(ValidationReliabilityAugmentCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output_jsonl"], "rows": report["rows"], "counts": report["counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
