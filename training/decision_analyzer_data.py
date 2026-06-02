"""Build decision-critical analyzer records from edge-decay diagnostics.

The previous edge-decay target asked the LLM to imitate five derived labels.  The
strict router results showed that this is too brittle: trend side was learned,
but the economically important route was not.  This module compresses the same
future-path teacher into the smallest trading decision target:

- TRADE_TREND: trade with the detected past trend.
- FADE_TREND: trade against the detected past trend.
- ABSTAIN: do not open a position.

Prompts remain past-only.  Targets still use future path diagnostics and are
therefore teacher labels, not deployable signals until model predictions replace
targets in strict backtests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.edge_decay_analyzer_data import write_jsonl
from training.edge_decay_router_backtest import _opposite, load_jsonl

VALID_DECISIONS = {"TRADE_TREND", "FADE_TREND", "ABSTAIN"}
VALID_ACTION_SIDES = {"LONG", "SHORT", "NONE"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}


@dataclass(frozen=True)
class DecisionAnalyzerConfig:
    min_actionable_edge: float = 0.001
    medium_margin: float = 0.004
    high_margin: float = 0.01


def _parse_target(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            return dict(json.loads(raw))
        except Exception:
            return {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _confidence_from_margin(margin: float, cfg: DecisionAnalyzerConfig) -> str:
    margin = abs(float(margin))
    if margin >= float(cfg.high_margin):
        return "HIGH"
    if margin >= float(cfg.medium_margin):
        return "MEDIUM"
    return "LOW"


def _rationale_for(edge_label: str, transition_label: str, decision: str) -> str:
    if decision == "TRADE_TREND":
        return "EDGE_PERSIST_CONTINUATION"
    if decision == "FADE_TREND":
        return "REVERSAL_CAPTURE"
    if edge_label == "ADVERSE_STRESS":
        return "ADVERSE_STRESS_SKIP"
    if edge_label == "NO_CLEAR_TREND":
        return "RANGE_UNKNOWN_SKIP"
    if edge_label == "NO_EDGE":
        return "NO_EDGE_SKIP"
    if transition_label == "CHOP_OR_DECAY":
        return "CHOP_DECAY_SKIP"
    return "LOW_CONFIDENCE_SKIP"


def derive_decision_target(edge_record: dict[str, Any], cfg: DecisionAnalyzerConfig) -> dict[str, Any]:
    """Compress an edge-decay teacher record into a decision-critical target."""
    target = _parse_target(edge_record.get("target", {}))
    hint = str(target.get("recommended_router_hint", ""))
    trend_side = str(target.get("trend_side", "NONE"))
    edge_label = str(target.get("edge_decay_label", ""))
    transition_label = str(target.get("transition_label", ""))
    diagnostics = edge_record.get("path_diagnostics") or {}
    long_same = diagnostics.get("long_same") or {}
    long_opp = diagnostics.get("long_opposite") or {}
    same_net = float(long_same.get("net_return", 0.0) or 0.0)
    opp_net = float(long_opp.get("net_return", 0.0) or 0.0)
    margin = same_net - opp_net

    if hint == "ALLOW_TREND_SPECIALIST" and trend_side in {"LONG", "SHORT"}:
        decision = "TRADE_TREND"
        action_side = trend_side
        evidence_margin = margin
    elif hint == "CONSIDER_REVERSAL_SPECIALIST" and trend_side in {"LONG", "SHORT"}:
        decision = "FADE_TREND"
        action_side = _opposite(trend_side)
        evidence_margin = -margin
    else:
        decision = "ABSTAIN"
        action_side = "NONE"
        evidence_margin = max(abs(same_net), abs(opp_net))

    target_out = {
        "decision": decision,
        "action_side": action_side,
        "confidence": _confidence_from_margin(evidence_margin, cfg),
        "rationale_class": _rationale_for(edge_label, transition_label, decision),
    }
    return target_out


def build_decision_prompt(edge_record: dict[str, Any]) -> str:
    summary = str(edge_record.get("past_summary") or edge_record.get("prompt", ""))
    if not isinstance(edge_record.get("past_summary"), str):
        # The edge prompt already embeds a natural-language past-only summary.
        # Reuse the original prompt text when the structured summary is a dict.
        summary = str(edge_record.get("prompt", ""))
    return "\n".join(
        [
            "You are the decision analyzer for a BTCUSDT futures router.",
            "Use only the past-only market and macro context below.",
            "Choose the smallest safe trading decision. Do not optimize thresholds or mention future returns.",
            "Decision set: TRADE_TREND means trade with the current trend; FADE_TREND means trade against it; ABSTAIN means no position.",
            "Return exactly one JSON object with keys decision, action_side, confidence, rationale_class.",
            "Allowed action_side values: LONG, SHORT, NONE. Use NONE only with ABSTAIN.",
            "Allowed confidence values: HIGH, MEDIUM, LOW.",
            "",
            f"Past-only context: {summary}",
        ]
    )


def build_decision_record(edge_record: dict[str, Any], cfg: DecisionAnalyzerConfig) -> dict[str, Any]:
    target = derive_decision_target(edge_record, cfg)
    source_target = _parse_target(edge_record.get("target", {}))
    return {
        "task": "decision_analyzer",
        "date": edge_record.get("date"),
        "signal_pos": edge_record.get("signal_pos"),
        "prompt": build_decision_prompt(edge_record),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "source_edge_target": source_target,
        "path_diagnostics": edge_record.get("path_diagnostics"),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_uses_future_path": True,
            "source_edge_prompt_was_past_only": not bool((edge_record.get("leakage_guard") or {}).get("prompt_uses_future_path", True)),
            "decision_target_is_compressed_from_edge_teacher": True,
            "not_gate_threshold_optimization": True,
        },
    }


def build_decision_records(edge_records: list[dict[str, Any]], cfg: DecisionAnalyzerConfig) -> list[dict[str, Any]]:
    return [build_decision_record(rec, cfg) for rec in edge_records]


def summarize_decision_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}
    counts: dict[str, Counter[str]] = {
        "decision": Counter(),
        "action_side": Counter(),
        "confidence": Counter(),
        "rationale_class": Counter(),
    }
    for rec in records:
        target = _parse_target(rec.get("target", {}))
        for key in counts:
            counts[key][str(target.get(key, ""))] += 1
    return {
        "num_records": len(records),
        "period": {"start": records[0].get("date"), "end": records[-1].get("date")},
        **{key: dict(counter) for key, counter in counts.items()},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_are_future_path_labels": True,
            "decision_target_is_compressed_from_edge_teacher": True,
            "not_gate_threshold_optimization": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build decision-critical analyzer SFT data from edge-decay records")
    p.add_argument("--edge-records", required=True)
    p.add_argument("--output", default="data/decision_analyzer.jsonl")
    p.add_argument("--summary-output", default="")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--min-actionable-edge", type=float, default=0.001)
    p.add_argument("--medium-margin", type=float, default=0.004)
    p.add_argument("--high-margin", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DecisionAnalyzerConfig(
        min_actionable_edge=float(args.min_actionable_edge),
        medium_margin=float(args.medium_margin),
        high_margin=float(args.high_margin),
    )
    edge_records = load_jsonl(args.edge_records)
    if args.max_records:
        edge_records = edge_records[: int(args.max_records)]
    records = build_decision_records(edge_records, cfg)
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "source": {"edge_records": args.edge_records},
        "records": summarize_decision_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
