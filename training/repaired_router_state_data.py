"""Build repaired compact router-state labels for learnability-first LLM SFT.

The compact run1 target exposed two failures: ``risk_budget`` collapsed and
``action_path`` mixed too many trend/fade/skip questions into one class.  This
module decomposes the teacher path-shape state into simpler ordinal/binary
questions that should be checked with a cheap baseline before another SFT:

- trend_continuation_quality: how safe/useful following trend_side appears
- fade_warning: whether fade/reversal pressure is meaningful
- skip_reason: why the router should avoid or downsize
- horizon_policy: coarse step choice retained because it was learnable

Targets are still derived from future path-shape teacher labels; prompts are
past-only and outputs are router-state supervision, not executable orders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.compact_path_shape_analyzer_data import RETURN_SCORE, MAE_PENALTY
from training.decision_feature_learnability import load_jsonl
from training.edge_decay_analyzer_data import write_jsonl
from training.eval_multi_horizon_path_shape_analyzer import parse_path_shape_json
from training.multi_horizon_edge_report import parse_horizons


@dataclass(frozen=True)
class RepairedRouterStateConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)
    continuation_strong: float = 3.0
    continuation_watch: float = 1.25
    fade_strong: float = 3.0
    fade_watch: float = 1.25
    short_horizon_bars: int = 72
    long_horizon_bars: int = 288


def _score(hdata: dict[str, Any], prefix: str) -> float:
    return float(RETURN_SCORE.get(str(hdata.get(f"{prefix}_return_bucket", "UNAVAILABLE")), -99.0)) - float(
        MAE_PENALTY.get(str(hdata.get(f"{prefix}_mae_bucket", "UNAVAILABLE")), 99.0)
    )


def _horizon_policy(h: int, cfg: RepairedRouterStateConfig) -> str:
    if h <= int(cfg.short_horizon_bars):
        return "SHORT_STEP"
    if h < int(cfg.long_horizon_bars):
        return "MID_STEP"
    return "LONG_STEP"


def derive_repaired_router_state_target(source_target: str | dict[str, Any], cfg: RepairedRouterStateConfig) -> dict[str, Any]:
    parsed = parse_path_shape_json(source_target if isinstance(source_target, str) else json.dumps(source_target), horizons=cfg.hold_bars_list)
    best_trend = {"score": -99.0, "horizon": 0}
    best_fade = {"score": -99.0, "horizon": 0}
    for h in cfg.hold_bars_list:
        hdata = dict((parsed.get("horizons") or {}).get(str(int(h))) or {})
        ts = _score(hdata, "trend")
        fs = _score(hdata, "fade")
        if ts > float(best_trend["score"]):
            best_trend = {"score": ts, "horizon": int(h)}
        if fs > float(best_fade["score"]):
            best_fade = {"score": fs, "horizon": int(h)}

    trend_score = float(best_trend["score"])
    fade_score = float(best_fade["score"])
    margin = trend_score - fade_score
    risk = str(parsed.get("risk_profile", "MIXED_PATH_RISK"))
    reversal = str(parsed.get("reversal_pressure", "LOW"))
    stability = str(parsed.get("direction_stability", "NO_STABLE_EDGE"))

    if trend_score >= float(cfg.continuation_strong) and margin >= 0.75:
        trend_quality = "CONTINUE_STRONG"
    elif trend_score >= float(cfg.continuation_watch) and margin >= -0.75:
        trend_quality = "CONTINUE_WATCH"
    else:
        trend_quality = "NO_CONTINUATION"

    if fade_score >= float(cfg.fade_strong) and margin <= -0.75:
        fade_warning = "FADE_STRONG"
    elif fade_score >= float(cfg.fade_watch) or reversal == "HIGH" or stability == "HORIZON_CONFLICT":
        fade_warning = "FADE_WATCH"
    else:
        fade_warning = "NO_FADE_WARNING"

    if trend_quality == "NO_CONTINUATION" and fade_warning == "NO_FADE_WARNING":
        skip_reason = "NO_EDGE"
    elif risk == "EXTREME_PATH_RISK":
        skip_reason = "ADVERSE_RISK"
    elif fade_warning == "FADE_STRONG":
        skip_reason = "TRADEABLE_FADE"
    elif trend_quality == "CONTINUE_STRONG" and fade_warning != "FADE_STRONG":
        skip_reason = "TRADEABLE_TREND"
    elif stability == "HORIZON_CONFLICT":
        skip_reason = "CONFLICTING_HORIZONS"
    else:
        skip_reason = "LOW_CONFIDENCE"

    if skip_reason == "TRADEABLE_FADE":
        primary_route = "FADE"
        horizon = int(best_fade["horizon"])
    elif skip_reason == "TRADEABLE_TREND":
        primary_route = "TREND"
        horizon = int(best_trend["horizon"])
    else:
        primary_route = "SKIP"
        horizon = 0

    return {
        "trend_side": parsed.get("trend_side", "NONE"),
        "trend_continuation_quality": trend_quality,
        "fade_warning": fade_warning,
        "skip_reason": skip_reason,
        "primary_route": primary_route,
        "horizon_policy": _horizon_policy(horizon, cfg) if horizon else "SKIP_STEP",
        "trend_score_bucket": "HIGH" if trend_score >= 3.0 else "MEDIUM" if trend_score >= 1.25 else "LOW_OR_NEGATIVE",
        "fade_score_bucket": "HIGH" if fade_score >= 3.0 else "MEDIUM" if fade_score >= 1.25 else "LOW_OR_NEGATIVE",
    }


def build_repaired_prompt(source_record: dict[str, Any], cfg: RepairedRouterStateConfig) -> str:
    prompt = str(source_record.get("prompt", ""))
    if "Past-only analyzer summary:" in prompt:
        past_summary = prompt.split("Past-only analyzer summary:", 1)[1].strip()
    else:
        past_summary = prompt[-3000:]
    return "\n".join(
        [
            "You are a repaired router-state analyzer for BTCUSDT futures.",
            "Use only the past-only analyzer summary below.",
            "Answer simpler router questions instead of forcing final action sizing.",
            "Return exactly one JSON object with keys trend_side, trend_continuation_quality, fade_warning, skip_reason, primary_route, horizon_policy, trend_score_bucket, fade_score_bucket.",
            "Allowed trend_continuation_quality: CONTINUE_STRONG, CONTINUE_WATCH, NO_CONTINUATION.",
            "Allowed fade_warning: FADE_STRONG, FADE_WATCH, NO_FADE_WARNING.",
            "Allowed skip_reason: TRADEABLE_TREND, TRADEABLE_FADE, CONFLICTING_HORIZONS, ADVERSE_RISK, LOW_CONFIDENCE, NO_EDGE.",
            "Allowed primary_route: TREND, FADE, SKIP. Allowed horizon_policy: SHORT_STEP, MID_STEP, LONG_STEP, SKIP_STEP.",
            "",
            f"Past-only analyzer summary: {past_summary}",
        ]
    )


def build_repaired_record(source_record: dict[str, Any], cfg: RepairedRouterStateConfig) -> dict[str, Any]:
    target = derive_repaired_router_state_target(str(source_record.get("target", "{}")), cfg)
    return {
        "task": "repaired_router_state_analyzer",
        "date": source_record.get("date"),
        "signal_pos": source_record.get("signal_pos"),
        "prompt": build_repaired_prompt(source_record, cfg),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "source_task": source_record.get("task"),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_is_decomposed_from_future_path_shape_label": True,
            "target_is_final_order": False,
            "risk_budget_removed_due_to_class_collapse": True,
        },
    }


def build_repaired_records(rows: list[dict[str, Any]], cfg: RepairedRouterStateConfig) -> list[dict[str, Any]]:
    return [build_repaired_record(row, cfg) for row in rows]


def summarize_repaired_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("trend_continuation_quality", "fade_warning", "skip_reason", "primary_route", "horizon_policy", "trend_side")
    counters = {k: Counter() for k in keys}
    prompt_lens: list[int] = []
    target_lens: list[int] = []
    for rec in records:
        prompt_lens.append(len(str(rec.get("prompt", ""))))
        target_lens.append(len(str(rec.get("target", ""))))
        target = json.loads(str(rec.get("target", "{}")))
        for key in keys:
            counters[key][str(target.get(key, ""))] += 1
    return {
        "num_records": len(records),
        "period": {"start": records[0].get("date") if records else None, "end": records[-1].get("date") if records else None},
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "target_chars": {"min": min(target_lens) if target_lens else 0, "max": max(target_lens) if target_lens else 0, "mean": sum(target_lens) / max(1, len(target_lens))},
        **{key: dict(counter) for key, counter in counters.items()},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_are_decomposed_from_future_path_shape_labels": True,
            "targets_are_router_states_not_final_orders": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build repaired router-state analyzer SFT data")
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RepairedRouterStateConfig(hold_bars_list=parse_horizons(args.hold_bars_list))
    rows = load_jsonl(args.records)
    if args.max_records:
        rows = rows[: int(args.max_records)]
    records = build_repaired_records(rows, cfg)
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records": args.records},
        "config": asdict(cfg),
        "records": summarize_repaired_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
