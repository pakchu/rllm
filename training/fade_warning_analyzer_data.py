"""Build narrow fade-warning analyzer SFT records.

This is the first repaired target that passed cheap no-leak learnability checks.
It intentionally outputs only ``fade_warning`` plus auxiliary ``skip_reason`` and
trend context, instead of full route/horizon decisions that failed majority
baselines.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.decision_feature_learnability import load_jsonl
from training.edge_decay_analyzer_data import write_jsonl
from training.repaired_router_state_data import RepairedRouterStateConfig, derive_repaired_router_state_target
from training.multi_horizon_edge_report import parse_horizons


@dataclass(frozen=True)
class FadeWarningConfig:
    hold_bars_list: tuple[int, ...] = (36, 72, 144, 288, 432)


def derive_fade_warning_target(source_target: str | dict[str, Any], cfg: FadeWarningConfig) -> dict[str, Any]:
    repaired = derive_repaired_router_state_target(
        source_target,
        RepairedRouterStateConfig(hold_bars_list=cfg.hold_bars_list),
    )
    return {
        "trend_side": repaired["trend_side"],
        "fade_warning": repaired["fade_warning"],
        "skip_reason": repaired["skip_reason"],
        "trend_continuation_quality": repaired["trend_continuation_quality"],
    }


def build_fade_warning_prompt(source_record: dict[str, Any]) -> str:
    prompt = str(source_record.get("prompt", ""))
    if "Past-only analyzer summary:" in prompt:
        past_summary = prompt.split("Past-only analyzer summary:", 1)[1].strip()
    else:
        past_summary = prompt[-3000:]
    return "\n".join(
        [
            "You are a fade-risk analyzer for BTCUSDT futures.",
            "Use only the past-only analyzer summary below.",
            "Focus on whether following the current trend has fade/reversal risk. Do not output an order, size, or final route.",
            "Return exactly one JSON object with keys trend_side, fade_warning, skip_reason, trend_continuation_quality.",
            "Allowed fade_warning: FADE_STRONG, FADE_WATCH, NO_FADE_WARNING.",
            "Allowed skip_reason: TRADEABLE_TREND, TRADEABLE_FADE, CONFLICTING_HORIZONS, ADVERSE_RISK, LOW_CONFIDENCE, NO_EDGE.",
            "Allowed trend_continuation_quality: CONTINUE_STRONG, CONTINUE_WATCH, NO_CONTINUATION.",
            "",
            f"Past-only analyzer summary: {past_summary}",
        ]
    )


def build_fade_warning_record(source_record: dict[str, Any], cfg: FadeWarningConfig) -> dict[str, Any]:
    target = derive_fade_warning_target(str(source_record.get("target", "{}")), cfg)
    return {
        "task": "fade_warning_analyzer",
        "date": source_record.get("date"),
        "signal_pos": source_record.get("signal_pos"),
        "prompt": build_fade_warning_prompt(source_record),
        "target": json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "source_task": source_record.get("task"),
        "leakage_guard": {
            "prompt_uses_future_path": False,
            "target_is_decomposed_from_future_path_shape_label": True,
            "target_is_final_order": False,
            "focus_key_passed_nb_learnability_gate": True,
        },
    }


def build_fade_warning_records(rows: list[dict[str, Any]], cfg: FadeWarningConfig) -> list[dict[str, Any]]:
    return [build_fade_warning_record(row, cfg) for row in rows]


def summarize_fade_warning_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {k: Counter() for k in ("fade_warning", "skip_reason", "trend_continuation_quality", "trend_side")}
    prompt_lens: list[int] = []
    target_lens: list[int] = []
    for rec in records:
        prompt_lens.append(len(str(rec.get("prompt", ""))))
        target_lens.append(len(str(rec.get("target", ""))))
        target = json.loads(str(rec.get("target", "{}")))
        for key, counter in counters.items():
            counter[str(target.get(key, ""))] += 1
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
    p = argparse.ArgumentParser(description="Build narrow fade-warning analyzer SFT records")
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-bars-list", default="36,72,144,288,432")
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FadeWarningConfig(hold_bars_list=parse_horizons(args.hold_bars_list))
    rows = load_jsonl(args.records)
    if args.max_records:
        rows = rows[: int(args.max_records)]
    records = build_fade_warning_records(rows, cfg)
    write_jsonl(args.output, records)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"records": args.records},
        "config": asdict(cfg),
        "records": summarize_fade_warning_records(records),
        "outputs": {"records": args.output},
    }
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
