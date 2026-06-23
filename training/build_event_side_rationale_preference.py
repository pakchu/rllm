"""Build evidence-rich side-map rationale preference rows.

Unlike bare `{"side_pair":"normal"}` labels, each candidate response contains a
compact causal rationale.  The prompt/response evidence is signal-time only;
future returns are used only to choose which candidate is preferred for training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORE_STATE_KEYS = (
    "trend_alignment",
    "trend_288",
    "trend_96",
    "trend_12",
    "range_pos",
    "rsi_norm",
    "bb_z",
    "window_drawdown",
    "pa_event_pressure",
    "pa_long_window_event",
    "pa_upside_rejection",
    "pa_downside_reclaim",
    "dxy_zscore",
    "dxy_momentum",
    "kimchi_premium_zscore",
    "kimchi_premium_change",
    "usdkrw_zscore",
    "usdkrw_momentum",
    "funding_zscore",
    "taker_imbalance",
)


@dataclass(frozen=True)
class BuildEventSideRationalePreferenceCfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def target_side_pair(row: dict[str, Any]) -> str:
    obj = json.loads(str(row.get("target", "{}")))
    label = str(obj.get("side_pair", obj.get("side_map", ""))).strip().lower()
    return label if label in {"normal", "inverse"} else ""


def _score_tokens(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("score_tokens", {}) if isinstance(row.get("score_tokens"), dict) else {}


def _state_tokens(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("state_tokens", {}) if isinstance(row.get("state_tokens"), dict) else {}


def build_prompt(row: dict[str, Any]) -> str:
    score = _score_tokens(row)
    state = _state_tokens(row)
    lines = [
        "You are a BTCUSDT futures side-map judge.",
        "Use only signal-time causal tokens. Do not infer or mention future returns.",
        "Compare two possible side-map actions for the generated policy side:",
        "- normal: trust the generated side",
        "- inverse: invert the generated side",
        "Return the stronger candidate response.",
        f"date: {row.get('date')}",
        f"generated_side: {row.get('generated_side')}",
        "score_geometry:",
    ]
    for key in sorted(score):
        lines.append(f"- {key}: {score[key]}")
    lines.append("causal_state_tokens:")
    for key in CORE_STATE_KEYS:
        if key in state:
            lines.append(f"- {key}: {state[key]}")
    return "\n".join(lines)


def _setup_tags(row: dict[str, Any]) -> list[str]:
    state = _state_tokens(row)
    score = _score_tokens(row)
    tags: list[str] = []
    for key in ("trend_alignment", "pa_event_pressure", "pa_long_window_event", "window_drawdown", "range_pos"):
        if key in state:
            tags.append(f"{key}={state[key]}")
    for key in ("score_side_gap", "score_edge_over_wait", "score_long_minus_short"):
        if key in score:
            tags.append(f"{key}={score[key]}")
    return tags[:8]


def candidate_response(row: dict[str, Any], side_pair: str) -> str:
    label = str(side_pair).strip().lower()
    if label not in {"normal", "inverse"}:
        raise ValueError("side_pair must be normal or inverse")
    state = _state_tokens(row)
    score = _score_tokens(row)
    if label == "normal":
        rationale_class = "trust_generated_score_geometry"
        action = "keep generated side because score geometry is accepted unless causal state shows a fade warning"
        counter = "fails when state tokens imply generated-side exhaustion or regime inversion"
    else:
        rationale_class = "invert_generated_side_on_fade_warning"
        action = "invert generated side because causal state is treated as a fade or regime-inversion warning"
        counter = "fails when score geometry is genuinely directional and state tokens confirm continuation"
    payload = {
        "side_pair": label,
        "rationale_class": rationale_class,
        "causal_evidence": {
            "action_logic": action,
            "counter_risk": counter,
            "setup_tags": _setup_tags(row),
            "trend_alignment": state.get("trend_alignment", "unknown"),
            "price_action_pressure": state.get("pa_event_pressure", "unknown"),
            "long_window_event": state.get("pa_long_window_event", "unknown"),
            "drawdown_state": state.get("window_drawdown", "unknown"),
            "score_gap": score.get("score_side_gap", "unknown"),
            "edge_over_wait": score.get("score_edge_over_wait", "unknown"),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build(cfg: BuildEventSideRationalePreferenceCfg) -> dict[str, Any]:
    rows = read_jsonl(cfg.input_jsonl)
    out: list[dict[str, Any]] = []
    chosen_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for row in rows:
        chosen_label = target_side_pair(row)
        if not chosen_label:
            skipped["non_pair_label"] += 1
            continue
        rejected_label = "inverse" if chosen_label == "normal" else "normal"
        nr = {
            "task": "event_side_rationale_preference",
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "month": row.get("month"),
            "generated_side": row.get("generated_side"),
            "prompt": build_prompt(row),
            "chosen": candidate_response(row, chosen_label),
            "rejected": candidate_response(row, rejected_label),
            "chosen_side_pair": chosen_label,
            "rejected_side_pair": rejected_label,
            "leakage_guard": {
                "prompt_uses_signal_time_tokens_only": True,
                "candidate_rationales_use_signal_time_tokens_only": True,
                "chosen_rejected_preference_uses_future_realized_side_returns_for_training_only": True,
            },
        }
        out.append(nr)
        chosen_counts[chosen_label] += 1
    write_jsonl(cfg.output_jsonl, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows_in": len(rows),
        "pairs_out": len(out),
        "chosen_counts": dict(chosen_counts),
        "skipped_counts": dict(skipped),
        "prompt_chars": _char_stats([r["prompt"] for r in out]),
        "chosen_chars": _char_stats([r["chosen"] for r in out]),
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _char_stats(values: list[str]) -> dict[str, float]:
    lens = [len(str(v)) for v in values]
    return {"min": min(lens) if lens else 0, "max": max(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event side rationale preference rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(BuildEventSideRationalePreferenceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
