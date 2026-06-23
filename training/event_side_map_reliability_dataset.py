"""Build event-level side-map reliability labels from generated trade proposals.

Each generated TRADE row is labeled by comparing the realized return of the
predicted side with the flipped side:

- normal: predicted side is profitable and better than flipped side;
- inverse: flipped side is profitable and better than predicted side;
- unreliable: neither side is profitable enough.

Prompts/features are causal state tokens and prediction-score geometry. Targets
use future realized returns and are for training/evaluation only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventSideMapReliabilityDatasetCfg:
    predictions_jsonl: str
    source_context_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    train_end_month: str = "2024-12"
    val_end_month: str = "2025-12"
    min_abs_edge_pct: float = 0.0


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("date")), int(row.get("signal_pos", -1) or -1)


def _month(row: dict[str, Any]) -> str:
    return str(row.get("date", ""))[:7]


def _split(month: str, train_end: str, val_end: str) -> str:
    if month <= train_end:
        return "train"
    if month <= val_end:
        return "val"
    return "eval"


def _trade_side(row: dict[str, Any]) -> str:
    pred = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
    if pred.get("gate") != "TRADE":
        return "NONE"
    side = str(pred.get("side", "NONE")).upper()
    return side if side in {"LONG", "SHORT"} else "NONE"


def _actual(row: dict[str, Any], side: str) -> float:
    if side == "LONG":
        return float(row.get("actual_long_pct", 0.0) or 0.0)
    if side == "SHORT":
        return float(row.get("actual_short_pct", 0.0) or 0.0)
    return 0.0


def _label(row: dict[str, Any], side: str, min_abs_edge_pct: float) -> tuple[str, dict[str, float]]:
    other = "SHORT" if side == "LONG" else "LONG"
    chosen = _actual(row, side)
    inverted = _actual(row, other)
    edge = chosen - inverted
    min_edge = float(min_abs_edge_pct)
    if chosen > 0.0 and edge >= min_edge:
        return "normal", {"chosen_pct": chosen, "inverted_pct": inverted, "edge_pct": edge}
    if inverted > 0.0 and -edge >= min_edge:
        return "inverse", {"chosen_pct": chosen, "inverted_pct": inverted, "edge_pct": edge}
    return "unreliable", {"chosen_pct": chosen, "inverted_pct": inverted, "edge_pct": edge}


def _bucket_signed(x: float, small: float, large: float) -> str:
    if x >= large:
        return "high_positive"
    if x >= small:
        return "positive"
    if x <= -large:
        return "high_negative"
    if x <= -small:
        return "negative"
    return "neutral"


def _score_tokens(row: dict[str, Any]) -> dict[str, str]:
    long_s = float(row.get("score_long", 0.0) or 0.0)
    short_s = float(row.get("score_short", 0.0) or 0.0)
    wait_s = float(row.get("score_wait", 0.0) or 0.0)
    side_gap = abs(long_s - short_s)
    edge_wait = max(long_s, short_s) - wait_s
    return {
        "score_side_gap": _bucket_unsigned(side_gap, 0.05, 0.20),
        "score_edge_over_wait": _bucket_signed(edge_wait, 0.05, 0.20),
        "score_long_minus_short": _bucket_signed(long_s - short_s, 0.05, 0.20),
    }


def _bucket_unsigned(x: float, small: float, large: float) -> str:
    if x >= large:
        return "high"
    if x >= small:
        return "medium"
    return "low"


def _prompt(row: dict[str, Any], src: dict[str, Any], side: str, score_tokens: dict[str, str]) -> str:
    state = src.get("state_tokens") if isinstance(src.get("state_tokens"), dict) else {}
    lines = [
        "You are an event-level side-map reliability classifier for a BTCUSDT RLLM policy.",
        "Use only current causal state tokens and generated score geometry.",
        "Classify whether to trust the generated side, invert it, or avoid the event.",
        "Return one JSON object with keys: side_map, confidence, reason_code.",
        "Allowed side_map: normal, inverse, unreliable.",
        "",
        f"date: {row.get('date')}",
        f"generated_side: {side}",
        "score_geometry:",
    ]
    for k in sorted(score_tokens):
        lines.append(f"- {k}: {score_tokens[k]}")
    lines.append("causal_state_tokens:")
    for k in sorted(state):
        lines.append(f"- {k}: {state[k]}")
    lines.append("Policy intent: choose normal only when side evidence is stable, inverse only when reversal is clear, otherwise unreliable.")
    return "\n".join(lines)


def build(cfg: EventSideMapReliabilityDatasetCfg) -> dict[str, Any]:
    preds = _read_jsonl(cfg.predictions_jsonl)
    src_by_key = {_key(r): r for r in _read_jsonl(cfg.source_context_jsonl)}
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for row in preds:
        side = _trade_side(row)
        if side == "NONE":
            continue
        src = src_by_key.get(_key(row), {})
        label, audit = _label(row, side, float(cfg.min_abs_edge_pct))
        month = _month(row)
        split = _split(month, cfg.train_end_month, cfg.val_end_month)
        counts.setdefault(split, {})[label] = counts.setdefault(split, {}).get(label, 0) + 1
        score_tokens = _score_tokens(row)
        target = {"side_map": label, "confidence": "HIGH", "reason_code": f"event_audit_{label}"}
        rows.append({
            "task": "event_side_map_reliability_sft",
            "split": split,
            "date": row.get("date"),
            "month": month,
            "signal_pos": int(row.get("signal_pos", -1) or -1),
            "generated_side": side,
            "prompt": _prompt(row, src, side, score_tokens),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "state_tokens": src.get("state_tokens", {}) if isinstance(src.get("state_tokens"), dict) else {},
            "score_tokens": score_tokens,
            "label_audit": audit,
            "source_prediction": {k: row.get(k) for k in ("score_wait", "score_long", "score_short", "edge_over_wait", "runner_up_gap_pct")},
            "leakage_guard": {
                "prompt_uses_future_returns": False,
                "target_uses_future_realized_side_returns": True,
                "source_context_tokens_are_causal": True,
                "not_a_live_selector_without_rolling_eval": True,
            },
        })
    _write_jsonl(cfg.output_jsonl, rows)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(rows), "counts": counts, "leakage_guard": {"prompts_are_causal": True, "targets_are_event_realized_labels": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event-level side-map reliability SFT labels")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--source-context-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--train-end-month", default=EventSideMapReliabilityDatasetCfg.train_end_month)
    p.add_argument("--val-end-month", default=EventSideMapReliabilityDatasetCfg.val_end_month)
    p.add_argument("--min-abs-edge-pct", type=float, default=EventSideMapReliabilityDatasetCfg.min_abs_edge_pct)
    return p.parse_args()


def main() -> None:
    report = build(EventSideMapReliabilityDatasetCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output_jsonl"], "rows": report["rows"], "counts": report["counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
