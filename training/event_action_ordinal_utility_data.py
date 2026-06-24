"""Build candidate-wise ordinal utility labels from event-action value rows.

This avoids the two failed target shapes:
- binary TAKE/SKIP dominated by SKIP prior,
- pairwise A/B labels dominated by position prior.

Each row keeps one candidate action and asks for an ordinal utility class derived
from future strict utility for training/eval labels only.  Prompts remain
past-only and contain no utility audits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABELS = ("AVOID", "LOW", "MID", "HIGH")


@dataclass(frozen=True)
class EventActionOrdinalUtilityConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    avoid_below: float = -0.01
    mid_at: float = 0.004
    high_at: float = 0.012
    max_mae_high: float = 0.018


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _audit(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}


def _utility(row: dict[str, Any]) -> float:
    audit = _audit(row)
    val = audit.get("rank_utility", audit.get("utility", -1e9))
    return -1e9 if val is None else float(val)


def _mae(row: dict[str, Any]) -> float:
    val = _audit(row).get("mae", 1e9)
    return 1e9 if val is None else float(val)


def label_for_row(row: dict[str, Any], cfg: EventActionOrdinalUtilityConfig) -> str:
    util = _utility(row)
    mae = _mae(row)
    if util < float(cfg.avoid_below):
        return "AVOID"
    if util >= float(cfg.high_at) and mae <= float(cfg.max_mae_high):
        return "HIGH"
    if util >= float(cfg.mid_at):
        return "MID"
    return "LOW"


def _base_context(prompt: str) -> str:
    lines = []
    for line in str(prompt).splitlines():
        if line.startswith("Output exactly one label:"):
            continue
        if line.startswith("TAKE only if"):
            continue
        if line.startswith("Do not output JSON"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _ordinal_prompt(prompt: str) -> str:
    return "\n".join(
        [
            "You are an ordinal action utility judge for BTCUSDT futures.",
            "Use only the past-only state, prompt-visible action book, and candidate action.",
            "Classify the candidate's expected utility after path risk.",
            "Output exactly one label: AVOID, LOW, MID, or HIGH.",
            "Definitions: AVOID=negative/fragile, LOW=weak/no edge, MID=usable edge, HIGH=strong edge with controlled adverse path.",
            "",
            _base_context(prompt),
        ]
    )


def build_ordinal_rows(rows: list[dict[str, Any]], cfg: EventActionOrdinalUtilityConfig) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        label = label_for_row(row, cfg)
        out.append(
            {
                "task": "event_action_ordinal_utility",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "action": row.get("action"),
                "prompt": _ordinal_prompt(str(row.get("prompt", ""))),
                "target": label,
                "action_audit": row.get("action_audit", {}),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_utility_for_training_only": True,
                    "candidate_book_uses_past_only_features": True,
                },
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], cfg: EventActionOrdinalUtilityConfig) -> dict[str, Any]:
    counts = Counter(str(r.get("target")) for r in rows)
    side_counts = Counter(str((r.get("action") or {}).get("side")) for r in rows)
    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    return {
        "input_jsonl": str(Path(cfg.input_jsonl).resolve()),
        "output_jsonl": cfg.output_jsonl,
        "rows": len(rows),
        "signals": len({(r.get("date"), r.get("signal_pos")) for r in rows}),
        "target_counts": dict(sorted(counts.items())),
        "side_counts": dict(sorted(side_counts.items())),
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "config": asdict(cfg),
        "leakage_guard": {"prompts_are_past_only": True, "future_utility_only_in_target_and_audit": True, "not_a_backtest_result": True},
    }


def build_ordinal_jsonl(**kwargs: Any) -> dict[str, Any]:
    cfg = EventActionOrdinalUtilityConfig(**kwargs)
    rows = build_ordinal_rows(read_jsonl(cfg.input_jsonl), cfg)
    write_jsonl(cfg.output_jsonl, rows)
    summary = summarize(rows, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ordinal utility labels for event-action candidates")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--avoid-below", type=float, default=-0.01)
    p.add_argument("--mid-at", type=float, default=0.004)
    p.add_argument("--high-at", type=float, default=0.012)
    p.add_argument("--max-mae-high", type=float, default=0.018)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_ordinal_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
