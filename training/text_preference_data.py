"""Build leak-safe preference pairs from text trader records.

The direct JSON-label SFT path can memorize labels without learning a useful
ranking signal.  This stage keeps the same deployable prompt, but expands each
record into chosen/rejected responses so the next LLM stage can use DPO/ORPO or
reward-model style training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.eval_text_trader import parse_trader_json


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _target_with_hold(row: dict[str, Any]) -> dict[str, Any]:
    target = parse_trader_json(str(row["target"]))
    try:
        raw = json.loads(str(row["target"]))
    except Exception:
        raw = {}
    hold = int(raw.get("hold_bars", row.get("preferred_step_bars", 0)) or 0)
    if target["gate"] == "NO_TRADE":
        hold = 0
        target["side"] = "NONE"
    return {"gate": target["gate"], "side": target["side"], "hold_bars": hold}


def _response(action: dict[str, Any]) -> str:
    payload = {
        "gate": str(action["gate"]),
        "side": str(action["side"]),
        "hold_bars": int(action.get("hold_bars", 0) or 0),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _rejected_actions(chosen: dict[str, Any], hold_candidates: tuple[int, ...]) -> list[tuple[dict[str, Any], str]]:
    side = str(chosen["side"])
    hold = int(chosen.get("hold_bars", 0) or 0)
    if str(chosen["gate"]) == "NO_TRADE":
        default_hold = int(hold_candidates[0]) if hold_candidates else 48
        return [
            ({"gate": "TRADE", "side": "LONG", "hold_bars": default_hold}, "rejected_action_forces_long_trade_when_label_says_no_trade"),
            ({"gate": "TRADE", "side": "SHORT", "hold_bars": default_hold}, "rejected_action_forces_short_trade_when_label_says_no_trade"),
        ]
    opposite = "SHORT" if side == "LONG" else "LONG"
    rejected = [
        ({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, "rejected_action_skips_positive_utility_trade"),
        ({"gate": "TRADE", "side": opposite, "hold_bars": hold}, "rejected_action_uses_opposite_direction"),
    ]
    other_holds = [int(x) for x in hold_candidates if int(x) != hold]
    if other_holds:
        rejected.append(
            (
                {"gate": "TRADE", "side": side, "hold_bars": other_holds[0]},
                "rejected_action_uses_lower_ranked_hold_horizon",
            )
        )
    return rejected


def build_preference_pairs(
    rows: list[dict[str, Any]],
    *,
    hold_candidates: tuple[int, ...] = (48, 96, 144, 288),
    max_pairs_per_row: int = 2,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["prompt"])
        chosen_action = _target_with_hold(row)
        chosen = _response(chosen_action)
        for rejected_action, reason in _rejected_actions(chosen_action, hold_candidates)[: max(1, int(max_pairs_per_row))]:
            pairs.append(
                {
                    "task": "trader_preference",
                    "date": row.get("date"),
                    "signal_pos": row.get("signal_pos"),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": _response(rejected_action),
                    "chosen_action": chosen_action,
                    "rejected_action": rejected_action,
                    "rejection_reason": reason,
                    "leakage_guard": {
                        "prompt_reused_from_leak_safe_trader_record": True,
                        "chosen_rejected_use_future_path_labels_for_training_only": True,
                    },
                }
            )
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]], *, input_jsonl: str, output_jsonl: str) -> dict[str, Any]:
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    for row in pairs:
        c = row["chosen_action"]
        r = row["rejected_action"]
        chosen_counts[f"gate={c['gate']},side={c['side']},hold={c['hold_bars']}"] += 1
        rejected_counts[f"gate={r['gate']},side={r['side']},hold={r['hold_bars']}"] += 1
    return {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "output_jsonl": output_jsonl,
        "pairs": len(pairs),
        "chosen_counts": dict(sorted(chosen_counts.items())),
        "rejected_counts": dict(sorted(rejected_counts.items())),
        "leakage_guard": {
            "prompts_are_reused_from_trader_records": True,
            "future_labels_only_in_chosen_rejected_not_prompt": True,
        },
    }


def build_preference_jsonl(
    *,
    input_jsonl: str,
    output_jsonl: str,
    summary_output: str = "",
    hold_candidates: str = "48,96,144,288",
    max_pairs_per_row: int = 2,
) -> dict[str, Any]:
    holds = tuple(sorted({int(x.strip()) for x in str(hold_candidates).split(",") if x.strip()}))
    if not holds:
        raise ValueError("hold_candidates must not be empty")
    rows = read_jsonl(input_jsonl)
    pairs = build_preference_pairs(rows, hold_candidates=holds, max_pairs_per_row=max_pairs_per_row)
    write_jsonl(output_jsonl, pairs)
    summary = summarize_pairs(pairs, input_jsonl=input_jsonl, output_jsonl=output_jsonl)
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chosen/rejected preference pairs from text trader JSONL")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--hold-candidates", default="48,96,144,288")
    p.add_argument("--max-pairs-per-row", type=int, default=2)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_preference_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
