"""Convert economic preference pairs into unique chosen-action SFT rows.

DPO-only training can collapse to a short candidate because it never first
learns the desired JSON action manifold.  This converter creates one
leakage-aware SFT row per signal timestamp using the highest-utility chosen
response already present in the preference file.  It is for warm-starting the
trader before DPO, not for reporting backtest performance.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _action_key(text: str) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text[:80]
    if not isinstance(obj, dict):
        return str(obj)[:80]
    return f"gate={obj.get('gate')},side={obj.get('side')},hold={obj.get('hold_bars')}"


def load_preference_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def preference_rows_to_sft(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_signal: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", -1)))
        chosen_action = row.get("chosen_action") or {}
        util = float(chosen_action.get("utility", 0.0) or 0.0)
        prev = best_by_signal.get(key)
        prev_util = float(((prev or {}).get("chosen_action") or {}).get("utility", -1e9) or -1e9)
        if prev is None or util > prev_util:
            best_by_signal[key] = row
    out: list[dict[str, Any]] = []
    for row in sorted(best_by_signal.values(), key=lambda r: (str(r.get("date")), int(r.get("signal_pos", -1)))):
        out.append(
            {
                "task": "economic_counterfactual_chosen_sft",
                "date": row.get("date"),
                "signal_pos": row.get("signal_pos"),
                "prompt": str(row["prompt"]),
                "target": str(row["chosen"]),
                "chosen_action": row.get("chosen_action"),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_ohlc_utility_for_training_only": True,
                    "one_row_per_signal_timestamp": True,
                },
            }
        )
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")


def summarize_sft_rows(rows: list[dict[str, Any]], source_rows: int) -> dict[str, Any]:
    targets = Counter(_action_key(str(r["target"])) for r in rows)
    prompts = [len(str(r.get("prompt", ""))) for r in rows]
    targets_len = [len(str(r.get("target", ""))) for r in rows]
    return {
        "source_preference_rows": source_rows,
        "sft_rows": len(rows),
        "deduped_preference_pairs": source_rows - len(rows),
        "period": {"start": rows[0].get("date") if rows else None, "end": rows[-1].get("date") if rows else None},
        "target_counts": dict(sorted(targets.items())),
        "prompt_chars": {"min": min(prompts) if prompts else 0, "max": max(prompts) if prompts else 0, "mean": sum(prompts) / max(1, len(prompts))},
        "target_chars": {"min": min(targets_len) if targets_len else 0, "max": max(targets_len) if targets_len else 0, "mean": sum(targets_len) / max(1, len(targets_len))},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_use_future_ohlc_utility": True,
            "not_a_backtest_result": True,
            "deduped_to_one_row_per_signal_timestamp": True,
        },
    }


def build_economic_preference_sft_jsonl(*, preferences: str, output: str, summary_output: str = "") -> dict[str, Any]:
    rows = load_preference_rows(preferences)
    sft_rows = preference_rows_to_sft(rows)
    write_jsonl(output, sft_rows)
    summary = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preferences": str(Path(preferences).resolve())},
        "outputs": {"sft_jsonl": output},
        "summary": summarize_sft_rows(sft_rows, len(rows)),
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert economic preference pairs to unique chosen-action SFT rows")
    p.add_argument("--preferences", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_economic_preference_sft_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
