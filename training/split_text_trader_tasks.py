"""LEGACY / DEPRECATED: kept only to reproduce historical analyzer/trader experiments.

Split analyzer-summary trader JSONL into separate gate and side SFT tasks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.eval_text_trader import parse_trader_json


def _extract_summary(prompt: str) -> str:
    marker = "Analyzer summary:"
    if marker not in prompt:
        return prompt.strip()
    return prompt.split(marker, 1)[1].strip()


def _replace_instruction(prompt: str, *, task: str) -> str:
    summary = _extract_summary(prompt)
    if task == "gate":
        return "\n".join(
            [
                "You are the gate stage for a BTCUSDT futures trading bot.",
                "You receive only the analyzer's past-only symbolic market summary.",
                "Decide whether an executable trade is justified after fees/slippage and adverse excursion risk.",
                "Output exactly one JSON object with key gate. gate must be TRADE or NO_TRADE.",
                "",
                f"Analyzer summary: {summary}",
            ]
        )
    if task == "side":
        return "\n".join(
            [
                "You are the side stage for a BTCUSDT futures trading bot.",
                "The gate has already decided TRADE. Use only the analyzer's past-only symbolic market summary.",
                "Choose the executable direction after fees/slippage and adverse excursion risk.",
                "Output exactly one JSON object with key side. side must be LONG or SHORT.",
                "",
                f"Analyzer summary: {summary}",
            ]
        )
    raise ValueError(f"unsupported task: {task}")


def split_trader_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gate_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    for row in rows:
        target = parse_trader_json(str(row["target"]))
        common = {k: v for k, v in row.items() if k not in {"task", "prompt", "target"}}
        gate_rows.append(
            {
                **common,
                "task": "trader_gate",
                "prompt": _replace_instruction(str(row["prompt"]), task="gate"),
                "target": json.dumps({"gate": target["gate"]}, sort_keys=True, separators=(",", ":")),
            }
        )
        if target["gate"] == "TRADE" and target["side"] in {"LONG", "SHORT"}:
            side_rows.append(
                {
                    **common,
                    "task": "trader_side",
                    "prompt": _replace_instruction(str(row["prompt"]), task="side"),
                    "target": json.dumps({"side": target["side"]}, sort_keys=True, separators=(",", ":")),
                }
            )
    return gate_rows, side_rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def split_trader_jsonl(*, input_jsonl: str, gate_output: str, side_output: str, summary_output: str = "") -> dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    gate_rows, side_rows = split_trader_rows(rows)
    write_jsonl(gate_output, gate_rows)
    write_jsonl(side_output, side_rows)
    summary = {
        "input_jsonl": str(Path(input_jsonl).resolve()),
        "outputs": {"gate": gate_output, "side": side_output},
        "records": {"input": len(rows), "gate": len(gate_rows), "side": len(side_rows)},
        "leakage_guard": {
            "prompts_use_analyzer_summary_only": True,
            "side_rows_include_trade_targets_only": True,
        },
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split text trader JSONL into gate and side tasks")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--gate-output", required=True)
    p.add_argument("--side-output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(split_trader_jsonl(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
