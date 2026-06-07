"""Split stable trader action/risk data into gate and side specialist tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_target(row: dict[str, Any]) -> dict[str, str]:
    obj = json.loads(str(row["target"]))
    action = str(obj.get("action", "NO_TRADE")).upper()
    risk = str(obj.get("risk", "HIGH")).upper()
    return {"action": action, "risk": risk}


def gate_prompt(row: dict[str, Any]) -> str:
    return str(row["prompt"]).replace(
        'Return exactly one JSON object: {"action": <LONG|SHORT|NO_TRADE>, "risk": <LOW|MEDIUM|HIGH>}',
        'Return exactly one JSON object: {"gate": <TRADE|NO_TRADE>, "risk": <LOW|MEDIUM|HIGH>}',
    )


def side_prompt(row: dict[str, Any]) -> str:
    return str(row["prompt"]).replace(
        'Return exactly one JSON object: {"action": <LONG|SHORT|NO_TRADE>, "risk": <LOW|MEDIUM|HIGH>}',
        'Return exactly one JSON object: {"side": <LONG|SHORT>, "risk": <LOW|MEDIUM|HIGH>}. A trade has already been approved by the gate stage.',
    )


def to_gate_row(row: dict[str, Any]) -> dict[str, Any]:
    target = parse_target(row)
    gate = "NO_TRADE" if target["action"] == "NO_TRADE" else "TRADE"
    out = dict(row)
    out["task"] = "stable_trader_gate_sft"
    out["prompt"] = gate_prompt(row)
    out["target"] = json.dumps({"gate": gate, "risk": target["risk"]}, ensure_ascii=False, sort_keys=True)
    out["gate"] = gate
    return out


def to_side_row(row: dict[str, Any]) -> dict[str, Any] | None:
    target = parse_target(row)
    if target["action"] == "NO_TRADE":
        return None
    out = dict(row)
    out["task"] = "stable_trader_side_sft"
    out["prompt"] = side_prompt(row)
    out["target"] = json.dumps({"side": target["action"], "risk": target["risk"]}, ensure_ascii=False, sort_keys=True)
    out["side"] = target["action"]
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get(key))] = counts.get(str(r.get(key)), 0) + 1
        split_counts[str(r.get("split"))] = split_counts.get(str(r.get("split")), 0) + 1
    return {"rows": len(rows), f"by_{key}": counts, "by_split": split_counts}


def split_tasks(*, input_jsonl: str, output_prefix: str, summary_output: str) -> dict[str, Any]:
    rows = load_jsonl(input_jsonl)
    gate_rows = [to_gate_row(r) for r in rows]
    side_rows = [s for r in rows if (s := to_side_row(r)) is not None]
    outputs = {
        "gate": f"{output_prefix}_gate.jsonl",
        "side": f"{output_prefix}_side.jsonl",
    }
    write_jsonl(outputs["gate"], gate_rows)
    write_jsonl(outputs["side"], side_rows)
    for split in ["train", "val", "eval"]:
        outputs[f"gate_{split}"] = f"{output_prefix}_gate_{split}.jsonl"
        outputs[f"side_{split}"] = f"{output_prefix}_side_{split}.jsonl"
        write_jsonl(outputs[f"gate_{split}"], [r for r in gate_rows if r.get("split") == split])
        write_jsonl(outputs[f"side_{split}"], [r for r in side_rows if r.get("split") == split])
    report = {"input": input_jsonl, "outputs": outputs, "gate": summarize(gate_rows, "gate"), "side": summarize(side_rows, "side")}
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--summary-output", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(split_tasks(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
