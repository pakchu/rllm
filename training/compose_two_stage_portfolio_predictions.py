"""Compose gate and side label predictions into deployable portfolio actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["signal_pos"]), str(row.get("date", "")))


def _compose(gate_rows: list[dict[str, Any]], side_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    side_by_key = {_key(r): r for r in side_rows}
    out: list[dict[str, Any]] = []
    missing_side = 0
    for gate in gate_rows:
        k = _key(gate)
        side = side_by_key.get(k)
        if str(gate.get("prediction")) == "TRADE":
            if side is None:
                missing_side += 1
                pred = "NO_TRADE"
            else:
                pred = str(side.get("prediction", "NO_TRADE"))
        else:
            pred = "NO_TRADE"
        out.append(
            {
                "date": gate.get("date"),
                "signal_pos": gate.get("signal_pos"),
                "prediction": pred,
                "gate_prediction": gate.get("prediction"),
                "gate_target": gate.get("target"),
                "side_prediction": side.get("prediction") if side else None,
                "side_target": side.get("target") if side else None,
            }
        )
    if missing_side:
        raise ValueError(f"missing side predictions for {missing_side} gate TRADE rows; use side_eval_all")
    return out


def run(gate_predictions: str, side_predictions: str, output: str, summary_output: str) -> dict[str, Any]:
    gate_rows = _load_jsonl(gate_predictions)
    side_rows = _load_jsonl(side_predictions)
    out = _compose(gate_rows, side_rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    counts: dict[str, int] = {}
    for r in out:
        p = str(r["prediction"])
        counts[p] = counts.get(p, 0) + 1
    report = {
        "gate_predictions": gate_predictions,
        "side_predictions": side_predictions,
        "output": output,
        "rows": len(out),
        "prediction_counts": dict(sorted(counts.items())),
        "contract": "Deployable composition: gate TRADE opens a position using side model prediction for the same timestamp; gate NO_TRADE stays flat.",
    }
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose gate and side text-label predictions")
    p.add_argument("--gate-predictions", required=True)
    p.add_argument("--side-predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
