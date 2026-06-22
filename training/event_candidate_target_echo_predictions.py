"""Convert event candidate-ranking target labels into prediction JSONL for oracle sanity backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _decision(row: dict[str, Any]) -> str:
    target = row.get("target", {})
    if isinstance(target, dict):
        return str(target.get("decision", "ABSTAIN"))
    return str(target)


def run(input_jsonl: str, output: str, small_scale: float = 0.5) -> dict[str, Any]:
    rows = _load(input_jsonl)
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for r in rows:
        dec = _decision(r)
        side = str(r.get("side", "NONE"))
        hold = int(r.get("candidate", {}).get("hold_bars", 288) or 288)
        if dec in {"TAKE_FULL", "TAKE_SMALL"} and side in {"LONG", "SHORT"}:
            pred = {"gate": "TRADE", "side": side, "hold_bars": hold, "confidence": "HIGH", "family": "event_candidate_target_echo"}
            scale = 1.0 if dec == "TAKE_FULL" else float(small_scale)
        else:
            pred = {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW", "family": "event_candidate_target_echo"}
            scale = 0.0
        counts[dec] = counts.get(dec, 0) + 1
        out.append({"date": r.get("date"), "signal_pos": r.get("signal_pos"), "prediction": pred, "position_scale": scale, "target_decision": dec, "side": side, "candidate": r.get("candidate", {})})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"input": input_jsonl, "output": output, "rows": len(out), "decision_counts": dict(sorted(counts.items())), "small_scale": small_scale}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert candidate target labels to prediction JSONL")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--small-scale", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
