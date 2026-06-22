"""Transform portfolio label predictions for validation-only policy tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _transform(pred: str, mode: str) -> str:
    pred = str(pred)
    if mode == "identity":
        return pred
    if mode == "invert_side":
        if pred == "LONG":
            return "SHORT"
        if pred == "SHORT":
            return "LONG"
        return "NO_TRADE"
    if mode == "trade_to_no_trade":
        return "NO_TRADE" if pred in {"LONG", "SHORT"} else pred
    raise ValueError(f"unknown mode: {mode}")


def run(input_jsonl: str, output: str, mode: str) -> dict[str, Any]:
    rows = _load(input_jsonl)
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in rows:
        rr = dict(row)
        rr["raw_prediction"] = row.get("prediction")
        rr["prediction"] = _transform(str(row.get("prediction", "NO_TRADE")), mode)
        counts[str(rr["prediction"])] = counts.get(str(rr["prediction"]), 0) + 1
        out.append(rr)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"input": input_jsonl, "output": output, "mode": mode, "rows": len(out), "prediction_counts": dict(sorted(counts.items()))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform portfolio predictions")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["identity", "invert_side", "trade_to_no_trade"], required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
