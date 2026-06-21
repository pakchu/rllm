"""Apply live-usable position scaling rules to generated action predictions.

This preserves trade decisions but annotates selected trades with ``position_scale``
so strict backtests can model conditional sizing without deleting the signal.
Rules are intentionally simple action predicates to keep selection auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _read(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _predicate(name: str) -> Callable[[dict[str, Any]], bool]:
    if name == "short":
        return lambda p: str(p.get("side", "")).upper() == "SHORT"
    if name == "short_or_mean_reversion":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" or str(p.get("family")) == "mean_reversion_stretch"
    if name == "short_or_higher_tf_momentum":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" or str(p.get("family")) == "higher_tf_momentum"
    if name == "short_or_mean_reversion_or_higher_tf_momentum":
        return lambda p: str(p.get("side", "")).upper() == "SHORT" or str(p.get("family")) in {"mean_reversion_stretch", "higher_tf_momentum"}
    if name == "kimchi_short_or_mean_reversion_or_higher_tf_momentum":
        return lambda p: (
            str(p.get("side", "")).upper() == "SHORT" and str(p.get("family")) == "kimchi_extreme_fade"
        ) or str(p.get("family")) in {"mean_reversion_stretch", "higher_tf_momentum"}
    raise ValueError(f"unknown rule: {name}")


def run(*, predictions_jsonl: str, output: str, rule: str, scale: float) -> dict[str, Any]:
    pred = _predicate(rule)
    rows = _read(predictions_jsonl)
    out: list[dict[str, Any]] = []
    scaled = 0
    for row in rows:
        p = row.get("prediction", {}) if isinstance(row.get("prediction"), dict) else {}
        if p.get("gate") == "TRADE" and pred(p):
            row = {**row, "position_scale": float(scale)}
            scaled += 1
        out.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"predictions_jsonl": predictions_jsonl, "output": output, "rule": rule, "scale": float(scale), "rows": len(rows), "scaled_trade_rows": scaled}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply position scaling to selected action predictions")
    p.add_argument("--predictions-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--rule",
        required=True,
        choices=[
            "short",
            "short_or_mean_reversion",
            "short_or_higher_tf_momentum",
            "short_or_mean_reversion_or_higher_tf_momentum",
            "kimchi_short_or_mean_reversion_or_higher_tf_momentum",
        ],
    )
    p.add_argument("--scale", type=float, required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
