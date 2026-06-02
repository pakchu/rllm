"""Balance calibrated-policy trader SFT rows so rare trade labels are learnable."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _target_reason(row: dict[str, Any]) -> str:
    target = json.loads(str(row["target"]))
    return str(target.get("reason", ""))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def balance_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    trade_repeat: int = 8,
    no_edge_per_trade: int = 2,
    skip_per_trade: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(int(seed))
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_reason[_target_reason(row)].append(row)
    trades = list(by_reason.get("CALIBRATED_EDGE", []))
    if not trades:
        raise ValueError("no CALIBRATED_EDGE rows found")
    target_trade = len(trades) * max(1, int(trade_repeat))
    balanced: list[dict[str, Any]] = []
    for _ in range(max(1, int(trade_repeat))):
        balanced.extend(trades)
    for reason, per_trade in [("NO_CALIBRATED_EDGE", no_edge_per_trade), ("POSITION_OPEN_SKIP", skip_per_trade)]:
        pool = list(by_reason.get(reason, []))
        take = min(len(pool), len(trades) * max(0, int(per_trade)))
        if take:
            balanced.extend(rng.sample(pool, take))
    rng.shuffle(balanced)
    summary = {
        "input_rows": len(rows),
        "output_rows": len(balanced),
        "input_reason_counts": dict(Counter(_target_reason(r) for r in rows)),
        "output_reason_counts": dict(Counter(_target_reason(r) for r in balanced)),
        "trade_repeat": int(trade_repeat),
        "no_edge_per_trade": int(no_edge_per_trade),
        "skip_per_trade": int(skip_per_trade),
        "seed": int(seed),
        "target_trade_rows": target_trade,
    }
    return balanced, summary


def run_balance(
    *,
    input: str,
    output: str,
    summary_output: str = "",
    seed: int = 42,
    trade_repeat: int = 8,
    no_edge_per_trade: int = 2,
    skip_per_trade: int = 2,
) -> dict[str, Any]:
    balanced, summary = balance_rows(
        read_jsonl(input),
        seed=seed,
        trade_repeat=trade_repeat,
        no_edge_per_trade=no_edge_per_trade,
        skip_per_trade=skip_per_trade,
    )
    write_jsonl(output, balanced)
    if summary_output:
        Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Balance calibrated-policy trader SFT rows")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trade-repeat", type=int, default=8)
    p.add_argument("--no-edge-per-trade", type=int, default=2)
    p.add_argument("--skip-per-trade", type=int, default=2)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run_balance(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
