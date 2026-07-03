"""Export fixed dual-regime REX predictions for online risk-overlay tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.sweep_conjunctive_event_gates import Gate, load_rows


def _match(row: dict[str, Any], gates: tuple[Gate, ...]) -> bool:
    return all(g.match(row) for g in gates)


def export_dual_regime_predictions(*, input_jsonl: str, output_jsonl: str) -> dict[str, Any]:
    rows = load_rows(input_jsonl)
    gate_sets = [
        (
            Gate("range_vol", ">=", 0.023959233645008706),
            Gate("kimchi_premium_change", "<=", 0.0),
        ),
        (
            Gate("rex_8640_range_width_pct", ">=", 0.2836633876944003),
            Gate("usdkrw_zscore", "<=", 0.2603593471820541),
        ),
    ]
    out = []
    counts = {"TRADE": 0, "NO_TRADE": 0}
    for row in rows:
        action = row["action"]
        hit = any(_match(row, gates) for gates in gate_sets)
        pred = {"gate": "TRADE", "family": action["family"], "side": action["side"], "hold_bars": action["hold_bars"]} if hit else {"gate": "NO_TRADE", "family": action["family"], "side": "NONE", "hold_bars": 0}
        counts["TRADE" if hit else "NO_TRADE"] += 1
        out.append({"date": row["date"], "signal_pos": row["signal_pos"], "prediction": pred, "source_action": action})
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(output_jsonl).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in out) + "\n")
    return {"input_rows": len(rows), "output_rows": len(out), "counts": counts, "output": output_jsonl}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export fixed dual-regime REX predictions")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    return p.parse_args()


def main() -> None:
    print(json.dumps(export_dual_regime_predictions(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
