"""Export path-shape trader SFT targets as prediction rows for oracle backtests.

This is a label upper-bound diagnostic: it converts future-derived trader targets
into the same prediction schema consumed by online_risk_overlay_backtest.  It is
not deployable by itself; model-generated predictions must replace target echo.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathShapeTargetExportCfg:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _parse_target(raw: Any) -> dict[str, Any]:
    obj = json.loads(str(raw)) if not isinstance(raw, dict) else raw
    gate = str(obj.get("gate", "NO_TRADE")).upper()
    side = str(obj.get("side", "NONE")).upper()
    if gate != "TRADE":
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    hold = int(obj.get("hold_bars", obj.get("max_hold_bars", 0)) or 0)
    if side not in {"LONG", "SHORT"} or hold <= 0:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}
    return {
        "gate": "TRADE",
        "side": side,
        "hold_bars": hold,
        "target_pct": float(obj.get("target_pct", 0.0) or 0.0),
        "stop_pct": float(obj.get("stop_pct", 0.0) or 0.0),
    }


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    action = _parse_target(row.get("target", {}))
    return {
        "date": row.get("date"),
        "signal_pos": int(row.get("signal_pos", -1)),
        "prediction": action,
        "target_echo": True,
        "source_task": row.get("task"),
        "pressure": row.get("pressure"),
        "leakage_guard": {
            "prediction_is_future_target_echo": True,
            "for_oracle_upper_bound_only": True,
        },
    }


def export_targets(cfg: PathShapeTargetExportCfg) -> dict[str, Any]:
    rows = _load_jsonl(cfg.input_jsonl)
    out = [convert_row(r) for r in rows]
    Path(cfg.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(cfg.output_jsonl).open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    actions = Counter(f"{r['prediction']['gate']}/{r['prediction']['side']}" for r in out)
    holds = Counter(int(r["prediction"].get("hold_bars", 0) or 0) for r in out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "rows": len(out),
        "actions": dict(sorted(actions.items())),
        "hold_bars": dict(sorted((str(k), v) for k, v in holds.items())),
        "period": {"start": out[0].get("date") if out else None, "end": out[-1].get("date") if out else None},
        "leakage_note": "target echo uses future-derived labels and is only an oracle upper-bound diagnostic",
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export path-shape trader targets as target-echo prediction rows")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    print(json.dumps(export_targets(PathShapeTargetExportCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
