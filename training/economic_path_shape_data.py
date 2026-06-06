"""Generate path-shape labels for analyzer/trader training.

The previous hold-only future-best label creates an oracle action lottery.  This
module produces explicit path diagnostics (MFE/MAE, first target/stop timing,
side pressure) so an analyzer can describe risk/reward shape and a trader/RL
stage can learn from predeclared stop/target templates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.economic_opportunity_baseline import best_rows_by_signal
from training.economic_value_baseline import load_jsonl, write_jsonl


@dataclass(frozen=True)
class PathTemplate:
    horizon_bars: int = 144
    target_pct: float = 1.0
    stop_pct: float = 0.6
    entry_delay_bars: int = 1


def _first_hit(highs: list[float], lows: list[float], *, entry: float, side: str, target_pct: float, stop_pct: float) -> dict[str, Any]:
    target = target_pct / 100.0
    stop = stop_pct / 100.0
    for i, (h, l) in enumerate(zip(highs, lows), start=1):
        if side == "LONG":
            hit_target = (h / entry - 1.0) >= target
            hit_stop = (l / entry - 1.0) <= -stop
        else:
            hit_target = (entry / l - 1.0) >= target if l > 0 else False
            hit_stop = (entry / h - 1.0) <= -stop if h > 0 else False
        if hit_target and hit_stop:
            return {"first_event": "AMBIGUOUS_SAME_BAR", "bars": i}
        if hit_target:
            return {"first_event": "TARGET", "bars": i}
        if hit_stop:
            return {"first_event": "STOP", "bars": i}
    return {"first_event": "NONE", "bars": None}


def compute_path_shape(market: pd.DataFrame, signal_pos: int, template: PathTemplate = PathTemplate()) -> dict[str, Any] | None:
    entry_pos = int(signal_pos) + int(template.entry_delay_bars)
    end_pos = entry_pos + int(template.horizon_bars)
    if entry_pos < 0 or end_pos >= len(market):
        return None
    entry = float(market.iloc[entry_pos]["open"])
    future = market.iloc[entry_pos + 1 : end_pos + 1]
    if entry <= 0 or future.empty:
        return None
    highs = [float(x) for x in future["high"].to_list()]
    lows = [float(x) for x in future["low"].to_list()]
    close = float(future.iloc[-1]["close"])

    long_mfe = (max(highs) / entry - 1.0) * 100.0
    long_mae = (min(lows) / entry - 1.0) * 100.0
    short_mfe = (entry / min(lows) - 1.0) * 100.0
    short_mae = (entry / max(highs) - 1.0) * 100.0
    long_close = (close / entry - 1.0) * 100.0
    short_close = (entry / close - 1.0) * 100.0

    long_hit = _first_hit(highs, lows, entry=entry, side="LONG", target_pct=template.target_pct, stop_pct=template.stop_pct)
    short_hit = _first_hit(highs, lows, entry=entry, side="SHORT", target_pct=template.target_pct, stop_pct=template.stop_pct)
    long_rr = long_mfe / max(abs(long_mae), 1e-9)
    short_rr = short_mfe / max(abs(short_mae), 1e-9)

    def grade(hit: dict[str, Any], rr: float, close_ret: float) -> str:
        if hit["first_event"] == "TARGET" and rr >= 1.5:
            return "CLEAN_TARGET"
        if hit["first_event"] == "TARGET":
            return "NOISY_TARGET"
        if hit["first_event"] == "STOP":
            return "STOP_FIRST"
        if close_ret > 0 and rr >= 1.2:
            return "DRIFT_POSITIVE"
        return "NO_EDGE"

    long_grade = grade(long_hit, long_rr, long_close)
    short_grade = grade(short_hit, short_rr, short_close)
    if long_grade in {"CLEAN_TARGET", "NOISY_TARGET"} and short_grade not in {"CLEAN_TARGET", "NOISY_TARGET"}:
        pressure = "LONG_FAVORED"
    elif short_grade in {"CLEAN_TARGET", "NOISY_TARGET"} and long_grade not in {"CLEAN_TARGET", "NOISY_TARGET"}:
        pressure = "SHORT_FAVORED"
    elif long_grade in {"CLEAN_TARGET", "NOISY_TARGET"} and short_grade in {"CLEAN_TARGET", "NOISY_TARGET"}:
        pressure = "BOTH_SIDES_VOLATILE"
    else:
        pressure = "NO_TRADE_FAVORED"

    return {
        "template": {
            "horizon_bars": template.horizon_bars,
            "target_pct": template.target_pct,
            "stop_pct": template.stop_pct,
            "entry_delay_bars": template.entry_delay_bars,
        },
        "entry": {"pos": entry_pos, "price": entry},
        "long_path": {
            "mfe_pct": long_mfe,
            "mae_pct": long_mae,
            "close_ret_pct": long_close,
            "rr_mfe_to_mae": long_rr,
            "first_event": long_hit["first_event"],
            "first_event_bars": long_hit["bars"],
            "grade": long_grade,
        },
        "short_path": {
            "mfe_pct": short_mfe,
            "mae_pct": short_mae,
            "close_ret_pct": short_close,
            "rr_mfe_to_mae": short_rr,
            "first_event": short_hit["first_event"],
            "first_event_bars": short_hit["bars"],
            "grade": short_grade,
        },
        "direction_pressure": pressure,
    }


def build_path_shape_rows(*, value_jsonl: str, market_csv: str, output: str, horizon_bars: int, target_pct: float, stop_pct: float, entry_delay_bars: int = 1) -> dict[str, Any]:
    market = pd.read_csv(market_csv)
    best = best_rows_by_signal(load_jsonl(value_jsonl))
    template = PathTemplate(horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct, entry_delay_bars=entry_delay_bars)
    out = []
    pressure_counts: dict[str, int] = {}
    skipped = 0
    for row in best:
        shape = compute_path_shape(market, int(row.get("signal_pos", -1)), template)
        if shape is None:
            skipped += 1
            continue
        pressure_counts[shape["direction_pressure"]] = pressure_counts.get(shape["direction_pressure"], 0) + 1
        out.append({
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prompt": row.get("prompt"),
            "analyzer_target": shape,
            "trader_template": {
                "allowed_actions": [
                    {"gate": "TRADE", "side": "LONG", "target_pct": target_pct, "stop_pct": stop_pct, "max_hold_bars": horizon_bars},
                    {"gate": "TRADE", "side": "SHORT", "target_pct": target_pct, "stop_pct": stop_pct, "max_hold_bars": horizon_bars},
                    {"gate": "NO_TRADE", "side": "NONE", "target_pct": 0.0, "stop_pct": 0.0, "max_hold_bars": 0},
                ],
                "preferred_pressure": shape["direction_pressure"],
            },
        })
    write_jsonl(output, out)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "source": value_jsonl,
        "output": output,
        "rows": len(out),
        "skipped": skipped,
        "pressure_counts": dict(sorted(pressure_counts.items())),
        "template": template.__dict__,
        "leakage_note": "Targets are future path labels for supervised/RL training data generation; split discipline must still train on train and select on val before OOS reporting.",
    }
    Path(output).with_suffix(".summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--value-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--horizon-bars", type=int, default=144)
    p.add_argument("--target-pct", type=float, default=1.0)
    p.add_argument("--stop-pct", type=float, default=0.6)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    return p.parse_args()


def main() -> None:
    print(json.dumps(build_path_shape_rows(**vars(parse_args())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
