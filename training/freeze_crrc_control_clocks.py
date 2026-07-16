"""Freeze CRRC-72 mechanism-control clocks before opening 2023 outcomes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import qualify_cross_venue_radial_refill_compression as qualify


SUPPORT = Path(
    "results/cross_venue_radial_refill_compression_support_2026-07-17.json"
)
SUPPORT_SHA256 = "b5fc746bbc2e82e1ceaf57e61efa709ad084e1db74d4769da21c9ee345996e95"
PRIMARY_CLOCK = Path(
    "results/cross_venue_radial_refill_compression_event_clock_2026-07-17.json"
)
PRIMARY_CLOCK_SHA256 = "09d2ca954c5c4d06b981575c6b0f0e4dc6b49d8a693da418f3f26e5cc454c835"
DEFAULT_OUTPUT = Path(
    "results/cross_venue_radial_refill_compression_control_clocks_2026-07-17.json"
)

CONTROL_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "um_only": {"venues": ("um",), "metrics": ("add", "withdraw", "net", "flicker")},
    "cm_only": {"venues": ("cm",), "metrics": ("add", "withdraw", "net", "flicker")},
    "without_credibility": {"venues": ("um", "cm"), "metrics": ("add", "withdraw")},
    "inner_add_only": {"venues": ("um", "cm"), "metrics": ("add",)},
    "outer_withdraw_only": {"venues": ("um", "cm"), "metrics": ("withdraw",)},
}


def _metric_pass(
    raw: dict[tuple[str, str, str], pd.Series],
    thresholds: dict[tuple[str, str, str], pd.Series],
    complete: pd.Series,
    *,
    venue: str,
    side: str,
    metrics: Iterable[str],
) -> pd.Series:
    passed = complete.astype(bool).copy()
    for metric in metrics:
        value = raw[(venue, side, metric)]
        threshold = thresholds[(venue, side, metric)]
        observed = (
            value.notna()
            & threshold.notna()
            & np.isfinite(value)
            & np.isfinite(threshold)
            & threshold.ne(0.0)
        )
        comparison = (
            value.le(threshold) if metric == "flicker" else value.ge(threshold)
        )
        passed &= observed & comparison
    return passed


def control_signal(
    dates: pd.Series,
    raw: dict[tuple[str, str, str], pd.Series],
    thresholds: dict[tuple[str, str, str], pd.Series],
    complete: pd.Series,
    name: str,
    cfg: qualify.Config,
) -> pd.DataFrame:
    if name not in CONTROL_SPECS:
        raise ValueError(f"unknown CRRC control: {name}")
    spec = CONTROL_SPECS[name]
    side_pass: dict[str, pd.Series] = {}
    for side in ("m", "p"):
        passed = complete.astype(bool).copy()
        for venue in spec["venues"]:
            passed &= _metric_pass(
                raw,
                thresholds,
                complete,
                venue=venue,
                side=side,
                metrics=spec["metrics"],
            )
        side_pass[side] = passed
    side = pd.Series(0, index=dates.index, dtype=np.int8)
    side.loc[side_pass["m"] & ~side_pass["p"]] = 1
    side.loc[side_pass["p"] & ~side_pass["m"]] = -1
    branch = pd.Series("none", index=dates.index, dtype="string")
    branch.loc[side.gt(0)] = f"{name}_bid"
    branch.loc[side.lt(0)] = f"{name}_ask"
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "candidate": side.ne(0),
            "side": side,
            "conflict": side_pass["m"] & side_pass["p"],
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(np.int16),
        }
    )


def _verify_dependencies() -> dict[str, Any]:
    if qualify.sha256(SUPPORT) != SUPPORT_SHA256:
        raise RuntimeError("CRRC support artifact changed")
    if qualify.sha256(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("CRRC primary event clock changed")
    support = json.loads(SUPPORT.read_text())
    primary = json.loads(PRIMARY_CLOCK.read_text())
    if support.get("all_support_gates_pass") is not True:
        raise RuntimeError("CRRC support did not pass")
    if support.get("protocol", {}).get("outcomes_opened_for_crrc72") is not False:
        raise RuntimeError("CRRC support opened outcomes")
    if primary.get("outcomes_opened") is not False or primary.get("event_count") != 156:
        raise RuntimeError("CRRC primary clock is not its frozen outcome-blind clock")
    if primary.get("event_clock_sha256") != support.get("event_clock_sha256"):
        raise RuntimeError("CRRC support-to-clock binding changed")
    return {
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "primary_event_clock_sha256": primary["event_clock_sha256"],
    }


def build() -> dict[str, Any]:
    dependencies = _verify_dependencies()
    cfg = qualify.Config()
    shells, credibility, source = qualify.load_sources(cfg)
    raw, complete = qualify.raw_features(shells, credibility)
    selected_cell = {
        "q_add": cfg.selected_q_add,
        "q_withdraw": cfg.selected_q_withdraw,
        "q_net": cfg.selected_q_net,
        "q_flicker": cfg.selected_q_flicker,
    }
    thresholds = qualify.thresholds_for_cell(raw, selected_cell, cfg)
    controls: dict[str, Any] = {}
    for name in CONTROL_SPECS:
        signal = control_signal(
            shells["date"], raw, thresholds, complete, name, cfg
        )
        schedule = qualify.quarter_schedule(signal, cfg)
        controls[name] = {
            "spec": {
                "venues": list(CONTROL_SPECS[name]["venues"]),
                "metrics": list(CONTROL_SPECS[name]["metrics"]),
            },
            "raw_candidates": int(signal["candidate"].sum()),
            "conflicts_flattened": int(signal["conflict"].sum()),
            "event_count": int(len(schedule)),
            "side_counts": {
                "long": int(schedule["side"].gt(0).sum()),
                "short": int(schedule["side"].lt(0).sum()),
            },
            "event_clock_sha256": qualify.event_clock_hash(schedule),
            "events": schedule.to_dict("records"),
        }
    return {
        "protocol": "CRRC-72 outcome-blind mechanism-control clock freeze v1",
        "outcomes_opened": False,
        "price_funding_return_pnl_or_equity_loaded": False,
        "selection_end_exclusive": str(qualify.SELECTION_END),
        "dependencies": dependencies,
        "source": source,
        "selected_thresholds": selected_cell,
        "clock": {
            "entry_delay_bars": cfg.entry_delay_bars,
            "hold_bars": cfg.hold_bars,
            "quarter_contained": True,
        },
        "controls_are_diagnostics_not_repair_candidates": True,
        "controls": controls,
    }


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError("refusing to overwrite frozen CRRC control clocks")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if args.output != str(DEFAULT_OUTPUT):
        raise ValueError("CRRC control-clock freeze path is immutable")
    payload = build()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "outcomes_opened": False,
                "controls": {
                    name: {
                        "events": row["event_count"],
                        "side_counts": row["side_counts"],
                        "event_clock_sha256": row["event_clock_sha256"],
                    }
                    for name, row in payload["controls"].items()
                },
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
