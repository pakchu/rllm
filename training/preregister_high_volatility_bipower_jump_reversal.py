"""Outcome-blind preregistration for HVBJR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_variance_concentration_release as base


DEFAULT_OUTPUT = Path(
    "results/high_volatility_bipower_jump_reversal_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = base.SOURCE_BINDINGS


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def build() -> dict[str, Any]:
    payload = base.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    core.update(
        {
            "protocol_version": "high_volatility_bipower_jump_reversal_v1",
            "policy_id": "HVBJR-12",
            "mechanism": {
                "claim": (
                    "A five-minute BTC return that is extreme relative to strictly prior 24-hour "
                    "bipower variation, while the prior day is already high-volatility, reflects "
                    "urgent discontinuous inventory transfer and should reverse over twelve hours."
                ),
                "side": "opposite the strict nonzero sign of the completed jump return",
                "why_distinct": (
                    "HVBJR isolates discontinuous return innovation relative to robust bipower "
                    "variation. It does not use variance concentration/release, fixed candle wicks, "
                    "calendar anchors, order flow, OI, funding, or a prior candidate control."
                ),
            },
            "clock": {
                "decision": "each completed 5m bar; accepted jump enters at the next exact 5m open",
                "return": "current completed close-to-close log return",
                "bipower_variation": (
                    "pi/2 times the sum of adjacent absolute-return products over the strictly prior "
                    "288 completed returns, excluding the current return"
                ),
                "jump_score": "abs(current return)/sqrt(prior bipower variation/288)",
                "high_volatility": "strictly prior 24h high/low-1",
                "calibration": (
                    "2023H1 source-only high-volatility q60 and jump-score q99; no candidate incidence "
                    "or post-entry prices"
                ),
                "onset": (
                    "false-to-true jump score >= frozen q99 while prior high-volatility >= frozen q60; "
                    "current return must be finite and nonzero"
                ),
                "entry": "next exact 5m open",
                "hold": "12 elapsed hours",
                "reservation": "global half-open; exit first on equal open",
                "split_crossing_action": "skip",
                "gross_exposure": 0.5,
            },
            "diagnostic_controls": {
                "names": [
                    "no_high_volatility_gate",
                    "raw_return_tail_instead_of_bipower",
                    "one_bar_stale_jump_inputs",
                    "direction_flip",
                ],
                "diagnostic_controls_cannot_be_promoted": True,
            },
            "research_boundary": {
                "exact_bipower_jump_reversal_outcomes_known": False,
                "candidate_incidence_opened": False,
                "postentry_return_or_pnl_opened": False,
                "gross9_rows_opened": False,
                "candidate_count": 1,
                "grid": False,
                "repair_of_prior_candidate": False,
                "promoted_prior_control": False,
            },
            "stopping_rule": "terminal first failure; no threshold, side, hold, or gate repair",
        }
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVBJR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVBJR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
