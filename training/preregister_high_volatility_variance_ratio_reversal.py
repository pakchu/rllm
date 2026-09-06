"""Outcome-blind preregistration for HVVRR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_variance_concentration_release as base


DEFAULT_OUTPUT = Path(
    "results/high_volatility_variance_ratio_reversal_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = base.SOURCE_BINDINGS


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def build() -> dict[str, Any]:
    payload = base.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    core.update(
        {
            "protocol_version": "high_volatility_variance_ratio_reversal_v1",
            "policy_id": "HVVRR-12",
            "mechanism": {
                "claim": (
                    "During an already high-volatility BTC day, an extreme excess of six-hour "
                    "five-minute realized variance over variance measured from six non-overlapping "
                    "one-hour returns identifies offsetting microstructure noise; the net six-hour "
                    "move should reverse over twelve hours."
                ),
                "side": "opposite the strict nonzero completed six-hour return",
                "why_distinct": (
                    "HVVRR measures cross-scale variance non-additivity. It does not use temporal "
                    "variance concentration/release, jump scores, sign entropy, semivariance, candle "
                    "wicks, external feeds, or a prior candidate control."
                ),
            },
            "clock": {
                "decision": "each completed 5m bar; accepted onset enters at the next exact 5m open",
                "fine_variance": "sum squared close-to-close log returns over prior 72 completed returns",
                "coarse_variance": (
                    "sum squared returns of the six consecutive non-overlapping 12-bar blocks ending "
                    "at the same decision bar"
                ),
                "noise_ratio": "fine_variance/coarse_variance; coarse variance must be positive",
                "high_volatility": "prior 288 completed-bar high/low-1",
                "calibration": (
                    "2023H1 source-only high-volatility q60 and noise-ratio q85; no candidate incidence "
                    "or post-entry prices"
                ),
                "onset": (
                    "false-to-true noise ratio >= q85 while high-volatility >= q60 and the completed "
                    "six-hour return is finite and nonzero"
                ),
                "hold": "12 elapsed hours",
                "reservation": "global half-open; exit first on equal open",
                "split_crossing_action": "skip",
                "gross_exposure": 0.5,
            },
            "diagnostic_controls": {
                "names": [
                    "no_high_volatility_gate",
                    "no_noise_ratio_gate",
                    "one_bar_stale_variance_ratio",
                    "direction_flip",
                ],
                "diagnostic_controls_cannot_be_promoted": True,
            },
            "research_boundary": {
                "exact_cross_scale_variance_ratio_outcomes_known": False,
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
        raise RuntimeError("HVVRR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVVRR source drift: {raw}")


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
