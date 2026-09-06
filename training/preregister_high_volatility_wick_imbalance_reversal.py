"""Outcome-blind preregistration for HVWIR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_wick_imbalance_reversal_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_wick_imbalance_reversal_v1",
        "policy_id": "HVWIR-8",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "An unusually wide completed four-hour BTC candle with a strongly dominant rejection "
                "wick records failed directional price discovery; entering toward the rejected wick "
                "at the next four-hour boundary should capture an eight-hour inventory reversal."
            ),
            "side": "long for dominant lower wick; short for dominant upper wick",
            "why_distinct": (
                "HVWIR uses fixed four-hour auction morphology and no order-flow, OI, funding, options, "
                "rolling range boundary, or continuous price-crossing state. It is not a threshold, "
                "side, or hold repair of a terminal candidate or a promoted diagnostic control."
            ),
        },
        "clock": {
            "bar": "fixed UTC 4h bars aggregated from 48 contiguous completed 5m bars",
            "decision": "at each exact 00/04/08/12/16/20 UTC boundary after the 4h bar completes",
            "range": "high/low-1",
            "upper_wick": "high-max(open,close), normalized by open",
            "lower_wick": "min(open,close)-low, normalized by open",
            "wick_imbalance": "abs(upper_wick-lower_wick)/(upper_wick+lower_wick)",
            "calibration": (
                "2023H1 source-only four-hour bars: range q70 and wick-imbalance q70; no candidate "
                "incidence, post-entry return, execution price, or PnL"
            ),
            "eligibility": (
                "range >= frozen q70, wick imbalance >= frozen q70, both wicks finite with positive "
                "total wick, and unequal wicks"
            ),
            "entry": "decision boundary exact 5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first at an equal entry boundary",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "no_high_range_gate",
                "no_wick_imbalance_gate",
                "one_anchor_stale_morphology",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "exact_four_hour_wick_reversal_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no bar, threshold, side, hold, or gate repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVWIR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVWIR source drift: {raw}")


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
