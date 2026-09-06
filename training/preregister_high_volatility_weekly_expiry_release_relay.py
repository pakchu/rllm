"""Outcome-blind preregistration for HVWER-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_weekly_expiry_release_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_weekly_expiry_release_relay_v1",
        "policy_id": "HVWER-24",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A large directional BTC displacement into the recurring Friday 08:00 UTC "
                "crypto-options expiry, when the preceding day is also high range-volatility, "
                "contains expiry-hedging pressure that should partially unwind after expiry."
            ),
            "side": "opposite the completed pre-expiry 24-hour close displacement",
            "why_distinct": (
                "HVWER is a predetermined weekly derivatives-calendar release event. It is not a "
                "continuous price threshold, funding settlement rule, CBOE session signal, or a "
                "repair/control of any terminal high-volatility candidate."
            ),
            "calendar_basis": (
                "Friday 08:00 UTC is fixed before observing BTC data and is used only as a weekly "
                "expiry-release anchor; no contract-specific future expiry file is required."
            ),
        },
        "clock": {
            "decision": "each Friday 08:00:00 UTC using only completed bars through 07:55",
            "lookback": "the prior 288 completed contiguous 5m bars",
            "range_volatility": "max(high)/min(low)-1 over the frozen 24-hour lookback",
            "displacement": "last completed close / first lookback open - 1",
            "calibration": (
                "2023H1 Friday anchors only: range-volatility q60 and absolute-displacement q50; "
                "source incidence and all post-entry prices remain unopened"
            ),
            "eligibility": (
                "range-volatility >= frozen q60 and absolute displacement >= frozen q50; "
                "zero displacement is ineligible"
            ),
            "entry": "the exact Friday 08:00 UTC open after the completed 07:55 bar",
            "hold": "24 elapsed hours to Saturday 08:00 UTC",
            "reservation": "global half-open; weekly anchors cannot overlap at the frozen hold",
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
                "no_volatility_gate",
                "no_displacement_gate",
                "thursday_same_time_anchor",
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
            "exact_weekly_expiry_release_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no calendar, threshold, side, hold, or gate repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVWER preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVWER source drift: {raw}")


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
