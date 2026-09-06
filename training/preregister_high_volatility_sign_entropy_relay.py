"""Outcome-blind preregistration for HVSER-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_sign_entropy_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_sign_entropy_relay_v1",
        "policy_id": "HVSER-12",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "mechanism": {
            "claim": (
                "During high realized range volatility, unusually low binary entropy of completed "
                "five-minute return signs over 24 hours marks concentrated directional price discovery "
                "whose net direction can persist for another 12 hours."
            ),
            "side": "sign of the completed trailing 24-hour open-to-open return",
            "why_distinct": (
                "HVSER uses the full 288-observation sign distribution rather than HVDPR's two "
                "four-hour agreement, displacement, path efficiency, or implied-volatility levels. "
                "It does not alter or reuse a terminal rule, threshold, control, incidence, or outcome."
            ),
        },
        "clock": {
            "decision": "fixed six-hour anchors from position 143; all features use completed bars only",
            "trailing_window_bars": 288,
            "sign_entropy": "binary Shannon entropy in bits over nonzero 5m open-to-open return signs; at least 276 valid signs",
            "direction": "sign of open(anchor)/open(anchor-288)-1; zero invalid",
            "calibration": (
                "on 2023H1 anchors only, freeze sign-entropy q35 and range_vol q65; calibration "
                "uses no forward return, execution price, or PnL"
            ),
            "eligibility": "sign_entropy <= frozen q35 and range_vol >= frozen q65",
            "entry": "anchor completed bar plus one 5m open",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
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
            "names": ["no_entropy_gate", "no_volatility_gate", "one_anchor_stale_features", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and novelty pass",
        },
        "research_boundary": {
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "freeze, open source incidence, then novelty and sequential economics; terminal first failure",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
