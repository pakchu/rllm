"""Outcome-blind preregistration for CAFACR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "CAFACR-8"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
DEFAULT_OUTPUT = Path(
    "results/cross_alt_funding_acceleration_contradiction_relay_preregistration_2026-08-09.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_alt_funding_acceleration_contradiction_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When funding changes across at least four of six liquid alt perpetuals share "
                "one sign while BTC funding changes in the opposite direction, broad leveraged "
                "crowding is contradicted by the BTC venue state. Fade the alt funding-acceleration "
                "majority on BTC during the next settlement interval."
            ),
            "side": "negative strict majority sign of the six alt funding-rate changes",
            "why_distinct": (
                "CAFACR uses cross-sectional changes in six realized alt funding rates plus an "
                "opposing BTC funding change. It uses no BTC price path, MFDH monotone level path, "
                "STCR trend, analog model, order flow, OI, implied volatility, or Gross9 state."
            ),
            "volatile_market_target": (
                "broad cross-alt leveraged acceleration contradicted by BTC is a crowding-release "
                "mechanism; exact BTC RV20 q90 remains a later audit rather than an entry filter"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "only common funding settlements with cross-sectional acceleration contradiction qualify"
            ),
        },
        "features": {
            "universe": ["BTCUSDT", *ALTS],
            "common_settlement": (
                "all seven symbols have exactly one finite funding row at S and at the immediately "
                "preceding common settlement S-8h"
            ),
            "change": "delta_i=funding_rate_i(S)-funding_rate_i(S-8h)",
            "alt_majority": (
                "at least four of six alt deltas are strictly positive or at least four are strictly "
                "negative; a 3-3 tie or all-zero majority is ineligible"
            ),
            "btc_contradiction": "BTC delta is nonzero and has the opposite sign to the alt majority",
            "availability": (
                "both completed settlement rows must be available at S; missing, duplicate, late, "
                "non-finite, or nonconsecutive rows make S ineligible; no imputation"
            ),
        },
        "rv20_stress_slice": {
            "daily_return": "log(C_d/C_{d-1}) assigned to UTC calendar day d",
            "rv20": "sqrt(365*mean(r_d^2)) over exact returns t-20 through t-1",
            "threshold": (
                "numpy linear 0.90 quantile over 756 available RV20 observations with decision "
                "timestamps strictly before t; current excluded"
            ),
            "active": "RV20(t)>=threshold(t)",
            "entry_filter": False,
            "source_stage_opened": False,
            "future_use": "opened only after every full-calendar economic stage passes",
        },
        "clock": {
            "decision": "actual common funding settlement S",
            "entry": "exact BTCUSDT S+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_accounting": "exact realized BTC funding during [entry,exit) only after novelty",
        },
        "policy": {
            "alt_majority_minimum": 4,
            "settlement_gap_hours": 8,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
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
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "persistent_long_vol_comparator": "same accepted clock and 0.5 gross, side forced long",
            "full_calendar_decomposition": "candidate net return minus comparator net return",
            "rv20_q90_decomposition": "same decomposition restricted by causal RV20 at decision",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_btc_contradiction",
                "btc_change_only",
                "follow_alt_majority",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "funding": {
                "table": "funding_rates_binance",
                "symbols": ["BTCUSDT", *ALTS],
                "columns": ["symbol", "funding_time", "funding_rate"],
                "read_after_preregistration": True,
            },
            "btc_price_or_execution": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "exact_cafacr_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "btc_price_rows_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no universe, majority, contradiction, direction, settlement, "
            "hold, RV20, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CAFACR preregistration hash mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
