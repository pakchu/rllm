"""Outcome-blind preregistration for HVVAR-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVVAR-8"
DEFAULT_OUTPUT = Path("results/high_volatility_value_area_rejection_relay_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_value_area_rejection_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": "In an elevated-variation BTC auction, a final-hour excursion outside the completed trailing twenty-four-hour quote-volume profile's contiguous seventy-percent value area that closes back inside accepted value is a rejected price-discovery attempt; fade the rejected side for eight elapsed hours at the onset of that state.",
            "side": "short after an upper-value-area rejection and long after a lower-value-area rejection",
            "why_distinct": "HVVAR builds a fixed-bin volume-at-price distribution and a contiguous value area around its point of control. Weighted-median migration compares two scalar centers, dominant-volume rules select one time bar, close-VWAP rules compare an endpoint with one mean, and range-rejection rules do not condition on volume-at-price acceptance. HVVAR uses no fitted outcome, prior event set, control, OI, funding, or cross-asset input.",
            "why_suited_to_volatile_regimes": "the completed trailing twenty-four-hour realized variation must rank in its causal upper thirty-five percent, while the rejected excursion directly identifies failed discovery during turbulent trade",
            "why_low_gross9_overlap_is_plausible": "sparse four-hour volume-profile rejection onsets with an eight-hour reservation are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "auction-market volume-profile value-area construction",
            "fixed_definition": "partition log price into twenty-four equal bins, locate the maximum quote-volume point of control, and expand contiguously toward the larger adjacent volume until at least seventy percent of quote volume is enclosed",
            "selection_use": "the conventional point-of-control and contiguous value-area construction only; no directional return, incidence, or outcome claim is imported",
        },
        features={
            "decision_grid": "every exact four-hour UTC boundary",
            "profile_window": "288 exact consecutive completed five-minute BTCUSDT bars [T-24h,T)",
            "five_minute_bar": "exact aggregation of five unique coherent one-minute rows; OHLC finite positive; base and quote volume finite strict positive",
            "bar_value_price": "sum quote_asset_volume divided by sum volume over each five-minute bar, finite strict positive",
            "profile_range": "log(minimum low) through log(maximum high) over the 288 bars, strict positive width",
            "profile_bins": "twenty-four equal-width log-price bins; each bar's full quote volume is assigned by its log bar-value price; left-closed right-open except the final bin includes the upper endpoint",
            "point_of_control": "bin with maximum quote volume; ties select the lower-price bin",
            "value_area": "start at the point of control and repeatedly add the available adjacent bin with larger quote volume, ties selecting the lower-price bin, until cumulative enclosed quote volume is at least seventy percent of total; lower and upper bin edges define accepted value",
            "final_hour": "the final twelve completed five-minute bars [T-1h,T)",
            "upper_rejection": "final-hour maximum high is strictly above the value-area upper edge, final-hour minimum low is not below the lower edge, and final completed close lies inside the closed value area",
            "lower_rejection": "final-hour minimum low is strictly below the value-area lower edge, final-hour maximum high is not above the upper edge, and final completed close lies inside the closed value area",
            "rejection_state": "exactly one of upper_rejection or lower_rejection is true",
            "variation": "sqrt(sum squared close-to-close log returns across the 288 completed five-minute bars), finite strict positive",
            "variation_rank": "strict-prior midrank over at most 180 earlier source-valid four-hour decisions, minimum 120, current excluded; rank>=0.65",
            "onset": "current rejection_state and variation gate are true after the immediately preceding exact source-valid decision was not eligible; a missing prior opportunity cannot trigger",
            "no_imputation": True,
        },
        clock={
            "feature_available": "four-hour boundary after all profile and final-hour bars complete",
            "entry": "exact BTCUSDT open five elapsed minutes later",
            "side": "-1 for upper rejection; +1 for lower rejection",
            "hold": "8 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "bar_minutes": 5,
            "profile_bars": 288,
            "decision_hours": 4,
            "profile_bins": 24,
            "value_area_share": 0.70,
            "final_hour_bars": 12,
            "variation_history_decisions": 180,
            "minimum_history_decisions": 120,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": ["no_variation_gate", "no_onset", "direction_flip", "value_area_68pct", "forced_long"],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "volume", "quote_asset_volume"],
                "window": ["2023-05-25T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "auction_market_value_area_definition_read": True,
            "repository_exact_value_area_rejection_candidate_found": False,
            "adjacent_weighted_median_dominant_volume_and_range_rejection_candidates_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed auction-market contiguous seventy-percent value-area rejection under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no profile window, bin count, value-area share, final-hour definition, variation rank, onset, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVVAR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
