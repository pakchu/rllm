"""Write the outcome-blind DPSLR-24 preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/debt_public_supply_liquidity_relay_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def payload() -> dict[str, Any]:
    core = {
        "protocol_version": "debt_public_supply_liquidity_relay_v1",
        "policy_id": "DPSLR-24",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "An unusually large increase in official US debt held by the public expands the "
                "Treasury collateral stock and can absorb private liquidity, while a decrease releases "
                "balance-sheet capacity. During high BTC variation, positive supply shocks map short "
                "BTC and negative shocks map long BTC for the next day."
            ),
            "side": "opposite strict sign of the standardized log change in debt held by the public",
            "why_distinct": (
                "DFFB used category breadth inside Daily Treasury Statement cash flows; Treasury auction "
                "candidates used issue-specific demand; TPYSR/TRTSR used yield curves. DPSLR uses the "
                "separate Debt to the Penny public-debt stock and promotes no prior control."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed pre-entry BTC 24-hour realized variation must rank in its causal upper 35%."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "a seven-calendar-day-delayed official fiscal stock clock is absent from Gross9 primitives."
            ),
        },
        "features": {
            "source": "US Treasury Fiscal Data Debt to the Penny API",
            "dataset_page": "https://fiscaldata.treasury.gov/datasets/debt-to-the-penny/",
            "api_endpoint": (
                "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/"
                "od/debt_to_penny"
            ),
            "observation": (
                "record_date, debt_held_public_amt, intragov_hold_amt, and tot_pub_debt_out_amt"
            ),
            "official_schedule": (
                "dataset reported daily and updated at the end of each business day with previous-business-day data"
            ),
            "availability": (
                "conservatively record_date + 7 calendar days at 00:00 UTC; no earlier use even when "
                "the endpoint would normally update sooner"
            ),
            "valid_transition": (
                "current and previous records have unique increasing dates 1 to 5 calendar days apart "
                "and all three debt amounts are finite positive with total equal to public plus intragovernmental"
            ),
            "public_supply_change": (
                "log(current debt_held_public_amt / previous debt_held_public_amt); strict nonzero"
            ),
            "standardization": (
                "public-supply change standardized against at most 90 strictly prior valid changes, "
                "minimum 60, current excluded, sample std positive"
            ),
            "magnitude_rank": (
                "strict-prior midrank of absolute standardized change over at most 90 valid standardized "
                "changes, minimum 60, current excluded; rank>=0.70"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared exact completed hourly BTC log returns over [decision-24h,decision))"
            ),
            "volatility_rank": (
                "strict-prior midrank over at most 90 valid source decisions, minimum 60, current excluded; "
                "rank>=0.65"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact record_date + 7 calendar days at 00:00 UTC",
            "entry": "exact decision + 5 minutes at BTCUSDT open",
            "hold": "24 elapsed hours",
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "total_public_debt_change", "intragovernmental_change", "no_magnitude_tail",
                "no_volatility_gate", "one_observation_stale_supply_change", "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "fiscal_data": (
                "download and hash-bind the official API response for 2023-01-01 through 2026-07-24 "
                "only after this preregistration commit; the end date permits the fixed seven-day delay"
            ),
            "completed_btc": "hash-bound completed-hour BTC source through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_treasury_cashflow_auction_and_yield_outcomes_known": True,
            "debt_to_the_penny_rows_opened": False,
            "prior_candidate_outcomes_used_to_set_dpslr_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent official public-debt stock supply",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no "
            "field, lag, rank, side, hold, availability, volatility, or subset repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload(), indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
