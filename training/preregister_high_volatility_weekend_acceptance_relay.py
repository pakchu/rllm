"""Outcome-blind preregistration for HVWAR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVWAR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_weekend_acceptance_relay_preregistration_2026-08-09.json")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_weekend_acceptance_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-09", "singleton": True,
        "current_candidate_outcomes_opened": False, "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "A large Saturday-Sunday BTC range that closes in the directional outer quartile has been accepted, rather than rejected, by continuous crypto liquidity. Follow that accepted weekend repricing through the first full Monday UTC day when broader participation returns.",
            "side": "strict sign of the completed 48-hour weekend displacement",
            "why_distinct": "WHVOF fades incomplete 12-hour moves inside the weekend. WMLHR waits through Monday 08:00 and requires an opposite-direction liquidity handoff. HVWAR waits for the entire weekend, requires terminal range acceptance, enters before Monday price information, and follows rather than repairs either event law.",
            "volatile_market_target": "completed weekend high-low range rank at least 0.65 against strictly prior weekends",
            "why_low_gross9_overlap_is_plausible": "at most one Monday 00:05 UTC position per week and a weekend-only acceptance state are absent from Gross9",
        },
        "features": {
            "weekend": "exact 576 completed contiguous BTCUSDT 5m bars [Saturday 00:00, Monday 00:00)",
            "validity": "unique exact grid, finite positive coherent OHLC, no imputation",
            "displacement": "log(Sunday 23:55 close / Saturday 00:00 open), strict nonzero",
            "range": "log(max high / min low), finite strict positive",
            "close_location": "(final close-min low)/(max high-min low)",
            "directional_acceptance": "close_location>=0.75 for positive displacement or <=0.25 for negative displacement",
            "range_rank": "strict-prior midrank over at most 90 valid completed weekends, minimum 60, current excluded; rank>=0.65",
            "no_monday_price_or_volume_input": True, "no_imputation": True,
        },
        "clock": {
            "decision": "exact Monday 00:00 UTC after Sunday 23:55 bar completes",
            "entry": "exact Monday 00:05 UTC open", "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip",
            "gross_exposure": .5, "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        "policy": {
            "prior_weekends": 90, "prior_minimum_weekends": 60, "range_rank_min": .65,
            "upper_close_location_min": .75, "lower_close_location_max": .25,
            "entry_delay_minutes": 5, "hold_hours": 24, "leverage": .5,
            "base_cost_per_notional_side": .0006, "stress_cost_per_notional_side": .001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": .2, "max_month_share": .45},
        "novelty_gates": {"exact_entry_jaccard_max": .1, "candidate_near_6h_share_max": .35, "occupied_5m_bar_jaccard_max": .25, "absolute_signed_exposure_pearson_max": .35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3., "strict_mdd_max_pct": 15., "mean_gross_underlying_min_bp": 20., "weekly_signflip_one_sided_p_max": .1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes train, test, eval, final", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_range_gate", "no_acceptance_gate", "central_close_rejection", "sunday_only_geometry", "direction_flip", "same_clock_forced_long"], "cannot_be_promoted": True},
        "source_plan": {"historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_weekend_family_outcomes_known": True, "prior_event_sets_or_controls_reused": False, "exact_hvwar_incidence_or_outcomes_known": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent full-weekend terminal acceptance mechanism"},
        "stopping_rule": "terminal first failure; no weekend, rank, acceptance, side, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVWAR preregistration drift")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVWAR market binding drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
