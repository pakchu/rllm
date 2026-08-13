"""Outcome-blind preregistration for HVPIFSR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVPIFSR-8"
SLUG = "high_volatility_premium_implied_funding_surprise_reversal"
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_premium_implied_funding_surprise_reversal_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "At an actual BTCUSDT funding settlement, compare the realized funding rate with "
                "the rate mechanically implied by the completed time-weighted premium-index path. "
                "An upper-tail discrepancy is a last-mile leverage-pricing surprise not explained "
                "by the visible premium path; in elevated realized variation, fade its sign for "
                "the next funding cycle."
            ),
            "side": "negative strict sign of realized funding minus premium-path-implied funding",
            "why_distinct": (
                "HVEFR subtracts a rolling median of realized funding, HVPFPR uses premium/price "
                "direction without reading funding, and HVFPSC fits a pre-2023 multi-feature outcome "
                "classifier. HVPIFSR instead applies the published interest-plus-clamp funding map "
                "to one completed premium path and trades only the unexplained settlement residual. "
                "It uses no price direction, OI, cross-alt source, fitted outcome, prior event set, "
                "or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "the completed prior-24-hour BTC realized variation must rank in its causal upper "
                "35 percent, selecting July-like volatile funding cycles"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "premium-implied settlement-surprise onsets are absent from Gross9 primitives"
            ),
        },
        "features": {
            "decision_grid": "actual BTCUSDT funding_time S",
            "premium_path": (
                "480 exact coherent finite bars_binance_premium one-minute closes on [S-8h,S); "
                "no interpolation or forward fill"
            ),
            "premium_average": (
                "linearly time-weighted mean of the 480 closes with deterministic integer weights "
                "1..480, giving later observations greater weight"
            ),
            "implied_funding_proxy": (
                "premium_average + clip(0.0001-premium_average,-0.0005,+0.0005); this is explicitly "
                "a one-minute replay proxy because the exchange sampling implementation can change"
            ),
            "realized_funding": (
                "exact finite funding_rates_binance BTCUSDT rate at S, available before entry"
            ),
            "funding_surprise": "realized_funding-implied_funding_proxy, finite strict nonzero",
            "surprise_rank": (
                "strict-prior midrank of abs(funding_surprise) over at most 270 earlier source-valid "
                "settlements, minimum 180, current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sqrt(sum squared close-to-close log returns) from 1,440 exact coherent BTCUSDT "
                "bars_binance one-minute closes on [S-24h,S)"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid settlements, minimum "
                "180, current excluded; rank>=0.65"
            ),
            "onset": (
                "eligible now and immediately preceding source-valid actual settlement ineligible; "
                "a missing predecessor cannot trigger"
            ),
            "causal_availability": (
                "the actual funding row and every premium/price minute end no later than S; entry is S+5m"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "actual funding settlement S",
            "entry": "exact BTCUSDT perpetual S+5m open",
            "side": "-sign(funding_surprise)",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal entry",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_accounting": (
                "signal settlement precedes entry; exact held-interval settlements open only after novelty"
            ),
        },
        "policy": {
            "premium_minutes": 480,
            "interest_per_eight_hours": 0.0001,
            "clamp_absolute": 0.0005,
            "history_settlements": 270,
            "minimum_history_settlements": 180,
            "surprise_rank_min": 0.75,
            "variation_rank_min": 0.65,
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
                "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable "
                "then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged all-stage pass",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_surprise_tail",
                "no_variation_gate",
                "unweighted_premium_mean",
                "one_settlement_stale_surprise",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "funding": {
                "table": "funding_rates_binance",
                "symbol": "BTCUSDT",
                "columns": ["funding_time", "funding_rate"],
                "actual_timestamps": True,
            },
            "premium": {
                "table": "bars_binance_premium",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "published_formula_metadata_opened_before_preregistration": True,
            "database_schema_and_coverage_metadata_only_opened_before_preregistration": True,
            "prior_funding_median_premium_pressure_and_classifier_family_outcomes_known": True,
            "repository_premium_implied_realized_funding_surprise_candidate_found": False,
            "prior_event_sets_or_controls_reused": False,
            "prior_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent settlement-pricing-surprise mechanism",
        },
        "stopping_rule": (
            "terminal first failure; no premium weighting, formula constants, history, rank, onset, "
            "variation, side, clock, hold, subset, threshold, comparator, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core) or payload != build():
        raise RuntimeError("HVPIFSR preregistration drift")
    if payload["outcomes_opened"] or payload["source_incidence_opened"] or payload["gross9_rows_opened"]:
        raise RuntimeError("HVPIFSR evidence boundary opened")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(args.output)


if __name__ == "__main__":
    main()
