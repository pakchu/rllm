"""Preregister HVCACLR-8 without opening source incidence or outcomes."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any

POLICY_ID = "HVCACLR-8"
SLUG = "high_volatility_cross_alt_close_location_consensus_relay"
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-11.json")
ALTS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_alt_close_location_consensus_relay_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-11",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When at least five of six liquid alt perpetuals finish completed eight-hour auctions "
                "on the same side of their own high-low midranges, broad cross-sectional closing "
                "pressure represents market-wide price acceptance rather than a BTC-local endpoint. "
                "During elevated BTC realized variation, follow a fresh upper-tail consensus in BTC "
                "for eight elapsed hours."
            ),
            "side": "strict sign of the cross-sectional median signed alt close-location value",
            "why_distinct": (
                "Cross-alt price candidates use endpoint returns, sign breadth, residuals, rank "
                "persistence, or return tails; HVCAWCR uses accumulated rejection wicks. HVCACLR "
                "instead uses each alt auction's terminal close position inside its completed full "
                "high-low range. It uses no return direction, wick sums, volume, flow, funding, OI, "
                "premium, fitted outcome, prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC completed trailing eight-hour realized variation must rank in its causal upper "
                "35%, while absolute cross-alt close-location consensus must enter its upper quartile."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "fresh hourly cross-alt auction-location consensus onsets are absent from Gross9 primitives"
            ),
        },
        "external_basis": {
            "fixed_definition": (
                "Chaikin-style Close Location Value CLV=(2C-H-L)/(H-L), applied separately to each "
                "completed eight-hour alt auction; zero-range or zero-CLV auctions are invalid"
            ),
            "selection_use": "the bounded close-location construction only; no outcome claim is imported",
        },
        "features": {
            "decision_grid": "every exact UTC hour D",
            "symbols": list(ALTS),
            "source_window": "480 exact coherent one-minute bars per alt over [D-8h,D)",
            "auction_ohlc": "first open, maximum high, minimum low, final close in the completed window",
            "signed_close_location": "(2*close-high-low)/(high-low), finite strict nonzero in [-1,1]",
            "consensus_polarity": "strict sign of the cross-sectional median of six signed close locations",
            "breadth": "at least five of six individual close-location signs equal consensus polarity",
            "consensus_strength": "absolute cross-sectional median signed close location",
            "strength_rank": (
                "strict-prior midrank over at most 2160 earlier source-valid hourly decisions, minimum "
                "1440, current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sum squared BTCUSDT one-minute log(close/open) returns over [D-8h,D), finite strict positive"
            ),
            "variation_rank": "strict-prior 2160/1440 midrank, current excluded; rank>=0.65",
            "eligible_state": "breadth, strength-rank, and variation-rank gates pass",
            "onset": (
                "eligible now and immediately previous exact source-valid hourly decision ineligible; "
                "missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all seven completed paths are available",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 2160,
            "minimum_history_blocks": 1440,
            "minimum_consensus_breadth": 5,
            "strength_rank_min": 0.75,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held "
                "5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_strength_tail",
                "no_variation_gate",
                "four_of_six_breadth",
                "equal_weight_close_location_sum",
                "one_hour_stale_geometry",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "symbols": ["BTCUSDT", *ALTS],
            "interval": "1m",
            "columns": ["ts", "symbol", "open", "high", "low", "close"],
            "query_window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_cross_alt_return_wick_and_close_location_outcomes_known": True,
            "repository_cross_alt_close_location_consensus_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent cross-alt terminal auction-location consensus mechanism",
        },
        "stopping_rule": (
            "Terminal first failure; no symbol basket, auction window, close-location formula, breadth, "
            "rank, onset, side, hold, clock, subset, threshold, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); payload = build(); args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != payload:
        raise RuntimeError(f"refusing overwrite of {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "manifest_hash": payload["manifest_hash"]}))


if __name__ == "__main__":
    main()
