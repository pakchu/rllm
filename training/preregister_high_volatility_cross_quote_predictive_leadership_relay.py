"""Preregister one outcome-blind cross-quote predictive-leadership alpha."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVQPLR-6"
DEFAULT_OUTPUT = Path("results/high_volatility_cross_quote_predictive_leadership_relay_preregistration_2026-08-11.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_cross_quote_predictive_leadership_relay_v1",
        "policy_id": POLICY_ID, "as_of_date": "2026-08-11", "singleton": True,
        "outcomes_opened": False, "source_incidence_opened": False, "gross9_rows_opened": False,
        "mechanism": {
            "claim": "When one Binance BTC spot stablecoin-quote book uniquely leads the next-minute common return of the other books, price discovery is concentrated rather than simultaneous noise. During elevated BTC variation, follow the concentrated leader only when its final-two-hour direction agrees with the common eight-hour displacement.",
            "side": "strict common sign of the leader book final-two-hour return and the three-book median eight-hour return",
            "why_distinct": "Stablecoin-flow candidates use taker imbalance, participation, or sequential flow. Dominant-quote disagreement uses contemporaneous return disagreement. Spot/perpetual error-correction uses a two-venue basis. HVQPLR uses only lagged predictive price leadership among three spot quote books and no prior event set or control.",
            "why_suited_to_volatile_regimes": "the completed BTCUSDT eight-hour realized variation must rank in its causal upper 35%",
            "why_low_gross9_overlap_is_plausible": "three fixed daily cross-quote predictive-leadership clocks are absent from Gross9",
        },
        "features": {
            "books": ["BTCUSDT", "BTCUSDC", "BTCFDUSD"], "venue": "Binance spot", "interval": "1m",
            "window": "exact 480 completed one-minute bars [D-8h,D) at D in 00:00/08:00/16:00 UTC",
            "return": "log(close_t/close_t-1) on each exact grid; first return uses the exact D-8h-1m close",
            "book_target": "for book j, median of the other two books' contemporaneous one-minute returns",
            "predictor": "book j return lagged exactly one minute",
            "lead_score": "max((SSE intercept-only - SSE intercept+slope)/SSE intercept-only,0) over the 479 aligned observations; nonpositive denominator or nonfinite fit invalid",
            "lead_share": "unique largest lead_score divided by sum of all three positive lead_scores; strict >=0.60; ties invalid",
            "direction_confirmation": "leader final 120-minute log return and median of all three eight-hour log returns have the same strict nonzero sign",
            "btc_variation": "sqrt(sum squared BTCUSDT one-minute log returns across the exact completed eight-hour window)",
            "variation_rank": "strict-prior midrank over at most 270 earlier fixed-boundary valid states, minimum 180, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "D after all source bars through D-1m complete", "entry": "exact D+5m BTCUSDT perpetual open",
            "hold": "6 elapsed hours", "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip",
            "gross_exposure": 0.5, "funding": "not a signal input; exact held settlements only after novelty",
            "rv20": "q90 audit only after unchanged all-stage pass",
        },
        "policy": {"window_minutes": 480, "direction_minutes": 120, "lead_share_min": 0.60, "variation_history_states": 270, "variation_minimum_states": 180, "variation_rank_min": 0.65, "entry_delay_minutes": 5, "hold_hours": 6, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_lead_concentration_gate", "no_variation_gate", "contemporaneous_instead_of_lagged", "one_block_stale_leader", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"spot": {"table": "bars_binance_spot", "symbols": ["BTCUSDT", "BTCUSDC", "BTCFDUSD"], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "read_after_preregistration": True}, "execution_prices": "sealed until source support and Gross9 novelty pass"},
        "research_boundary": {"published_basis": "price-discovery leadership is measured with lagged predictive contribution; this exact BTC adaptation is unpublished", "repository_information_share_or_cross_quote_predictive_leadership_found": False, "prior_event_sets_or_controls_reused": False, "prior_outcomes_used_to_set_formula_threshold_side_hold_or_clock": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-quote price-discovery concentration mechanism"},
        "stopping_rule": "Terminal first failure; no book, window, regression, threshold, variation, side, hold, clock, subset, or control repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core): raise RuntimeError("HVQPLR manifest drift")
    if payload["outcomes_opened"] or payload["source_incidence_opened"] or payload["gross9_rows_opened"]: raise RuntimeError("HVQPLR evidence boundary opened")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    payload = build(); validate(payload); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n"); print(args.output)


if __name__ == "__main__": main()
