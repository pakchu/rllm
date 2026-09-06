"""Outcome-blind preregistration for HVSPER-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVSPER-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_spot_perpetual_error_correction_relay_"
    "preregistration_2026-08-10.json"
)


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
        "protocol_version": "high_volatility_spot_perpetual_error_correction_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Cointegrated Bitcoin spot and perpetual prices need not incorporate information at "
                "the same speed. When a strictly prior error-correction model identifies the "
                "perpetual as the price-discovery leader, an unusually large same-direction "
                "one-hour perpetual-minus-spot return innovation during elevated realized variation "
                "marks information not yet fully incorporated by the cash market. Follow the "
                "perpetual innovation direction for eight hours."
            ),
            "side": "strict sign of the completed one-hour BTC perpetual return",
            "why_distinct": (
                "HVSPER estimates two causal error-correction equations for synchronized Binance "
                "spot and perpetual prices and conditions on the relative adjustment loadings. "
                "Prior spot/perpetual candidates use levels, participation, contemporaneous return "
                "gaps, premium tails, simple lead correlations, or supervised residual classifiers; "
                "the repository contains no VECM, cointegration, Hasbrouck, or adjustment-loading "
                "candidate. No prior event set, model output, threshold, control, OI, funding, or "
                "cross-asset input is reused."
            ),
            "why_suited_to_volatile_regimes": (
                "the immediately completed 24-hour BTC realized variation must rank in its causal "
                "upper 35%, while the spot-perpetual innovation magnitude must enter its own upper 30%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "an eight-hour synchronized cross-market error-correction leadership clock is absent "
                "from Gross9 primitives"
            ),
        },
        "research_basis": {
            "primary_reference": (
                "Hasbrouck (1995), One Security, Many Markets: Determining the Contributions to "
                "Price Discovery, Journal of Finance 50, 1175-1199"
            ),
            "primary_doi": "10.1111/j.1540-6261.1995.tb04054.x",
            "bitcoin_reference": (
                "Alexander and Heck (2020), Price discovery in Bitcoin: The impact of unregulated "
                "markets, Journal of Financial Stability 50, 100776"
            ),
            "bitcoin_doi": "10.1016/j.jfs.2020.100776",
            "paper_object": (
                "cointegrated prices, vector error-correction adjustment loadings, and derivatives "
                "market leadership in Bitcoin price discovery"
            ),
            "selection_boundary": (
                "the singleton synchronization, model, ranks, side, decision grid, and hold were fixed "
                "without opening candidate incidence, Gross9 rows, or post-entry outcomes"
            ),
        },
        "source_and_synchronization": {
            "decision_grid": "every exact 00:00, 08:00, and 16:00 UTC boundary D",
            "spot": "bars_binance_spot BTCUSDT interval=1m",
            "perpetual": "bars_binance BTCUSDT interval=1m",
            "columns": ["ts", "open", "high", "low", "close"],
            "model_window": "2016 exact paired minutes [D-7d,D), ending at D-1m",
            "source_validity": (
                "both sources have the identical unique exact UTC-minute grid, finite positive coherent "
                "OHLC, and no missing or imputed row"
            ),
            "prices": "natural logs of synchronized minute closes",
        },
        "error_correction_model": {
            "spread": "e_t=log(perpetual_close_t)-log(spot_close_t)",
            "rows": "t=second through final model-window minute, yielding 2014 regression rows",
            "spot_equation": (
                "delta_spot_t=a_s+alpha_s*e_(t-1)+beta_ss*delta_spot_(t-1)+"
                "beta_sp*delta_perpetual_(t-1)+residual_s_t"
            ),
            "perpetual_equation": (
                "delta_perpetual_t=a_p+alpha_p*e_(t-1)+beta_ps*delta_spot_(t-1)+"
                "beta_pp*delta_perpetual_(t-1)+residual_p_t"
            ),
            "estimation": (
                "separate numpy.linalg.lstsq(rcond=None) OLS fits with intercept; full column rank, "
                "finite coefficients, and finite residuals required"
            ),
            "stable_adjustment": "alpha_s>0 and alpha_p<0",
            "perpetual_leadership_share": "alpha_s/(alpha_s-alpha_p) under stable adjustment",
            "leadership_rule": "perpetual_leadership_share>=0.60",
            "refit": "repeat at every decision using only paired closes strictly before D",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "outcome_optimization": False,
        },
        "innovation_and_eligibility": {
            "completed_hour_return": (
                "r_spot=log(spot_close_(D-1m)/spot_open_(D-60m)); "
                "r_perpetual analogously on the same exact 60 rows"
            ),
            "lead_innovation": "u=r_perpetual-r_spot, strict nonzero",
            "direction_agreement": "sign(u)=sign(r_perpetual), with r_perpetual strict nonzero",
            "innovation_rank": (
                "strict-prior midrank of abs(u) over at most 270 earlier source-valid decisions, "
                "minimum 180, current excluded; rank>=0.70"
            ),
            "variation": (
                "sqrt(sum of squared 1439 close-to-close perpetual one-minute log returns)) over "
                "the exact complete interval [D-24h,D)"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, "
                "current excluded; rank>=0.65"
            ),
            "rule": (
                "source/model valid, stable adjustment, perpetual leadership share>=0.60, strict "
                "direction agreement, innovation rank>=0.70, and variation rank>=0.65"
            ),
            "no_imputation": True,
        },
        "causal_label_authorization": {
            "allowed": "only synchronized spot and perpetual source rows with timestamps strictly before D",
            "current_or_future_source_forbidden": True,
            "candidate_postentry_return_pnl_funding_cagr_mdd_forbidden_before_novelty": True,
            "future_stage_refit_reset_or_threshold_change": False,
        },
        "clock": {
            "decision": "exact eight-hour UTC boundary D after all source rows through D-1m are complete",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "strict sign of completed one-hour perpetual return",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "model_lookback_days": 7,
            "model_minutes": 2016,
            "ecm_lag_minutes": 1,
            "perpetual_leadership_share_min": 0.60,
            "innovation_hours": 1,
            "rank_lookback_decisions": 270,
            "rank_minimum_decisions": 180,
            "innovation_rank_min": 0.70,
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
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every "
                "held 5m favorable then adverse, global HWM, full-calendar CAGR"
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
                "no_variation_gate",
                "no_innovation_tail",
                "no_leadership_gate",
                "one_decision_stale_model",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "window": ["2023-03-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration": True,
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_spot_perpetual_family_outcomes_known": True,
            "repository_vecm_cointegration_adjustment_candidate_found": False,
            "prior_event_sets_models_predictions_or_controls_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_window_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "current_candidate_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published cointegrated multi-market price-discovery mechanism",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no "
            "source, synchronization, model, loading, leadership, rank, side, hold, clock, subset, "
            "comparator, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVSPER preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": canonical_hash(payload)}))
