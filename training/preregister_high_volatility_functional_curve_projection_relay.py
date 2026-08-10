"""Outcome-blind preregistration for HVFCPR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVFCPR-24"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_functional_curve_projection_relay_"
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
        "protocol_version": "high_volatility_functional_curve_projection_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "The shape of a complete Bitcoin cumulative intraday return curve contains a "
                "low-dimensional functional state. A rolling functional principal-component model "
                "fitted only to 180 already-completed daily curves can forecast the next day's "
                "terminal cumulative return. Trade that forecast only when both the forecast magnitude "
                "and the immediately completed day's realized variation are in causal upper regimes."
            ),
            "side": "strict sign of the rolling FPCA reconstruction's forecast terminal return",
            "why_distinct": (
                "HVFCPR forecasts a full 288-coordinate cumulative intraday return function through "
                "rolling SVD and componentwise score dynamics. HVPAR uses frozen nearest-neighbor "
                "distances on eight-hour half-hour segments; HVPRF predicts side-specific path utility "
                "with pre-2023 random forests; HVDOER selects among four momentum/reversal formulas; "
                "HVOCPR uses ordinal-pattern counts. No prior event set, model output, threshold, "
                "diagnostic control, funding, OI, volume, or cross-asset input is reused."
            ),
            "why_suited_to_volatile_regimes": (
                "the prior completed UTC day's realized variation must rank in its causal upper 35%, "
                "and only large functional forecasts are eligible"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "one sparse daily functional-forecast clock is absent from Gross9 primitives"
            ),
        },
        "research_basis": {
            "primary_reference": (
                "Bouri, Lau, Saeed, Wang, and Zhao (2021), On the intraday return curves of "
                "Bitcoin: Predictability and trading opportunities, International Review of "
                "Financial Analysis 76, 101784"
            ),
            "doi": "10.1016/j.irfa.2021.101784",
            "paper_object": (
                "cumulative intraday return curves, functional principal-component projection scores, "
                "and rolling one-day-ahead functional forecasts"
            ),
            "selection_boundary": (
                "the singleton rolling window, variance target, score law, ranks, side, clock, and hold "
                "were fixed without opening candidate incidence, Gross9 rows, or current-event outcomes"
            ),
        },
        "daily_curve": {
            "decision_grid": "every exact 00:00 UTC boundary D",
            "source_day": "1440 exact coherent BTCUSDT bars_binance interval=1m rows [D-24h,D)",
            "five_minute_aggregation": (
                "288 nonoverlapping UTC-aligned groups of five exact rows; group open is first open, "
                "high is maximum high, low is minimum low, close is last close"
            ),
            "curve_coordinates": (
                "CIDR_j=log(close_j/first five-minute open), j=0..287; all coordinates finite"
            ),
            "realized_variation": (
                "sqrt(sum of squared 1439 close-to-close one-minute log returns), strict positive"
            ),
            "source_validity": (
                "exact unique minute grid, finite positive coherent OHLC, exactly five rows per "
                "five-minute group, no imputation"
            ),
        },
        "rolling_functional_forecast": {
            "lookback_curves": 180,
            "minimum_curves": 180,
            "matrix": "180 prior complete CIDR curves ordered oldest to newest, shape 180x288",
            "centering": "subtract the coordinatewise arithmetic mean of the 180 curves",
            "decomposition": "numpy.linalg.svd(centered_matrix, full_matrices=False)",
            "component_count": (
                "smallest positive K whose cumulative squared singular values explain at least 90% "
                "of total squared singular values; zero total variation invalid"
            ),
            "scores": "U[:,0:K]*singular_values[0:K]",
            "score_forecast": (
                "for each component independently fit OLS score_t=a+b*score_(t-1) with intercept on "
                "the 179 consecutive prior score pairs using numpy.linalg.lstsq(rcond=None), then "
                "forecast from the newest score"
            ),
            "curve_forecast": "rolling mean curve plus forecast scores times Vt[0:K,:]",
            "terminal_forecast": "coordinate 287 of the reconstructed next-day curve, strict nonzero",
            "refit": "repeat causally at every decision using only curves ending at or before D",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "outcome_optimization": False,
        },
        "eligibility": {
            "forecast_strength_rank": (
                "strict-prior midrank of abs(terminal forecast) over at most 252 earlier valid "
                "forecasts, minimum 126, current excluded; rank>=0.75"
            ),
            "variation_rank": (
                "strict-prior midrank of the immediately completed source-day realized variation "
                "over at most 252 earlier valid source days, minimum 126, current excluded; rank>=0.65"
            ),
            "rule": (
                "source and forecast valid, terminal forecast strict nonzero, forecast-strength "
                "rank>=0.75, variation rank>=0.65"
            ),
            "no_imputation": True,
        },
        "causal_label_authorization": {
            "allowed": (
                "only complete historical CIDR curves whose final source minute is strictly before D; "
                "the model is an unsupervised functional decomposition plus score autoregression"
            ),
            "current_or_future_curve_forbidden": True,
            "candidate_postentry_return_pnl_funding_cagr_mdd_forbidden_before_novelty": True,
            "future_stage_refit_reset_or_threshold_change": False,
        },
        "clock": {
            "decision": "exact daily 00:00 UTC after the prior source day is complete",
            "entry": "exact BTCUSDT D+5m open",
            "side": "strict terminal-forecast sign",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "curve_coordinates": 288,
            "fpca_lookback_days": 180,
            "explained_variance_min": 0.90,
            "rank_lookback_days": 252,
            "rank_minimum_days": 126,
            "forecast_strength_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
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
                "no_forecast_strength_gate",
                "one_day_stale_forecast",
                "rolling_mean_terminal_only",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2022-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_price_path_model_family_outcomes_known": True,
            "repository_functional_cidr_fpca_candidate_found": False,
            "prior_event_sets_models_predictions_or_controls_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_window_ranks_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "current_candidate_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published cumulative-intraday-return functional forecasting",
        },
        "stopping_rule": (
            "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no "
            "window, decomposition, component, rank, side, hold, clock, subset, comparator, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVFCPR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
