"""Outcome-sequenced preregistration for HVPAR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_path_analog_relay_preregistration_2026-08-10.json")
PATH_FEATURES = tuple(f"normalized_path_return_{index:02d}" for index in range(16))
VARIATION_FEATURES = tuple(f"variation_share_{index:02d}" for index in range(16))
FEATURES = (*PATH_FEATURES, *VARIATION_FEATURES, "variation_rank")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_path_analog_relay_v1",
        "policy_id": "HVPAR-8",
        "as_of_date": "2026-08-10",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "The next eight-hour direction after a volatile BTC block is conditional on analog recurrence "
                "of the completed block's half-hour directional path and within-block variation allocation. "
                "A fixed nearest-analog estimator can represent nonlinear path states without OOS adaptation."
            ),
            "side": "strict sign of the frozen nearest-analog prediction",
            "why_distinct": (
                "HVPAR is a nonparametric path-recurrence model over 32 ordered half-hour coordinates and one "
                "causal volatility rank. HVITR was one linear ridge over aggregate topology statistics; prior "
                "handcrafted candidates used individual topology gates. HVPAR reuses no candidate event set, "
                "prediction, fitted threshold, diagnostic result, or OOS outcome. It intentionally retains the "
                "project-wide comparison clock, volatility gate, controls and economic gates, and uses no flow, "
                "funding, OI, basis, calendar, or cross-asset input."
            ),
            "why_suited_to_volatile_regimes": (
                "both reference analogs and OOS queries require causal eight-hour variation rank at least 0.65"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "fixed-boundary high-dimensional price-path recurrence is absent from Gross9"
            ),
        },
        "training_contract": {
            "source_window": ["2020-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "label_window": ["2021-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "decisions": "exact 00:00/08:00/16:00 UTC completed boundaries",
            "label": "log(exit_open/entry_open), entry=D+5m, exit=entry+8 elapsed hours",
            "training_rows": "all source-valid decisions with finite ordered features and complete label",
            "standardization": (
                "one population mean/std fit once on all training rows; the same full-training transform is used "
                "for purged leave-one-out calibration and frozen OOS inference; reject nonfinite or zero scale"
            ),
            "reference_population": "training rows with variation_rank>=0.65 only",
            "estimator": {
                "distance": "Euclidean distance over all 33 standardized ordered features",
                "neighbors": 64,
                "tie_break": "ascending distance then ascending decision_time",
                "weight": "1/max(distance,1e-12), normalized to sum one",
                "prediction": "weighted arithmetic mean of neighbor labels",
                "training_prediction": (
                    "purged leave-one-out: for query decision t, exclude the current reference and every reference "
                    "whose [decision-8h,decision) feature window intersects query label interval [t+5m,t+8h+5m]; "
                    "on the fixed grid this excludes t and any available t+8h and t+16h references"
                ),
                "minimum_reference_rows": 67,
            },
            "prediction_strength_threshold": (
                "numpy linear-method empirical 0.75 quantile of absolute purged leave-one-out predictions over "
                "every reference row; OOS eligibility uses absolute prediction greater than or equal to this value"
            ),
            "label_endpoint": "require entry and exit opens finite/positive and exit strictly before 2023-01-01T00:00:00Z",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "refit_after_2022_12_31": False,
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "feature_contract": {
            "ordered_features": list(FEATURES),
            "block": "480 exact coherent BTCUSDT bars_binance interval=1m rows [D-8h,D)",
            "path_partition": "sixteen consecutive 30-minute segments indexed 00..15",
            "path_levels": (
                "17 levels: first bar open followed by closes at minute offsets 29,59,...,479; "
                "the 16 log differences are divided by sqrt(full close-to-close squared-return sum)"
            ),
            "variation_shares": (
                "each close-to-close squared return is assigned to the segment containing its later close; "
                "the first segment therefore has 29 returns and each later segment has 30; each segment sum "
                "is divided by the full 479-return squared sum. The first normalized path return intentionally "
                "starts at first-bar open while variation uses only close-to-close returns."
            ),
            "variation_rank": "strict-prior midrank over at most 270 valid blocks, minimum 180, current excluded",
            "source_validity": (
                "exactly 480 unique expected timestamps, finite positive OHLC, high>=max(open,close), "
                "low<=min(open,close), high>=low, positive full variation, no imputation"
            ),
            "no_imputation": True,
        },
        "oos_clock": {
            "start": "2023-07-01T00:00:00Z",
            "eligibility": (
                "source valid, variation_rank>=0.65, frozen prediction finite/nonzero, and absolute prediction "
                ">= frozen leave-one-out training q75"
            ),
            "entry": "D+5m BTCUSDT open",
            "side": "sign of frozen prediction",
            "hold": "8 elapsed hours",
            "reservation": "fixed boundaries are naturally half-open; exit first on equal open",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "variation_rank_min": 0.65,
            "neighbors": 64,
            "prediction_strength_quantile": 0.75,
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
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "no_prediction_strength_gate",
                "one_boundary_stale_features",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "table": "bars_binance",
            "symbol": "BTCUSDT",
            "interval": "1m",
            "columns": ["ts", "open", "high", "low", "close"],
            "oos_execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_rule_and_model_family_outcomes_known": True,
            "prior_event_sets_or_controls_promoted": False,
            "oos_outcomes_used_to_fit_select_or_threshold_hvpar": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "shared_high_volatility_source_backbone_known_from_prior_candidates": True,
            "candidate_specific_analog_prediction_and_incidence_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "single nonlinear path-recurrence mechanism; proposed after terminating HVITR unchanged and "
                "therefore prospectively confirmatory only for its still-sealed candidate-specific analog incidence/outcomes"
            ),
        },
        "stopping_rule": (
            "Freeze preregistration, analog model, OOS source support, Gross9 novelty, then strict economics; "
            "terminal first failure with no feature, neighbor count, distance, weight, threshold, side, hold, "
            "clock, subset, or control repair."
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
