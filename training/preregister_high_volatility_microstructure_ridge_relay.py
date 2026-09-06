"""Outcome-sequenced preregistration for HVMRR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("results/high_volatility_microstructure_ridge_relay_preregistration_2026-08-09.json")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    features = [
        "normalized_full_return",
        "normalized_late_return",
        "variation_rank",
        "path_efficiency",
        "full_taker_imbalance",
        "late_taker_imbalance",
        "late_quote_volume_share",
        "normalized_cash_basis_change",
        "decision_hour_sin",
        "decision_hour_cos",
    ]
    core = {
        "protocol_version": "high_volatility_microstructure_ridge_relay_v1",
        "policy_id": "HVMRR-6",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "singleton": True,
        "mechanism": {
            "claim": "In high-realized-volatility BTC blocks, the continuation-versus-reversal balance depends jointly on normalized price path, path efficiency, aggressive-flow sponsorship, late participation, and cash-versus-perpetual basis repricing. A single ridge trained only before 2023H2 can estimate the six-hour direction without selecting an OOS rule family.",
            "side": "strict sign of the frozen ridge prediction",
            "why_distinct": "HVMRR combines ten pre-entry price, flow, basis, and clock features through one preregistered linear shrinkage model. Prior high-volatility singletons froze one hand-written gate and direction; causal expert work selected among discrete expert policies. HVMRR has one model, no OOS adaptation, and promotes no prior control.",
            "why_suited_to_volatile_regimes": "only blocks whose causal realized-variation rank is at least 0.65 can trade",
            "why_low_gross9_overlap_is_plausible": "a sparse fixed-eight-hour multivariate microstructure score is absent from Gross9",
        },
        "training_contract": {
            "feature_source_start": "2020-10-01T00:00:00Z",
            "label_start": "2021-01-01T00:00:00Z",
            "label_end_exclusive": "2023-07-01T00:00:00Z",
            "label": "BTCUSDT log(exit_open/entry_open), entry=decision+5m and exit=entry+6h",
            "training_rows": "all source-valid fixed 00:00/08:00/16:00 UTC decisions with finite features and complete pre-2023H2 label; no support or outcome filtering",
            "standardization": "population mean and population standard deviation computed on training rows only; reject any zero or nonfinite scale",
            "estimator": "sklearn.linear_model.Ridge(alpha=10.0, fit_intercept=True, solver='svd')",
            "sample_weight": "none",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "refit_after_2023_06_30": False,
            "prediction_strength_threshold": "strict empirical 0.80 quantile of absolute fitted predictions among training rows with variation_rank>=0.65",
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "feature_contract": {
            "ordered_features": features,
            "block": "480 exact aligned BTCUSDT perpetual and spot minute rows in [decision-8h,decision); finite positive coherent OHLC; perpetual quote_asset_volume and taker_buy_quote finite, nonnegative, taker<=quote, positive total and late quote volume; no imputation",
            "full_return": "log(perpetual close at decision-1m / open at decision-8h)",
            "late_return": "log(perpetual close at decision-1m / open at decision-2h)",
            "realized_variation": "sum squared perpetual 1m close-to-close log returns inside the block",
            "variation_rank": "strict-prior midrank among at most 270 source-valid fixed blocks, minimum 180, current excluded",
            "normalized_returns": "return/sqrt(realized_variation)",
            "path_efficiency": "abs(full_return)/sum(abs(perpetual 1m close-to-close log returns)); zero denominator invalid",
            "taker_imbalances": "(2*sum(taker_buy_quote)-sum(quote_asset_volume))/sum(quote_asset_volume), separately full block and final two hours",
            "late_quote_volume_share": "final-two-hour quote volume / full-block quote volume",
            "cash_basis_change": "log(perpetual/spot) at final close minus log(perpetual/spot) at final-two-hour opening boundary, divided by sqrt(realized_variation)",
            "clock_encoding": "sin and cos of 2*pi*decision_UTC_hour/24",
            "no_imputation": True,
        },
        "oos_clock": {
            "decisions": "exact 00:00,08:00,16:00 UTC from 2023-07-01 onward",
            "eligibility": "source valid, variation_rank>=0.65, finite frozen-model prediction, abs(prediction)>=frozen training q80, prediction!=0",
            "entry": "exact decision+5m BTCUSDT open",
            "side": "sign of frozen prediction",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        "policy": {
            "history_observations": 270,
            "minimum_history_observations": 180,
            "variation_rank_min": 0.65,
            "ridge_alpha": 10.0,
            "prediction_strength_quantile": 0.80,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
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
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {
            "perpetual": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"]},
            "spot": {"table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"]},
            "window": ["2020-10-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "oos_execution_price": "sealed until source-support and Gross9 novelty pass; pretraining labels explicitly authorized only before 2023-07-01",
        },
        "diagnostic_controls": {"names": ["no_volatility_gate", "no_prediction_strength_gate", "one_boundary_stale_features", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {
            "prior_rule_and_model_family_outcomes_known": True,
            "prior_event_sets_or_controls_promoted": False,
            "oos_outcomes_used_to_fit_select_or_threshold_hvmrr": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "single pre-2023H2 multivariate microstructure ridge specified before its pretraining labels are opened",
        },
        "stopping_rule": "Freeze preregistration, then model, then OOS source support, Gross9 novelty, and strict sequential economics; terminal first failure with no feature, model, alpha, threshold, side, hold, clock, or subset repair.",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2) + "\n")
    print(args.output)
