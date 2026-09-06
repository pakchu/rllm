"""Outcome-sequenced preregistration for HVPRF-48."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.search_tabicl_foundation_alpha import COMPACT_FEATURES


DEFAULT_OUTPUT = Path(
    "results/high_volatility_path_utility_random_forest_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "training/search_tabicl_foundation_alpha.py": "cf2971aa55eb80326eb4e27f530c3377553a2c30d12ec95372877dc00e821964",
    "preprocessing/market_features.py": "f9091ecb080656c69a08ac3b4d07f7316cc2ddcc1fe4efacb9e10e8334d5cafa",
    "training/long_regime_interest_gate_validation.py": "cc9f4b0ea85079992ca060719ad2ab6afc2018884325a893adb459304e312075",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_path_utility_random_forest_relay_v1",
        "policy_id": "HVPRF-48",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Two nonlinear regressors fitted before 2023 can forecast side-specific 48-hour "
                "path utility, defined as terminal directional return less a fixed penalty for "
                "the side's adverse excursion. Trading only high predicted utilities during "
                "high range volatility targets directional paths with controlled intra-trade risk."
            ),
            "side": "the side whose predicted path utility has the larger frozen normalized excess",
            "why_distinct": (
                "HVPRF uses two RandomForest path-utility targets that penalize future adverse "
                "excursion. HVGBR predicted only terminal 72-hour return; HVETCR predicted 48-hour "
                "class probabilities; HVMRR used a linear six-hour return target. "
                "No terminal model output, fitted artifact, threshold, control, or OOS result is reused."
            ),
            "why_suited_to_volatile_regimes": (
                "range_vol must exceed a calibration-only 65th percentile before either side can trade"
            ),
        },
        "training_contract": {
            "feature_source_start": "2020-01-01T00:00:00Z",
            "fit_start": "2020-01-01T00:00:00Z",
            "fit_end_exclusive": "2023-01-01T00:00:00Z",
            "calibration_start": "2023-01-01T00:00:00Z",
            "calibration_end_exclusive": "2023-07-01T00:00:00Z",
            "oos_start": "2023-07-01T00:00:00Z",
            "anchors": "every 72 completed 5m bars from fixed position 143",
            "labels": {
                "terminal_return": "log(open at anchor+1+576/open at anchor+1)",
                "long_adverse": "max(0,-min log(low/entry_open)) over the held 48h path",
                "short_adverse": "max(0,max log(high/entry_open)) over the held 48h path",
                "long_utility": "terminal_return-0.75*long_adverse",
                "short_utility": "-terminal_return-0.75*short_adverse",
            },
            "fit_rows": "all anchors with complete label and finite entry/exit opens in fit window",
            "feature_columns": list(COMPACT_FEATURES),
            "imputer": "sklearn SimpleImputer(strategy='median') fit only on fit rows",
            "estimator": {
                "class": "sklearn.ensemble.RandomForestRegressor",
                "models": 2,
                "n_estimators": 600,
                "max_depth": 10,
                "min_samples_leaf": 20,
                "max_features": 0.7,
                "bootstrap": True,
                "n_jobs": -1,
                "random_states": [809, 810],
            },
            "prediction_thresholds": (
                "separate strict empirical 75th percentiles of predicted long and short utility; "
                "separate calibration IQR scales, all on calibration anchors only"
            ),
            "volatility_threshold": "65th percentile of range_vol on calibration anchors only",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "refit_after_2022_12_31": False,
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "oos_clock": {
            "decisions": "same fixed six-hour anchor schedule from 2023-07-01 onward",
            "eligibility": (
                "all ordered features scoreable after fitted median imputation; range_vol "
                ">= frozen q65; at least one side-specific predicted utility >= its frozen q75"
            ),
            "entry": "anchor completed bar plus one 5m open",
            "side": (
                "if one side is eligible choose it; if both are eligible choose larger "
                "(prediction-q75)/max(IQR,1e-12), ties skip"
            ),
            "hold": "48 elapsed hours",
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per "
                "notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": ["no_volatility_gate", "no_prediction_tail_gate", "one_anchor_stale_features", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01 plus bound funding/premium auxiliaries",
            "live_extension": "read-only Postgres completed bars/features through 2026-08-01",
            "oos_execution_prices": "sealed until source support and novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_random_forest_path_families_exist_and_outcomes_are_known": True,
            "prior_hvgbr_hvetcr_and_hvmrr_terminal_outcomes_known": True,
            "exact_regressor_and_bidirectional_high_volatility_rule_outcomes_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze preregistration, fit/calibrate only before 2023-07, freeze model, then "
            "open OOS incidence, novelty, and sequential economics; terminal first failure"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVPRF preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVPRF source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
