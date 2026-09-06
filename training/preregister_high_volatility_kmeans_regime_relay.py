"""Outcome-sequenced preregistration for HVKMR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_kmeans_regime_relay_preregistration_2026-08-09.json"
)
FEATURES = (
    "range_vol",
    "trend_12",
    "trend_24",
    "trend_96",
    "rsi_norm",
    "bb_z",
    "range_pos",
    "body_to_range",
    "shadow_imbalance",
    "volume_zscore",
    "window_drawdown",
    "taker_imbalance",
    "funding_zscore",
    "premium_index_zscore",
    "premium_index_change",
    "oi_change",
    "oi_zscore",
    "htf_4h_return_1",
    "htf_1d_return_1",
    "dollar_flow_rel_4h_30d",
)
SOURCE_BINDINGS = {
    "preprocessing/market_features.py": "f9091ecb080656c69a08ac3b4d07f7316cc2ddcc1fe4efacb9e10e8334d5cafa",
    "training/long_regime_interest_gate_validation.py": "cc9f4b0ea85079992ca060719ad2ab6afc2018884325a893adb459304e312075",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz": "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7",
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz": "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7",
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_kmeans_regime_relay_v1",
        "policy_id": "HVKMR-24",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "pretraining_outcomes_authorized_after_preregistration": True,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A source-standardized, unsupervised market-state partition learned before "
                "2023 can isolate recurring high-volatility states whose pre-2023 24-hour "
                "conditional return signs persist out of sample."
            ),
            "side": "long in the two strongest positive states; short in the two strongest negative states",
            "why_distinct": (
                "HVKMR uses unsupervised Euclidean state partitioning followed by a frozen "
                "pre-2023 state-to-side map. It does not reuse any terminal candidate rule, "
                "model output, threshold, control, or outcome."
            ),
            "why_suited_to_volatile_regimes": (
                "range_vol must exceed a source-only 60th percentile before a mapped state can trade"
            ),
        },
        "training_contract": {
            "fit_start": "2020-01-01T00:00:00Z",
            "fit_end_exclusive": "2023-01-01T00:00:00Z",
            "calibration_start": "2023-01-01T00:00:00Z",
            "calibration_end_exclusive": "2023-07-01T00:00:00Z",
            "oos_start": "2023-07-01T00:00:00Z",
            "anchors": "every 36 completed 5m bars from fixed position 143",
            "label_for_state_mapping": "log(open at anchor+1+288/open at anchor+1)",
            "feature_columns": list(FEATURES),
            "pipeline": {
                "imputer": "sklearn SimpleImputer(strategy='median') fit on pre-2023 rows only",
                "scaler": "sklearn StandardScaler fit on imputed pre-2023 rows only",
                "clusterer": "sklearn KMeans(n_clusters=12,n_init=30,max_iter=500,random_state=809)",
            },
            "state_side_map": (
                "among clusters with at least 100 complete pre-2023 labels, freeze the two "
                "largest positive mean labels as long and two smallest negative mean labels "
                "as short; fail model freeze if either side has fewer than two states"
            ),
            "volatility_threshold": "60th percentile of range_vol on 2023H1 calibration anchors only",
            "hyperparameter_grid": False,
            "feature_selection": False,
            "refit_after_2022_12_31": False,
            "model_artifact_must_be_frozen_before_oos_incidence": True,
        },
        "oos_clock": {
            "decisions": "same fixed three-hour anchor schedule from 2023-07-01 onward",
            "eligibility": "mapped state and range_vol at or above frozen q60",
            "entry": "anchor completed bar plus one 5m open",
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
            "names": ["no_volatility_gate", "one_anchor_stale_features", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01 plus bound funding/premium auxiliaries",
            "live_extension": "read-only Postgres completed bars/features through 2026-08-01",
            "oos_execution_prices": "sealed until source support and novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "unsupervised_kmeans_state_family_outcomes_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze preregistration, fit/map states only before 2023 and calibrate source-only "
            "in 2023H1, freeze model, then open OOS incidence, novelty, and sequential economics"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVKMR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVKMR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
