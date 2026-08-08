"""Outcome-sequenced preregistration for HVDLDA-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_extra_trees_classification_relay as base


COMPACT_FEATURES = base.COMPACT_FEATURES
SOURCE_BINDINGS = base.SOURCE_BINDINGS
DEFAULT_OUTPUT = Path(
    "results/high_volatility_discriminant_state_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    return base.canonical_hash(payload)


def build() -> dict[str, Any]:
    payload = base.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    core.update(
        {
            "protocol_version": "high_volatility_discriminant_state_relay_v1",
            "policy_id": "HVDLDA-24",
            "mechanism": {
                "claim": (
                    "A shrinkage linear-discriminant classifier fitted only before 2023 can identify "
                    "broad causal market states whose next 24-hour BTC direction is persistent. "
                    "Probability tails are traded only during high range-volatility."
                ),
                "side": "long in frozen upper probability tail; short in frozen lower tail",
                "why_distinct": (
                    "HVDLDA uses a covariance-shrunk generative state classifier, a 24-hour label, "
                    "and three-hour anchors. HVETCR used nonlinear ExtraTrees and 48-hour labels; "
                    "HVMRR used six-hour ridge regression on a raw microstructure block. No fitted "
                    "artifact, threshold, control, or OOS outcome from either is reused."
                ),
                "why_suited_to_volatile_regimes": (
                    "range_vol must exceed the calibration-only 60th percentile"
                ),
            },
            "training_contract": {
                "feature_source_start": "2020-01-01T00:00:00Z",
                "fit_start": "2020-01-01T00:00:00Z",
                "fit_end_exclusive": "2023-01-01T00:00:00Z",
                "calibration_start": "2023-01-01T00:00:00Z",
                "calibration_end_exclusive": "2023-07-01T00:00:00Z",
                "oos_start": "2023-07-01T00:00:00Z",
                "anchors": "every 36 completed 5m bars from fixed position 143",
                "label": "1 iff log(open at anchor+1+288/open at anchor+1)>0, else 0",
                "fit_rows": "complete-label finite-open anchors wholly inside fit window",
                "feature_columns": list(COMPACT_FEATURES),
                "pipeline": [
                    "SimpleImputer(strategy='median') fit only on fit rows",
                    "StandardScaler fit only on fit rows",
                    "LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')",
                ],
                "probability_thresholds": (
                    "strict empirical 15th and 85th percentiles of positive-class probability on "
                    "calibration anchors only"
                ),
                "volatility_threshold": "60th percentile of range_vol on calibration anchors only",
                "hyperparameter_grid": False,
                "feature_selection": False,
                "refit_after_2022_12_31": False,
                "model_artifact_must_be_frozen_before_oos_incidence": True,
            },
            "oos_clock": {
                "decisions": "same fixed three-hour anchor schedule from 2023-07-01 onward",
                "eligibility": (
                    "all ordered features scoreable after fitted median imputation; range_vol >= "
                    "frozen q60; probability <= frozen q15 or >= frozen q85"
                ),
                "entry": "anchor completed bar plus one 5m open",
                "side": "probability>=q85 long; probability<=q15 short",
                "hold": "24 elapsed hours",
                "reservation": "global half-open; exit first on equal open",
                "split_crossing_action": "skip",
                "gross_exposure": 0.5,
            },
            "diagnostic_controls": {
                "names": [
                    "no_volatility_gate",
                    "no_probability_tail_gate",
                    "one_anchor_stale_features",
                    "direction_flip",
                ],
                "diagnostic_controls_cannot_be_promoted": True,
            },
            "research_boundary": {
                "prior_classifier_and_linear_model_outcomes_known": True,
                "exact_shrinkage_discriminant_24h_rule_outcomes_known": False,
                "oos_candidate_incidence_opened": False,
                "oos_post_entry_return_or_pnl_opened": False,
                "gross9_rows_opened": False,
                "candidate_count": 1,
                "grid": False,
                "repair_of_prior_candidate": False,
                "promoted_prior_control": False,
            },
            "stopping_rule": (
                "freeze preregistration, fit/calibrate only before 2023-07, freeze model, then open "
                "OOS incidence, novelty, and sequential economics; terminal first failure"
            ),
        }
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDLDA preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVDLDA source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
