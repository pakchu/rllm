"""Annual causal retraining for the frozen TabICLv2 Top-10 algorithm family."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim
from training.search_tabicl_foundation_alpha import (
    ANCHOR_STRIDE,
    HOLD_BARS,
    _fit_predict,
    anchor_dataset,
    feature_groups,
    policy_masks,
    split_mask_for_anchors,
    top10_promotions,
)


FOLDS = {
    "test2024": {
        "fit": ("2020-01-01", "2023-01-01"),
        "calibration": ("2023-01-01", "2024-01-01"),
        "evaluation": ("2024-01-01", "2025-01-01"),
    },
    "eval2025": {
        "fit": ("2020-01-01", "2024-01-01"),
        "calibration": ("2024-01-01", "2025-01-01"),
        "evaluation": ("2025-01-01", "2026-01-01"),
    },
    "ytd2026": {
        "fit": ("2020-01-01", "2025-01-01"),
        "calibration": ("2025-01-01", "2026-01-01"),
        "evaluation": ("2026-01-01", "2026-06-02"),
    },
}


def validate_fold_chronology(folds: dict = FOLDS) -> None:
    for name, fold in folds.items():
        fit_start, fit_end = map(pd.Timestamp, fold["fit"])
        cal_start, cal_end = map(pd.Timestamp, fold["calibration"])
        eval_start, eval_end = map(pd.Timestamp, fold["evaluation"])
        if not (fit_start < fit_end <= cal_start < cal_end <= eval_start < eval_end):
            raise ValueError(f"invalid causal fold chronology: {name}")


def _frozen_candidates(manifest: dict) -> list[dict]:
    return [dict(row) for row in manifest["top10"]]


def run(args: argparse.Namespace) -> dict:
    validate_fold_chronology()
    frozen_manifest = json.loads(Path(args.frozen_manifest).read_text())
    candidates = _frozen_candidates(frozen_manifest)
    if len(candidates) != 10 or frozen_manifest.get("later_metrics_included") is not False:
        raise ValueError("expected an untouched frozen Top-10 manifest")

    adaptation_manifest_path = Path(args.adaptation_manifest_output)
    if adaptation_manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen adaptation manifest: {adaptation_manifest_path}"
        )
    adaptation_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_frozen_manifest": args.frozen_manifest,
        "candidate_signal_hashes": [row["signal_hash"] for row in candidates],
        "folds": FOLDS,
        "algorithm": (
            "for each evaluation fold, fit the frozen model/feature specification before the "
            "calibration year; derive the same frozen score quantile on calibration predictions; "
            "evaluate the following period without updating"
        ),
        "later_metrics_included": False,
    }
    adaptation_manifest_path.write_text(
        json.dumps(adaptation_manifest, indent=2, ensure_ascii=False)
    )

    cfg = Config(
        input_csv=args.input_csv,
        output=args.output,
        funding_csv=args.funding_csv,
        premium_csv=args.premium_csv,
        exclude_from=args.exclude_from,
    )
    for name, fold in FOLDS.items():
        state_sim.W[name] = fold["evaluation"]
    state_sim.W["rolling_oos"] = ("2024-01-01", "2026-06-02")

    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    groups = feature_groups(features)
    positions, target, _ = anchor_dataset(market, features)
    matrices = {
        group: features[columns].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        for group, columns in groups.items()
    }
    candidate_by_model: dict[str, list[dict]] = {}
    for rank, candidate in enumerate(candidates, start=1):
        candidate["pre_evaluation_rank"] = rank
        candidate["rolling_metrics"] = {}
        candidate["_long"] = np.zeros(len(market), dtype=bool)
        candidate["_short"] = np.zeros(len(market), dtype=bool)
        candidate_by_model.setdefault(candidate["model_id"], []).append(candidate)

    fold_model_diagnostics: dict[str, dict] = {}
    for fold_name, fold in FOLDS.items():
        fit_mask = split_mask_for_anchors(dates, positions, *fold["fit"])
        calibration_mask = split_mask_for_anchors(
            dates, positions, *fold["calibration"]
        )
        evaluation_mask = split_mask_for_anchors(dates, positions, *fold["evaluation"])
        fit_target = target[fit_mask]
        low, high = np.quantile(fit_target, (0.01, 0.99))
        fit_target = np.clip(fit_target, low, high)
        fold_model_diagnostics[fold_name] = {}

        for model_id, model_candidates in candidate_by_model.items():
            reference = model_candidates[0]
            model_name = reference["model_name"]
            group_name = reference["feature_group"]
            matrix = matrices[group_name]
            calibration_positions = positions[calibration_mask]
            evaluation_positions = positions[evaluation_mask]
            prediction_positions = np.r_[calibration_positions, evaluation_positions]
            predictions = _fit_predict(
                model_name,
                matrix[positions[fit_mask]],
                fit_target,
                matrix[prediction_positions],
            )
            calibration_scores = predictions[: len(calibration_positions)]
            evaluation_scores = predictions[len(calibration_positions) :]
            from scipy.stats import spearmanr

            calibration_target = target[calibration_mask]
            evaluation_target = target[evaluation_mask]
            fold_model_diagnostics[fold_name][model_id] = {
                "fit_samples": int(fit_mask.sum()),
                "calibration_samples": int(calibration_mask.sum()),
                "evaluation_samples": int(evaluation_mask.sum()),
                "target_clip_values": [float(low), float(high)],
                "calibration_spearman": float(
                    spearmanr(calibration_scores, calibration_target).statistic
                ),
                "evaluation_spearman": float(
                    spearmanr(evaluation_scores, evaluation_target).statistic
                ),
            }

            for candidate in model_candidates:
                low_quantile = candidate["low_score_quantile"]
                high_quantile = candidate["high_score_quantile"]
                low_threshold = (
                    float(np.quantile(calibration_scores, low_quantile))
                    if low_quantile is not None
                    else None
                )
                high_threshold = (
                    float(np.quantile(calibration_scores, high_quantile))
                    if high_quantile is not None
                    else None
                )
                long_active, short_active = policy_masks(
                    evaluation_scores,
                    evaluation_positions,
                    len(market),
                    side_policy=candidate["side_policy"],
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                )
                candidate["_long"] |= long_active
                candidate["_short"] |= short_active
                candidate["rolling_metrics"][fold_name] = {
                    "calibrated_low_threshold": low_threshold,
                    "calibrated_high_threshold": high_threshold,
                    "stats": sim(
                        market,
                        dates,
                        long_active,
                        short_active,
                        cfg,
                        HOLD_BARS,
                        ANCHOR_STRIDE,
                        10.0,
                        10.0,
                        fold_name,
                    ),
                }

    stress: dict[str, dict] = {}
    for candidate in candidates:
        candidate["rolling_oos"] = sim(
            market,
            dates,
            candidate["_long"],
            candidate["_short"],
            cfg,
            HOLD_BARS,
            ANCHOR_STRIDE,
            10.0,
            10.0,
            "rolling_oos",
        )
        test = candidate["rolling_metrics"]["test2024"]["stats"]
        evaluation = candidate["rolling_metrics"]["eval2025"]["stats"]
        recent = candidate["rolling_metrics"]["ytd2026"]["stats"]
        candidate["passes_alpha_pool"] = bool(
            test["ratio"] >= 3.0
            and evaluation["ratio"] >= 3.0
            and test["trades"] >= 8
            and evaluation["trades"] >= 8
            and test["return_pct"] > 0.0
            and evaluation["return_pct"] > 0.0
        )
        candidate["passes_live_grade"] = bool(
            candidate["passes_alpha_pool"]
            and recent["ratio"] >= 5.0
            and recent["trades"] >= 6
            and recent["return_pct"] > 0.0
        )
        if candidate["passes_live_grade"]:
            stress[candidate["signal_hash"]] = {}
            for bps in (6, 8, 10, 15):
                stressed_cfg = replace(
                    cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate)
                )
                stress[candidate["signal_hash"]][str(bps)] = {
                    fold_name: sim(
                        market,
                        dates,
                        candidate["_long"],
                        candidate["_short"],
                        stressed_cfg,
                        HOLD_BARS,
                        ANCHOR_STRIDE,
                        10.0,
                        10.0,
                        fold_name,
                    )
                    for fold_name in (*FOLDS, "rolling_oos")
                }
        candidate.pop("_long")
        candidate.pop("_short")

    alpha_pool, live_grade = top10_promotions(candidates)
    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "source_frozen_manifest": args.frozen_manifest,
        "adaptation_manifest": str(adaptation_manifest_path),
        "protocol": (
            "frozen Top-10 algorithms; annual expanding fit ending before calibration year; "
            "same frozen score quantile calibrated on immediately preceding year; following "
            "period untouched; next-bar entry; hold576; 6bp/side; strict intratrade MDD"
        ),
        "folds": FOLDS,
        "model_diagnostics": fold_model_diagnostics,
        "selected": candidates,
        "alpha_pool_qualifiers": alpha_pool,
        "live_grade": live_grade,
        "cost_stress_bps_per_side": stress,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--adaptation-manifest-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "selected": len(result["selected"]),
                "alpha_pool": len(result["alpha_pool_qualifiers"]),
                "live_grade": len(result["live_grade"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
