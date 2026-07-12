"""Uncertainty-aware invariant ensemble alpha validation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim
from training.search_invariant_groupdro_alpha import (
    MODEL_SPECS,
    _score_quality,
    feature_set_definitions,
    half_year_environments,
    stable_feature_ranking,
    tail_labels,
    train_tail_classifier,
)
from training.search_river_contextual_utility_alpha import (
    effective_selection_signal_hash,
)
from training.search_river_online_alpha import (
    causal_rolling_thresholds,
    validate_output_paths,
)
from training.search_tabicl_foundation_alpha import (
    ANCHOR_STRIDE,
    HOLD_BARS,
    WINDOWS,
    _file_sha256,
    _git_head,
    anchor_dataset,
    feature_groups,
    split_mask_for_anchors,
    top10_promotions,
)


ROLLING_WINDOWS = (720, 1460)
SCORE_QUANTILES = (0.70, 0.80, 0.90, 0.95)
INVARIANT_MODEL_SPECS = tuple(
    spec for spec in MODEL_SPECS if spec[2] in {"vrex", "groupdro"}
)


def uncertainty_score_streams(
    member_scores: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build fixed confidence transforms from invariant ensemble members."""
    if len(member_scores) < 2:
        raise ValueError("uncertainty ensemble requires at least two members")
    member_ids = sorted(member_scores)
    stack = np.stack([np.asarray(member_scores[member_id], dtype=float) for member_id in member_ids])
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    positive_fraction = np.mean(stack > 0.0, axis=0)
    negative_fraction = np.mean(stack < 0.0, axis=0)
    agreement = np.maximum(positive_fraction, negative_fraction)
    sign = np.sign(mean)
    streams = {
        "mean": mean,
        "shrink_k0.5": sign * np.maximum(np.abs(mean) - 0.5 * std, 0.0),
        "shrink_k1.0": sign * np.maximum(np.abs(mean) - std, 0.0),
        "snr_floor0.05": mean / (std + 0.05),
        "agree_4of6": np.where(agreement >= 4.0 / 6.0, mean, 0.0),
        "agree_5of6": np.where(agreement >= 5.0 / 6.0, mean, 0.0),
        "agree_6of6": np.where(agreement >= 1.0, mean, 0.0),
    }
    diagnostics = {
        "member_ids": member_ids,
        "member_count": len(member_ids),
        "mean_member_std": float(np.nanmean(std)),
        "agreement_fraction": {
            "at_least_4of6": float(np.mean(agreement >= 4.0 / 6.0)),
            "at_least_5of6": float(np.mean(agreement >= 5.0 / 6.0)),
            "unanimous": float(np.mean(agreement >= 1.0)),
        },
    }
    return streams, diagnostics


def signed_dynamic_policy_masks(
    scores: np.ndarray,
    positions: np.ndarray,
    size: int,
    *,
    side_policy: str,
    low_thresholds: np.ndarray,
    high_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    low_thresholds = np.asarray(low_thresholds, dtype=float)
    high_thresholds = np.asarray(high_thresholds, dtype=float)
    long_active = np.zeros(int(size), dtype=bool)
    short_active = np.zeros(int(size), dtype=bool)
    if side_policy in {"long", "both"}:
        active = (
            np.isfinite(scores)
            & np.isfinite(high_thresholds)
            & (scores > 0.0)
            & (scores >= high_thresholds)
        )
        long_active[positions[active]] = True
    if side_policy in {"short", "both"}:
        active = (
            np.isfinite(scores)
            & np.isfinite(low_thresholds)
            & (scores < 0.0)
            & (scores <= low_thresholds)
        )
        short_active[positions[active]] = True
    return long_active, short_active


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_paths(args.output, args.manifest_output)
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    if source_manifest.get("later_metrics_included") is not False:
        raise ValueError("source manifest must be frozen before later metrics")
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    cfg = Config(
        input_csv=args.input_csv,
        output=args.output,
        funding_csv=args.funding_csv,
        premium_csv=args.premium_csv,
        exclude_from=args.exclude_from,
    )
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    base = build_market_feature_frame(market, window_size=144)
    features = pd.concat([base, build_interest_features(market, base)], axis=1)
    features = features.loc[:, ~features.columns.duplicated(keep="last")]
    positions, targets, _ = anchor_dataset(market, features)
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    fit_mask = masks["fit2020_2022"]
    full_columns = feature_groups(features)["full"]
    full_matrix = features.iloc[positions][full_columns].to_numpy(float)
    ranking = stable_feature_ranking(
        full_matrix,
        full_columns,
        targets,
        signal_dates,
        fit_mask,
    )
    feature_sets = feature_set_definitions(ranking)
    reproduced_feature_sets = {
        name: [row["name"] for row in rows]
        for name, rows in feature_sets.items()
    }
    if reproduced_feature_sets != source_manifest["feature_sets"]:
        raise ValueError("feature sets do not reproduce source manifest")
    labels, label_thresholds = tail_labels(targets, fit_mask)
    if not np.allclose(label_thresholds, source_manifest["target"]["fit_thresholds"]):
        raise ValueError("tail thresholds do not reproduce source manifest")
    environments = half_year_environments(signal_dates)

    transformed_streams: dict[str, np.ndarray] = {}
    transform_metadata: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for feature_set_name, feature_rows in feature_sets.items():
        indices = [int(row["index"]) for row in feature_rows]
        matrix = full_matrix[:, indices]
        members: dict[str, np.ndarray] = {}
        member_diagnostics: dict[str, Any] = {}
        for model_id, architecture, objective, vrex_penalty in INVARIANT_MODEL_SPECS:
            full_model_id = f"{model_id}_{feature_set_name}"
            print(f"training {full_model_id}", file=sys.stderr, flush=True)
            scores, training_diagnostics = train_tail_classifier(
                matrix,
                labels,
                fit_mask,
                environments,
                architecture=architecture,
                objective=objective,
                vrex_penalty=vrex_penalty,
            )
            members[full_model_id] = scores
            member_diagnostics[full_model_id] = {
                "architecture": architecture,
                "objective": objective,
                "vrex_penalty": vrex_penalty,
                "training": training_diagnostics,
                "score_quality": _score_quality(scores, targets, masks),
            }
        streams, uncertainty_diagnostics = uncertainty_score_streams(members)
        diagnostics[feature_set_name] = {
            "members": member_diagnostics,
            "uncertainty": uncertainty_diagnostics,
            "transforms": {},
        }
        for transform_name, scores in streams.items():
            stream_id = f"invariant_{feature_set_name}_{transform_name}"
            transformed_streams[stream_id] = scores
            transform_metadata[stream_id] = {
                "feature_set": feature_set_name,
                "transform": transform_name,
                "member_count": len(members),
            }
            diagnostics[feature_set_name]["transforms"][transform_name] = {
                "score_quality": _score_quality(scores, targets, masks)
            }

    raw_candidates: list[dict[str, Any]] = []
    for stream_id, scores in transformed_streams.items():
        metadata = transform_metadata[stream_id]
        for rolling_window in ROLLING_WINDOWS:
            for quantile in SCORE_QUANTILES:
                low_thresholds, high_thresholds = causal_rolling_thresholds(
                    scores,
                    window=rolling_window,
                    quantile=quantile,
                )
                for side_policy in ("long", "short", "both"):
                    long_active, short_active = signed_dynamic_policy_masks(
                        scores,
                        positions,
                        len(market),
                        side_policy=side_policy,
                        low_thresholds=low_thresholds,
                        high_thresholds=high_thresholds,
                    )
                    holdout = sim(
                        market,
                        dates,
                        long_active,
                        short_active,
                        cfg,
                        HOLD_BARS,
                        ANCHOR_STRIDE,
                        10.0,
                        10.0,
                        "holdout2023",
                    )
                    if holdout["trades"] < 8 or holdout["return_pct"] <= 0.0:
                        continue
                    raw_candidates.append(
                        {
                            "stream_id": stream_id,
                            **metadata,
                            "rolling_score_window_anchors": rolling_window,
                            "score_quantile": quantile,
                            "side_policy": side_policy,
                            "hold_bars": HOLD_BARS,
                            "anchor_stride_bars": ANCHOR_STRIDE,
                            "holdout2023": holdout,
                            "signal_hash": effective_selection_signal_hash(
                                market,
                                dates,
                                long_active,
                                short_active,
                                window=WINDOWS["holdout2023"],
                            ),
                            "_long": long_active,
                            "_short": short_active,
                        }
                    )

    raw_candidates.sort(
        key=lambda row: (
            row["holdout2023"]["ratio"],
            row["holdout2023"]["return_pct"],
            row["holdout2023"]["trades"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in raw_candidates:
        signal_hash = candidate["signal_hash"]
        if signal_hash in selected_signals:
            continue
        selected.append(candidate)
        selected_signals[signal_hash] = (
            candidate.pop("_long"),
            candidate.pop("_short"),
        )
        if len(selected) == 10:
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "source_manifest": args.source_manifest,
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "selection_policy": (
            "reproduce train-only invariant member models; fixed disagreement transforms; "
            "shifted rolling percentiles; 2023 executed-path Top-10 freeze"
        ),
        "target": source_manifest["target"],
        "feature_sets": reproduced_feature_sets,
        "transforms": sorted(
            {metadata["transform"] for metadata in transform_metadata.values()}
        ),
        "top10": selected,
        "trial_counts": {
            "member_model_specs": len(INVARIANT_MODEL_SPECS),
            "feature_sets": len(feature_sets),
            "transformed_streams": len(transformed_streams),
            "rolling_windows": len(ROLLING_WINDOWS),
            "score_quantiles": len(SCORE_QUANTILES),
            "side_policies": 3,
            "total_policy_specs": len(transformed_streams)
            * len(ROLLING_WINDOWS)
            * len(SCORE_QUANTILES)
            * 3,
            "eligible_holdout_candidates": len(raw_candidates),
            "distinct_top10": len(selected),
        },
        "data_sha256": {
            "market": _file_sha256(args.input_csv),
            "funding": _file_sha256(args.funding_csv),
            "premium": _file_sha256(args.premium_csv),
        },
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for rank, row in enumerate(selected, start=1):
        long_active, short_active = selected_signals[row["signal_hash"]]
        row["pre_evaluation_rank"] = rank
        for split in ("test2024", "eval2025", "ytd2026"):
            row[split] = sim(
                market,
                dates,
                long_active,
                short_active,
                cfg,
                HOLD_BARS,
                ANCHOR_STRIDE,
                10.0,
                10.0,
                split,
            )
        row["passes_alpha_pool"] = bool(
            row["test2024"]["ratio"] >= 3.0
            and row["eval2025"]["ratio"] >= 3.0
            and row["test2024"]["trades"] >= 8
            and row["eval2025"]["trades"] >= 8
            and row["test2024"]["return_pct"] > 0.0
            and row["eval2025"]["return_pct"] > 0.0
        )
        row["passes_live_grade"] = bool(
            row["passes_alpha_pool"]
            and row["ytd2026"]["ratio"] >= 5.0
            and row["ytd2026"]["trades"] >= 6
            and row["ytd2026"]["return_pct"] > 0.0
        )

    alpha_pool, live_grade = top10_promotions(selected)
    cost_stress: dict[str, Any] = {}
    for row in live_grade:
        long_active, short_active = selected_signals[row["signal_hash"]]
        cost_stress[row["signal_hash"]] = {}
        for bps in (6, 8, 10, 15):
            stressed_cfg = replace(
                cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate)
            )
            cost_stress[row["signal_hash"]][str(bps)] = {
                split: sim(
                    market,
                    dates,
                    long_active,
                    short_active,
                    stressed_cfg,
                    HOLD_BARS,
                    ANCHOR_STRIDE,
                    10.0,
                    10.0,
                    split,
                )
                for split in ("test2024", "eval2025", "ytd2026")
            }

    output = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "source_manifest": args.source_manifest,
        "protocol": manifest["selection_policy"]
        + "; next-bar entry; hold576; 6bp/side; full-window CAGR; strict intratrade MDD",
        "manifest": str(manifest_path),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "diagnostics": diagnostics,
        "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "tested_candidates": len(raw_candidates),
        "selected": selected,
        "alpha_pool_qualifiers": alpha_pool,
        "live_grade": live_grade,
        "cost_stress_bps_per_side": cost_stress,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    args = parser.parse_args()
    output = run(args)
    print(
        json.dumps(
            {
                "tested_candidates": output["tested_candidates"],
                "selected": len(output["selected"]),
                "alpha_pool": len(output["alpha_pool_qualifiers"]),
                "live_grade": len(output["live_grade"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
