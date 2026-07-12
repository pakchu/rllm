"""Causal rolling-calibration follow-up for invariant tail classifiers."""
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
    dynamic_policy_masks,
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


def ensemble_score_streams(
    individual_scores: dict[str, np.ndarray],
    model_metadata: dict[str, dict[str, str]],
) -> dict[str, np.ndarray]:
    """Create fixed objective ensembles within each train-only feature set."""
    ensembles: dict[str, np.ndarray] = {}
    feature_sets = sorted({metadata["feature_set"] for metadata in model_metadata.values()})
    for feature_set in feature_sets:
        members = [
            model_id
            for model_id, metadata in model_metadata.items()
            if metadata["feature_set"] == feature_set
        ]
        groups = {
            "all": members,
            "mlp": [
                model_id
                for model_id in members
                if model_metadata[model_id]["architecture"] == "mlp"
            ],
            "invariant": [
                model_id
                for model_id in members
                if model_metadata[model_id]["objective"] in {"vrex", "groupdro"}
            ],
        }
        for group_name, group_members in groups.items():
            if len(group_members) < 2:
                continue
            ensembles[f"ensemble_{group_name}_{feature_set}"] = np.nanmean(
                np.stack([individual_scores[model_id] for model_id in group_members]),
                axis=0,
            )
    return ensembles


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_paths(args.output, args.manifest_output)
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    if source_manifest.get("later_metrics_included") is not False:
        raise ValueError("source invariant manifest must be frozen before later metrics")
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
        raise ValueError("train-only feature ranking does not reproduce source manifest")
    labels, label_thresholds = tail_labels(targets, fit_mask)
    if not np.allclose(label_thresholds, source_manifest["target"]["fit_thresholds"]):
        raise ValueError("tail label thresholds do not reproduce source manifest")
    environments = half_year_environments(signal_dates)

    score_streams: dict[str, np.ndarray] = {}
    stream_metadata: dict[str, dict[str, Any]] = {}
    model_diagnostics: dict[str, Any] = {}
    for feature_set_name, feature_rows in feature_sets.items():
        indices = [int(row["index"]) for row in feature_rows]
        matrix = full_matrix[:, indices]
        for model_id, architecture, objective, vrex_penalty in MODEL_SPECS:
            full_model_id = f"{model_id}_{feature_set_name}"
            print(f"training {full_model_id}", file=sys.stderr, flush=True)
            scores, diagnostics = train_tail_classifier(
                matrix,
                labels,
                fit_mask,
                environments,
                architecture=architecture,
                objective=objective,
                vrex_penalty=vrex_penalty,
            )
            score_streams[full_model_id] = scores
            stream_metadata[full_model_id] = {
                "stream_kind": "individual",
                "architecture": architecture,
                "objective": objective,
                "feature_set": feature_set_name,
            }
            model_diagnostics[full_model_id] = {
                **stream_metadata[full_model_id],
                "vrex_penalty": vrex_penalty,
                "feature_names": [row["name"] for row in feature_rows],
                "training": diagnostics,
                "score_quality": _score_quality(scores, targets, masks),
            }

    ensembles = ensemble_score_streams(score_streams, stream_metadata)
    for ensemble_id, scores in ensembles.items():
        feature_set_name = ensemble_id.rsplit("_", 1)[-1]
        score_streams[ensemble_id] = scores
        stream_metadata[ensemble_id] = {
            "stream_kind": "ensemble",
            "architecture": "ensemble",
            "objective": ensemble_id.split("_")[1],
            "feature_set": feature_set_name,
        }
        model_diagnostics[ensemble_id] = {
            **stream_metadata[ensemble_id],
            "score_quality": _score_quality(scores, targets, masks),
        }

    raw_candidates: list[dict[str, Any]] = []
    for stream_id, scores in score_streams.items():
        metadata = stream_metadata[stream_id]
        for rolling_window in ROLLING_WINDOWS:
            for quantile in SCORE_QUANTILES:
                low_thresholds, high_thresholds = causal_rolling_thresholds(
                    scores,
                    window=rolling_window,
                    quantile=quantile,
                )
                for side_policy in ("long", "short", "both"):
                    long_active, short_active = dynamic_policy_masks(
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
            "reproduce train-only invariant models; fixed individual/objective ensembles; "
            "shifted 180/365d score percentiles; 2023 executed-path Top-10 freeze"
        ),
        "target": source_manifest["target"],
        "feature_sets": reproduced_feature_sets,
        "ensemble_definitions": [
            stream_id for stream_id in score_streams if stream_id.startswith("ensemble_")
        ],
        "top10": selected,
        "trial_counts": {
            "individual_streams": len(score_streams) - len(ensembles),
            "ensemble_streams": len(ensembles),
            "rolling_windows": len(ROLLING_WINDOWS),
            "score_quantiles": len(SCORE_QUANTILES),
            "side_policies": 3,
            "total_policy_specs": len(score_streams)
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
        "model_diagnostics": model_diagnostics,
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
