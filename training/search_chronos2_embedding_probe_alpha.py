"""Chronos-2 encoder representation plus invariant policy-probe alpha search."""
from __future__ import annotations

import argparse
import hashlib
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
from sklearn.decomposition import PCA

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_zero_shot_alpha import (
    CONTEXT_HOURS,
    COVARIATE_COLUMNS,
    MODEL_ID,
    ROLLING_WINDOWS,
    SCORE_QUANTILES,
    anchor_hour_indices,
    causal_hourly_frame,
)
from training.search_invariant_groupdro_alpha import (
    MODEL_SPECS,
    _score_quality,
    half_year_environments,
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
    split_mask_for_anchors,
    top10_promotions,
)


PCA_DIMS = (16, 32, 64)
EMBEDDING_VARIATES = ("log_close", *COVARIATE_COLUMNS)


def summarize_chronos_embedding(embedding: torch.Tensor) -> np.ndarray:
    """Compress patch/token states without using labels or future values."""
    if embedding.ndim != 3 or embedding.shape[1] < 3:
        raise ValueError("expected embedding shape (variates, patches+2, model_dim)")
    target_reg = embedding[0, -2]
    target_output = embedding[0, -1]
    group_reg = embedding[:, -2].mean(dim=0)
    group_patch = embedding[:, :-2].mean(dim=(0, 1))
    return (
        torch.cat([target_reg, target_output, group_reg, group_patch])
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def extract_embedding_summaries(
    pipeline: Any,
    hourly: pd.DataFrame,
    hour_indices: np.ndarray,
    *,
    context_hours: int,
    chunk_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = hourly.loc[:, list(EMBEDDING_VARIATES)].to_numpy(np.float32).T
    valid_anchor_indices = np.flatnonzero(hour_indices + 1 >= int(context_hours))
    summaries: np.ndarray | None = None
    completed_chunks = 0
    for chunk_start in range(0, len(valid_anchor_indices), int(chunk_size)):
        chunk_indices = valid_anchor_indices[chunk_start : chunk_start + int(chunk_size)]
        inputs = []
        for anchor_index in chunk_indices:
            end = int(hour_indices[anchor_index]) + 1
            start = end - int(context_hours)
            inputs.append(values[:, start:end])
        embeddings, _ = pipeline.embed(
            inputs,
            batch_size=int(batch_size),
            context_length=int(context_hours),
        )
        chunk_summaries = np.stack(
            [summarize_chronos_embedding(embedding) for embedding in embeddings]
        )
        if summaries is None:
            summaries = np.full(
                (len(hour_indices), chunk_summaries.shape[1]),
                np.nan,
                dtype=np.float32,
            )
        summaries[chunk_indices] = chunk_summaries
        completed_chunks += 1
        if completed_chunks % 4 == 0:
            print(
                f"Chronos embedding chunks completed: {completed_chunks}",
                file=sys.stderr,
                flush=True,
            )
    if summaries is None:
        raise RuntimeError("no valid Chronos contexts")
    return summaries, valid_anchor_indices, {
        "raw_summary_dim": summaries.shape[1],
        "valid_anchor_count": len(valid_anchor_indices),
        "embedding_variates": list(EMBEDDING_VARIATES),
        "summary_tokens": [
            "target_reg",
            "target_masked_output",
            "mean_group_reg",
            "mean_group_patch",
        ],
    }


def fit_pca_representations(
    summaries: np.ndarray,
    fit_mask: np.ndarray,
    valid_mask: np.ndarray,
    dimensions: tuple[int, ...] = PCA_DIMS,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fit_valid = np.asarray(fit_mask, dtype=bool) & np.asarray(valid_mask, dtype=bool)
    fit_rows = np.asarray(summaries, dtype=float)[fit_valid]
    if len(fit_rows) < max(dimensions):
        raise ValueError("not enough fit embeddings for requested PCA dimensions")
    output: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for dimension in dimensions:
        pca = PCA(
            n_components=int(dimension),
            whiten=True,
            svd_solver="randomized",
            random_state=713,
        )
        pca.fit(fit_rows)
        transformed = np.full((len(summaries), int(dimension)), np.nan, dtype=np.float32)
        transformed[valid_mask] = pca.transform(
            np.asarray(summaries, dtype=float)[valid_mask]
        ).astype(np.float32)
        key = f"pca{dimension}"
        output[key] = transformed
        component_bytes = np.ascontiguousarray(pca.components_).tobytes()
        metadata[key] = {
            "components": int(dimension),
            "fit_samples": int(fit_valid.sum()),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "components_sha256": hashlib.sha256(component_bytes).hexdigest(),
        }
    return output, metadata


def optional_file_sha256(path: str) -> str | None:
    return _file_sha256(path) if path else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_paths(args.output, args.manifest_output)
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    if source_manifest.get("later_metrics_included") is not False:
        raise ValueError("source Chronos manifest must be frozen")
    if source_manifest["model"]["id"] != args.model_id:
        raise ValueError("model id does not match source manifest")
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
    positions, targets, _ = anchor_dataset(market, pd.DataFrame(index=market.index))
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)

    from chronos import Chronos2Pipeline

    pinned_revision = source_manifest["model"]["revision"]
    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_id,
        revision=pinned_revision,
        device_map="cuda",
    )
    model_commit = getattr(pipeline.model.config, "_commit_hash", None)
    if model_commit != pinned_revision:
        raise ValueError("model revision does not match source manifest")
    summaries, valid_anchor_indices, embedding_metadata = extract_embedding_summaries(
        pipeline,
        hourly,
        hour_indices,
        context_hours=args.context_hours,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
    del pipeline
    torch.cuda.empty_cache()
    valid_mask = np.zeros(len(positions), dtype=bool)
    valid_mask[valid_anchor_indices] = True
    pca_representations, pca_metadata = fit_pca_representations(
        summaries,
        masks["fit2020_2022"],
        valid_mask,
    )
    fit_valid_mask = masks["fit2020_2022"] & valid_mask
    labels, label_thresholds = tail_labels(targets, fit_valid_mask)
    environments = half_year_environments(signal_dates)

    prefreeze_masks = {
        name: masks[name] for name in ("fit2020_2022", "holdout2023")
    }
    score_streams: dict[str, np.ndarray] = {}
    model_diagnostics: dict[str, Any] = {}
    for representation_name, representation in pca_representations.items():
        for model_id, architecture, objective, vrex_penalty in MODEL_SPECS:
            stream_id = f"chronos_{representation_name}_{model_id}"
            print(f"training {stream_id}", file=sys.stderr, flush=True)
            scores, diagnostics = train_tail_classifier(
                representation,
                labels,
                fit_valid_mask,
                environments,
                architecture=architecture,
                objective=objective,
                vrex_penalty=vrex_penalty,
            )
            scores[~valid_mask] = np.nan
            score_streams[stream_id] = scores
            model_diagnostics[stream_id] = {
                "representation": representation_name,
                "architecture": architecture,
                "objective": objective,
                "vrex_penalty": vrex_penalty,
                "training": diagnostics,
                "score_quality": _score_quality(
                    scores, targets, prefreeze_masks
                ),
            }

    raw_candidates: list[dict[str, Any]] = []
    for stream_id, scores in score_streams.items():
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
                            "representation": model_diagnostics[stream_id]["representation"],
                            "architecture": model_diagnostics[stream_id]["architecture"],
                            "objective": model_diagnostics[stream_id]["objective"],
                            "vrex_penalty": model_diagnostics[stream_id]["vrex_penalty"],
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
        "model": source_manifest["model"],
        "input": {
            "frequency": "1h completed candles",
            "context_hours": args.context_hours,
            "embedding_variates": list(EMBEDDING_VARIATES),
            **embedding_metadata,
        },
        "representation": pca_metadata,
        "target": {
            "name": "three_class_next_48h_return_tail",
            "fit_quantiles": [0.3, 0.7],
            "fit_thresholds": list(label_thresholds),
        },
        "selection_policy": (
            "frozen Chronos-2 encoder embeddings; PCA fit on 2020-2022 only; fixed "
            "ERM/V-REx/Group-DRO probes; shifted rolling percentiles; 2023 "
            "executed-path Top-10 freeze before 2024+ metrics"
        ),
        "top10": selected,
        "trial_counts": {
            "representations": len(pca_representations),
            "probe_specs": len(MODEL_SPECS),
            "score_streams": len(score_streams),
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
            "funding": optional_file_sha256(args.funding_csv),
            "premium": optional_file_sha256(args.premium_csv),
        },
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Future-target diagnostics are intentionally unavailable until after the
    # pre-evaluation Top-10 manifest has been durably frozen.
    for stream_id, scores in score_streams.items():
        model_diagnostics[stream_id]["score_quality"] = _score_quality(
            scores, targets, masks
        )

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
        "source": "https://github.com/amazon-science/chronos-forecasting",
        "protocol": manifest["selection_policy"]
        + "; completed-hour causal input; next-bar 5m entry; hold576; 6bp/side; "
        "full-window CAGR; strict intratrade MDD",
        "manifest": str(manifest_path),
        "model": manifest["model"],
        "input": manifest["input"],
        "representation": pca_metadata,
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
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--context-hours", type=int, default=CONTEXT_HOURS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
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
