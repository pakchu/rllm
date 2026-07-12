"""MOMENT-1-small strict-causal embedding probe alpha search.

The frozen MOMENT encoder receives only 512 completed hourly bins at each
signal anchor.  PCA and invariant tail probes are fit only on 2020-2022;
2023 is used solely to freeze a de-duplicated Top-10 manifest before any
2024+ diagnostics or simulation metrics are computed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_embedding_probe_alpha import (
    PCA_DIMS,
    fit_pca_representations,
    optional_file_sha256,
)
from training.search_chronos2_zero_shot_alpha import (
    COVARIATE_COLUMNS,
    ROLLING_WINDOWS,
    SCORE_QUANTILES,
    anchor_hour_indices,
    causal_hourly_frame,
)
from training.search_invariant_groupdro_alpha import (
    MODEL_SPECS,
    _score_quality,
    half_year_environments,
    train_tail_classifier,
)
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash
from training.search_river_online_alpha import causal_rolling_thresholds, validate_output_paths
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


MODEL_ID = "AutonLab/MOMENT-1-small"
MODEL_REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
MODEL_SOURCE_URLS = {
    "huggingface": "https://huggingface.co/AutonLab/MOMENT-1-small",
    "code": "https://github.com/moment-timeseries-foundation-model/moment",
    "paper": "https://arxiv.org/abs/2402.03885",
}
CONTEXT_HOURS = 512
EMBEDDING_VARIATES = ("log_close", *COVARIATE_COLUMNS)


def strict_validate_output_paths(output: str, manifest_output: str) -> None:
    """Refuse ambiguous or overwriting outputs before expensive inference."""
    validate_output_paths(output, manifest_output)
    for path in (Path(output), Path(manifest_output)):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def fit_only_tail_labels(
    targets: np.ndarray,
    fit_mask: np.ndarray,
    *,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Create labels only for fit rows; later targets remain entirely unused."""
    targets = np.asarray(targets, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_indices = np.flatnonzero(fit_mask)
    fit_values = targets[fit_indices]
    finite_fit = np.isfinite(fit_values)
    if int(finite_fit.sum()) < 3:
        raise ValueError("not enough finite fit targets for tail labels")
    low, high = np.quantile(
        fit_values[finite_fit], (float(low_quantile), float(high_quantile))
    )
    labels = np.ones(len(targets), dtype=np.int64)
    labels[fit_indices[fit_values <= low]] = 0
    labels[fit_indices[fit_values >= high]] = 2
    return labels, (float(low), float(high))


def summarize_moment_embedding(embedding: torch.Tensor) -> np.ndarray:
    """Summarize unreduced MOMENT states while preserving channel identity.

    MOMENT ``reduction='none'`` embeddings are expected as channels x patches x
    hidden-dim for one sample.  The summary concatenates, in channel order, each
    channel's mean-over-patches and each channel's final patch embedding.
    """
    if embedding.ndim != 3:
        raise ValueError("expected one MOMENT embedding with shape (channels, patches, dim)")
    if embedding.shape[0] != len(EMBEDDING_VARIATES) or embedding.shape[1] < 1:
        raise ValueError("unexpected MOMENT embedding channel/patch shape")
    per_channel_mean = embedding.mean(dim=1)
    per_channel_final = embedding[:, -1, :]
    return (
        torch.cat(
            [per_channel_mean.reshape(-1), per_channel_final.reshape(-1)],
            dim=0,
        )
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )


def summarize_moment_embedding_batch(embeddings: torch.Tensor) -> np.ndarray:
    """Vectorized summary for B,C,P,D unreduced MOMENT embeddings."""
    if embeddings.ndim != 4:
        raise ValueError("expected MOMENT embeddings with shape (batch, channels, patches, dim)")
    if embeddings.shape[1] != len(EMBEDDING_VARIATES) or embeddings.shape[2] < 1:
        raise ValueError("unexpected MOMENT embedding batch shape")
    per_channel_mean = embeddings.mean(dim=2)
    per_channel_final = embeddings[:, :, -1, :]
    summary = torch.cat(
        [
            per_channel_mean.reshape(embeddings.shape[0], -1),
            per_channel_final.reshape(embeddings.shape[0], -1),
        ],
        dim=1,
    )
    return summary.detach().to(dtype=torch.float32).cpu().numpy()


def _moment_embed_batch(pipeline: Any, batch: np.ndarray, *, device: str) -> torch.Tensor:
    x_enc = torch.as_tensor(batch, dtype=torch.float32, device=device)
    input_mask = torch.ones((x_enc.shape[0], x_enc.shape[2]), dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs = pipeline.embed(x_enc=x_enc, input_mask=input_mask, reduction="none")
    embeddings = getattr(outputs, "embeddings", outputs)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("MOMENT embed output did not contain a tensor embedding")
    if embeddings.ndim != 4:
        raise ValueError("MOMENT reduction='none' must return B,C,P,D embeddings")
    return embeddings


def extract_embedding_summaries(
    pipeline: Any,
    hourly: pd.DataFrame,
    hour_indices: np.ndarray,
    *,
    context_hours: int,
    chunk_size: int,
    batch_size: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract strict-causal MOMENT summaries for valid anchors only.

    For anchor ``i`` with latest completed hourly bin ``end - 1``, the model
    input is exactly ``values[:, end-context:end]``.  No later hourly value is
    read or batched for that anchor.
    """
    values = hourly.loc[:, list(EMBEDDING_VARIATES)].to_numpy(np.float32).T
    valid_anchor_indices = np.flatnonzero(hour_indices + 1 >= int(context_hours))
    summaries: np.ndarray | None = None
    completed_chunks = 0
    for chunk_start in range(0, len(valid_anchor_indices), int(chunk_size)):
        chunk_indices = valid_anchor_indices[chunk_start : chunk_start + int(chunk_size)]
        chunk_summaries: list[np.ndarray] = []
        for batch_start in range(0, len(chunk_indices), int(batch_size)):
            batch_indices = chunk_indices[batch_start : batch_start + int(batch_size)]
            inputs = []
            for anchor_index in batch_indices:
                end = int(hour_indices[anchor_index]) + 1
                start = end - int(context_hours)
                inputs.append(values[:, start:end])
            batch = np.stack(inputs, axis=0)
            embeddings = _moment_embed_batch(pipeline, batch, device=device)
            chunk_summaries.append(summarize_moment_embedding_batch(embeddings))
        if not chunk_summaries:
            continue
        stacked = np.concatenate(chunk_summaries, axis=0).astype(np.float32, copy=False)
        if summaries is None:
            summaries = np.full((len(hour_indices), stacked.shape[1]), np.nan, dtype=np.float32)
        summaries[chunk_indices] = stacked
        completed_chunks += 1
        if completed_chunks % 4 == 0:
            print(
                f"MOMENT embedding chunks completed: {completed_chunks}",
                file=sys.stderr,
                flush=True,
            )
    if summaries is None:
        raise RuntimeError("no valid MOMENT contexts")
    return summaries, valid_anchor_indices, {
        "raw_summary_dim": int(summaries.shape[1]),
        "valid_anchor_count": int(len(valid_anchor_indices)),
        "embedding_variates": list(EMBEDDING_VARIATES),
        "summary_tokens": [
            "per_channel_mean_over_patches_flattened_in_variate_order",
            "per_channel_final_patch_flattened_in_variate_order",
        ],
        "embedding_shape_expected": "B,C,P,D from MOMENTPipeline.embed(reduction='none')",
        "causal_context_rule": "for each completed-hour anchor end, use values[:, end-context:end] only",
    }


def _load_moment_pipeline(model_id: str, revision: str) -> Any:
    from momentfm import MOMENTPipeline

    pipeline = MOMENTPipeline.from_pretrained(
        model_id,
        revision=revision,
        model_kwargs={"task_name": "embedding"},
    )
    pipeline.init()
    pipeline.cuda()
    pipeline.eval()
    return pipeline


def _model_commit_hash(pipeline: Any) -> str | None:
    for owner in (pipeline, getattr(pipeline, "model", None), getattr(pipeline, "encoder", None)):
        config = getattr(owner, "config", None)
        commit = getattr(config, "_commit_hash", None) if config is not None else None
        if commit:
            return str(commit)
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict_validate_output_paths(args.output, args.manifest_output)
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
    masks = {name: split_mask_for_anchors(dates, positions, *bounds) for name, bounds in WINDOWS.items()}
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)

    pipeline = _load_moment_pipeline(args.model_id, args.model_revision)
    model_context = int(getattr(pipeline.config, "seq_len", CONTEXT_HOURS))
    if int(args.context_hours) != model_context:
        raise ValueError(
            f"MOMENT context must match pretrained seq_len={model_context}, "
            f"got {args.context_hours}"
        )
    model_commit = _model_commit_hash(pipeline)
    if model_commit is not None and model_commit != args.model_revision:
        raise ValueError("loaded MOMENT model revision does not match pinned revision")
    summaries, valid_anchor_indices, embedding_metadata = extract_embedding_summaries(
        pipeline,
        hourly,
        hour_indices,
        context_hours=args.context_hours,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device="cuda",
    )
    del pipeline
    torch.cuda.empty_cache()

    valid_mask = np.zeros(len(positions), dtype=bool)
    valid_mask[valid_anchor_indices] = True
    pca_representations, pca_metadata = fit_pca_representations(
        summaries,
        masks["fit2020_2022"],
        valid_mask,
        dimensions=PCA_DIMS,
    )
    fit_valid_mask = masks["fit2020_2022"] & valid_mask
    labels, label_thresholds = fit_only_tail_labels(targets, fit_valid_mask)
    environments = half_year_environments(signal_dates)

    prefreeze_masks = {name: masks[name] for name in ("fit2020_2022", "holdout2023")}
    score_streams: dict[str, np.ndarray] = {}
    model_diagnostics: dict[str, Any] = {}
    for representation_name, representation in pca_representations.items():
        for model_id, architecture, objective, vrex_penalty in MODEL_SPECS:
            stream_id = f"moment_{representation_name}_{model_id}"
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
                "score_quality": _score_quality(scores, targets, prefreeze_masks),
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
        selected_signals[signal_hash] = (candidate.pop("_long"), candidate.pop("_short"))
        if len(selected) == 10:
            break

    model_metadata = {
        "id": args.model_id,
        "revision": args.model_revision,
        "loaded_commit_hash": model_commit,
        "source_urls": MODEL_SOURCE_URLS,
        "loader": "MOMENTPipeline.from_pretrained(model_kwargs={'task_name':'embedding'}); init(); cuda(); eval()",
        "momentfm_version": importlib.metadata.version("momentfm"),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "model": model_metadata,
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
        "strict_protocol": {
            "fit_window": "fit2020_2022 only for PCA and probe fitting",
            "selection_window": "holdout2023 only; positive return and >=8 trades required",
            "future_diagnostics_before_manifest": False,
            "future_metric_keys_allowed_in_manifest": False,
            "dedupe": "executed-path signal_hash on holdout2023 actual long/short masks",
            "promotion": "existing top10_promotions over frozen Top-10 after future metrics are computed",
        },
        "selection_policy": (
            "frozen MOMENT-1-small embeddings; PCA fit on 2020-2022 only; fixed "
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
            "total_policy_specs": len(score_streams) * len(ROLLING_WINDOWS) * len(SCORE_QUANTILES) * 3,
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
        model_diagnostics[stream_id]["score_quality"] = _score_quality(scores, targets, masks)

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
            stressed_cfg = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
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
        "source": MODEL_SOURCE_URLS,
        "protocol": manifest["selection_policy"]
        + "; completed-hour causal input; next-bar 5m entry; hold576; 6bp/side; "
        "full-window CAGR; strict intratrade MDD",
        "manifest": str(manifest_path),
        "model": model_metadata,
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
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--context-hours", type=int, default=CONTEXT_HOURS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
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
