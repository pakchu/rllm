"""Train-window orientation calibration for Chronos-2 zero-shot scores."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_zero_shot_alpha import (
    CONTEXT_HOURS,
    MODEL_ID,
    PREDICTION_HOURS,
    ROLLING_WINDOWS,
    SCORE_QUANTILES,
    anchor_hour_indices,
    build_chronos_inputs,
    causal_hourly_frame,
    forecast_score_streams,
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


def fit_score_orientation(
    scores: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Orient a score using only fit-window rank correlation."""
    finite = np.asarray(fit_mask, dtype=bool) & np.isfinite(scores) & np.isfinite(targets)
    if int(finite.sum()) < 3:
        raise ValueError("not enough fit observations to orient score")
    correlation = spearmanr(scores[finite], targets[finite]).statistic
    if not np.isfinite(correlation) or abs(float(correlation)) < 1e-12:
        orientation = 1
    else:
        orientation = 1 if correlation > 0.0 else -1
    return orientation * np.asarray(scores, dtype=float), {
        "fit_samples": int(finite.sum()),
        "fit_spearman_before_orientation": float(correlation),
        "orientation": orientation,
        "fit_spearman_after_orientation": float(orientation * correlation),
    }


def _score_quality(
    scores: np.ndarray,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for split, split_mask in masks.items():
        finite = split_mask & np.isfinite(scores) & np.isfinite(targets)
        correlation = (
            spearmanr(scores[finite], targets[finite]).statistic
            if int(finite.sum()) >= 3
            else np.nan
        )
        output[split] = {
            "samples": int(finite.sum()),
            "spearman": float(correlation) if np.isfinite(correlation) else None,
        }
    return output


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
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)
    inputs, valid_anchor_indices = build_chronos_inputs(
        hourly,
        hour_indices,
        context_hours=args.context_hours,
    )
    entry_log_prices = np.log(market["open"].to_numpy(float)[positions + 1])

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda")
    model_commit = getattr(pipeline.model.config, "_commit_hash", None)
    if model_commit != source_manifest["model"]["revision"]:
        raise ValueError("model revision does not match source manifest")
    batch_count = 0

    def progress() -> None:
        nonlocal batch_count
        batch_count += 1
        if batch_count % 25 == 0:
            print(f"chronos batches completed: {batch_count}", file=sys.stderr, flush=True)

    predictions = pipeline.predict(
        inputs,
        prediction_length=PREDICTION_HOURS,
        batch_size=args.batch_size,
        context_length=args.context_hours,
        cross_learning=False,
        after_batch=progress,
    )
    raw_streams = forecast_score_streams(
        predictions,
        valid_anchor_indices,
        entry_log_prices,
        pipeline.quantiles,
        len(positions),
    )
    del predictions
    del pipeline
    torch.cuda.empty_cache()

    oriented_streams: dict[str, np.ndarray] = {}
    orientation_metadata: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for stream_name, raw_scores in raw_streams.items():
        oriented, metadata = fit_score_orientation(
            raw_scores,
            targets,
            masks["fit2020_2022"],
        )
        oriented_streams[stream_name] = oriented
        orientation_metadata[stream_name] = metadata
        diagnostics[stream_name] = _score_quality(oriented, targets, masks)

    raw_candidates: list[dict[str, Any]] = []
    for stream_name, scores in oriented_streams.items():
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
                            "stream_name": stream_name,
                            "fit_orientation": orientation_metadata[stream_name],
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
        "input": source_manifest["input"],
        "orientation": orientation_metadata,
        "selection_policy": (
            "orient each zero-shot score by fit2020-2022 Spearman sign only; shifted "
            "rolling percentiles; 2023 executed-path Top-10 freeze before 2024+ metrics"
        ),
        "top10": selected,
        "trial_counts": {
            "score_streams": len(oriented_streams),
            "rolling_windows": len(ROLLING_WINDOWS),
            "score_quantiles": len(SCORE_QUANTILES),
            "side_policies": 3,
            "total_policy_specs": len(oriented_streams)
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
        "source": "https://github.com/amazon-science/chronos-forecasting",
        "protocol": manifest["selection_policy"]
        + "; completed-hour causal input; next-bar 5m entry; hold576; 6bp/side; "
        "full-window CAGR; strict intratrade MDD",
        "manifest": str(manifest_path),
        "orientation": orientation_metadata,
        "score_diagnostics": diagnostics,
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
