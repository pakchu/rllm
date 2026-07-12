"""Strict direct-utility MOMENT PCA32 retrieval alpha evaluation.

This experiment reuses the delayed dense-retrieval path-distribution machinery
from ``search_moment_retrieval_path_critic_alpha`` but removes all rolling
percentile/quantile gates.  A signed utility score is executable directly:
positive means long, negative means short, zero means flat, subject only to the
fixed side policy.

Protocol is two-phase.  Phase 1 extracts/features/targets/retrieval only for
pre-2024 signal rows and freezes a 2023 Top-10 manifest.  Phase 2 reruns only
the selected retrieval specs from scratch on full data, verifies the frozen 2023
executable path hashes, and then computes 2024+ strict metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import training.search_bidirectional_state_alpha as state_sim
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_embedding_probe_alpha import optional_file_sha256
from training.search_invariant_groupdro_alpha import _score_quality
from training.search_moment_continual_probe_alpha import assert_data_hashes_match_source, assert_pca_hashes_match_source, strict_validate_paths
from training.search_moment_embedding_probe_alpha import CONTEXT_HOURS, EMBEDDING_VARIATES, MODEL_ID, MODEL_REVISION
from training.search_moment_mamba2_alpha import SEED, extract_pca32_for_phase, first_post_fit_index, seed_everything
from training.search_moment_retrieval_path_critic_alpha import (
    K_VALUES,
    PCA_DIMS,
    REPRESENTATIONS,
    SIDES,
    SOURCE_MANIFEST_NAME_PREFIX,
    TARGET_COMPONENTS,
    UTILITY_SPECS,
    RetrievalSpec,
    compact_retrieval_diagnostics,
    denormalize_sort_and_clamp_quantiles,
    executable_path_targets_48h,
    fit_admissible_mask,
    fit_component_scales,
    load_required_source_manifest,
    retrieval_path_quantiles,
    round_trip_unlevered_cost,
    select_distinct_top10,
    signed_utility_scores,
)
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash
from training.search_tabicl_foundation_alpha import ANCHOR_STRIDE, HOLD_BARS, WINDOWS, _file_sha256, _git_head, anchor_dataset, split_mask_for_anchors, top10_promotions

DIRECT_POLICY_DESCRIPTION = "finite score > 0 goes long, finite score < 0 goes short, zero/NaN flat; side policy may disable either side"
TOTAL_POLICY_SPECS = len(REPRESENTATIONS) * len(K_VALUES) * len(UTILITY_SPECS) * len(SIDES)


def direct_policy_masks(scores: np.ndarray, positions: np.ndarray, market_size: int, side_policy: Literal["long", "short", "both"]) -> tuple[np.ndarray, np.ndarray]:
    """Map signed utility scores directly to executable long/short anchor masks."""
    if side_policy not in SIDES:
        raise ValueError(f"unknown side policy: {side_policy}")
    scores = np.asarray(scores, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    if len(scores) != len(positions):
        raise ValueError("scores and positions length mismatch")
    long_active = np.zeros(int(market_size), dtype=bool)
    short_active = np.zeros(int(market_size), dtype=bool)
    finite = np.isfinite(scores)
    if side_policy in {"long", "both"}:
        long_active[positions[finite & (scores > 0.0)]] = True
    if side_policy in {"short", "both"}:
        short_active[positions[finite & (scores < 0.0)]] = True
    return long_active, short_active


def direct_policy_family() -> list[dict[str, Any]]:
    """Return the fixed 48-member direct policy family, without threshold gates."""
    family: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for k in K_VALUES:
            spec = RetrievalSpec(representation, k)  # type: ignore[arg-type]
            for utility_name, _mode, _lam in UTILITY_SPECS:
                for side_policy in SIDES:
                    family.append({
                        "stream_id": f"{spec.stream_prefix}_{utility_name}",
                        "retrieval_representation": representation,
                        "k": int(k),
                        "utility_score": utility_name,
                        "side_policy": side_policy,
                        "policy_rule": DIRECT_POLICY_DESCRIPTION,
                    })
    return family


def selection_window_signal_hash(long_active: np.ndarray, short_active: np.ndarray, *, market: pd.DataFrame, dates: pd.Series) -> str:
    return effective_selection_signal_hash(market, dates, long_active, short_active, window=WINDOWS["holdout2023"])


def build_direct_candidates(
    phase_scores: dict[str, np.ndarray],
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    positions: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    """Evaluate all fixed direct policies and keep positive 2023 candidates."""
    raw_candidates: list[dict[str, Any]] = []
    for policy in direct_policy_family():
        scores = phase_scores[policy["stream_id"]]
        long_active, short_active = direct_policy_masks(scores, positions, len(market), policy["side_policy"])
        holdout = sim(market, dates, long_active, short_active, cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, "holdout2023")
        if holdout["trades"] < 8 or holdout["return_pct"] <= 0.0:
            continue
        raw_candidates.append({
            **policy,
            "hold_bars": HOLD_BARS,
            "anchor_stride_bars": ANCHOR_STRIDE,
            "holdout2023": holdout,
            "signal_hash": selection_window_signal_hash(long_active, short_active, market=market, dates=dates),
            "_long": long_active,
            "_short": short_active,
        })
    return raw_candidates


def _score_streams_from_retrieval(
    rep: np.ndarray,
    valid_mask: np.ndarray,
    targets: np.ndarray,
    dates: pd.Series,
    positions: np.ndarray,
    fit_mask: np.ndarray,
    cfg: Config,
    *,
    start_index: int,
    stop_index: int | None,
    selected_specs: set[tuple[str, int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores_by_stream: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    specs = selected_specs if selected_specs is not None else {(str(r), int(k)) for r in REPRESENTATIONS for k in K_VALUES}
    for kind, k in sorted(specs):
        spec = RetrievalSpec(kind, k)  # type: ignore[arg-type]
        print(f"retrieving {spec.stream_prefix}", file=sys.stderr, flush=True)
        raw_q, diag = retrieval_path_quantiles(rep, valid_mask, targets, dates, positions, fit_mask, spec, start_index=start_index, stop_index=stop_index)
        raw_q[:start_index] = np.nan
        q_pred = denormalize_sort_and_clamp_quantiles(raw_q, np.ones(len(TARGET_COMPONENTS), dtype=np.float32))
        streams, utility_meta = signed_utility_scores(q_pred, cfg)
        for utility_name, scores in streams.items():
            stream_id = f"{spec.stream_prefix}_{utility_name}"
            scores[:start_index] = np.nan
            scores_by_stream[stream_id] = scores
            diagnostics[stream_id] = {
                "retrieval": compact_retrieval_diagnostics(diag),
                "utility": utility_name,
                "utility_cost": utility_meta,
                "first_policy_score_index": start_index,
                "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all()),
            }
    return scores_by_stream, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(SEED)
    if TOTAL_POLICY_SPECS != 48:
        raise RuntimeError(f"direct policy family must contain exactly 48 specs, got {TOTAL_POLICY_SPECS}")
    strict_validate_paths(args.output, args.manifest_output, args.source_manifest)
    source_manifest = load_required_source_manifest(args.source_manifest, model_id=args.model_id, model_revision=args.model_revision)
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    input_csv = args.market_csv or args.input_csv
    cfg = Config(input_csv=input_csv, output=args.output, funding_csv=args.funding_csv, premium_csv=args.premium_csv, exclude_from=args.exclude_from)
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    positions, _, _ = anchor_dataset(market, pd.DataFrame(index=market.index))
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    masks = {name: split_mask_for_anchors(dates, positions, *bounds) for name, bounds in WINDOWS.items()}
    cutoff_2024_pos = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64("2024-01-01"), side="left"))
    phase1_targets, target_meta = executable_path_targets_48h(market, positions, available_before_position=cutoff_2024_pos)
    data_hashes = {"market": _file_sha256(input_csv), "funding": optional_file_sha256(args.funding_csv), "premium": optional_file_sha256(args.premium_csv)}
    assert_data_hashes_match_source(data_hashes, source_manifest)

    # Phase 1: no 2024+ embeddings, targets, retrieval, policy hashes, or metrics.
    rep, valid_mask, pca_metadata, embedding_metadata, model_metadata = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=True)
    fit_mask = fit_admissible_mask(dates, positions, phase1_targets, valid_mask, sequence_length=1)
    start_index = first_post_fit_index(dates, positions, fit_mask)
    stop_index = int(np.searchsorted(signal_dates.to_numpy(), np.datetime64(pd.Timestamp("2024-01-01")), side="left"))
    scales = fit_component_scales(phase1_targets, fit_mask)
    phase1_scores, phase1_diagnostics = _score_streams_from_retrieval(rep, valid_mask, phase1_targets, dates, positions, fit_mask, cfg, start_index=start_index, stop_index=stop_index)
    for stream_id, scores in phase1_scores.items():
        phase1_diagnostics[stream_id]["score_quality_prefreeze"] = _score_quality(scores, phase1_targets[:, 0], {k: masks[k] for k in ("fit2020_2022", "holdout2023")})
    raw_candidates = build_direct_candidates(phase1_scores, market=market, dates=dates, positions=positions, cfg=cfg)
    selected, _phase1_signals = select_distinct_top10(raw_candidates)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_required_provenance": SOURCE_MANIFEST_NAME_PREFIX,
        "source_manifest_sha256": _file_sha256(args.source_manifest),
        "model": model_metadata,
        "algorithm": "strict delayed dense-retrieval path utility; no gradient training; cosine kNN empirical quantiles; direct signed-utility execution",
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), "anchor_stride_hours": 6, **embedding_metadata},
        "representation": {**pca_metadata, "retrieval_representations": list(REPRESENTATIONS)},
        "target": {**target_meta, "retrieval_k_values": list(K_VALUES)},
        "strict_protocol": {
            "phase1": "extract only signal dates <2024; generate path labels only when exit<2024; run retrieval only through signals<2024; fit/in-sample score prefix NaN",
            "memory": "initial memory only fit rows whose 48h path exits before 2023; after fit add/query only rows with ready_position=signal+1+576 <= current signal position",
            "selection": "2023 metrics only; positive return and >=8 trades; actual executable path hash de-dupe; no rolling percentile/quantile gates",
            "phase2": "after frozen manifest, extract full PCA only for selected retrieval specs, generate full path targets, rerun causal retrieval from scratch, assert frozen 2023 executable hashes unchanged, then compute OOS metrics/diagnostics/cost stress",
        },
        "policy": {"rule": DIRECT_POLICY_DESCRIPTION, "side_policies": list(SIDES), "rolling_percentile_gates": False, "score_quantiles": [], "score_windows": []},
        "utility": {"specs": [name for name, _mode, _lam in UTILITY_SPECS], "cost": {"round_trip_unlevered": round_trip_unlevered_cost(cfg), "formula": "2*(cfg.fee_rate+cfg.slippage_rate)", "expected_default": 0.0012}},
        "phase1_diagnostics": phase1_diagnostics,
        "top10": selected,
        "trial_counts": {"retrieval_representations": len(REPRESENTATIONS), "k_values": len(K_VALUES), "utility_scores": len(UTILITY_SPECS), "side_policies": len(SIDES), "total_policy_specs": TOTAL_POLICY_SPECS, "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    phase2_diagnostics: dict[str, Any] = {}
    selected_specs = {(str(r["retrieval_representation"]), int(r["k"])) for r in selected}
    if selected_specs:
        targets_full, _ = executable_path_targets_48h(market, positions)
        rep_full, valid_full, pca_full_metadata, _, _ = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=False)
        assert_pca_hashes_match_source(pca_full_metadata, source_manifest, dimensions=PCA_DIMS)
        fit_mask_full = fit_admissible_mask(dates, positions, targets_full, valid_full, sequence_length=1)
        start_index_full = first_post_fit_index(dates, positions, fit_mask_full)
        full_scales = fit_component_scales(targets_full, fit_mask_full)
        if not np.allclose(full_scales, scales, rtol=0.0, atol=1e-12):
            raise RuntimeError("fit-only path target scales changed after future target materialization")
        phase2_scores, phase2_diagnostics = _score_streams_from_retrieval(rep_full, valid_full, targets_full, dates, positions, fit_mask_full, cfg, start_index=start_index_full, stop_index=None, selected_specs=selected_specs)
        for stream_id, scores in phase2_scores.items():
            phase2_diagnostics[stream_id]["score_quality"] = _score_quality(scores, targets_full[:, 0], masks)
    else:
        targets_full = phase1_targets
        phase2_scores = phase1_scores

    for rank, row in enumerate(selected, start=1):
        scores = phase2_scores[row["stream_id"]]
        long_active, short_active = direct_policy_masks(scores, positions, len(market), row["side_policy"])
        rerun_hash = selection_window_signal_hash(long_active, short_active, market=market, dates=dates)
        if rerun_hash != row["signal_hash"]:
            raise RuntimeError(f"selected 2023 executable hash changed for {row['stream_id']}: {rerun_hash} != {row['signal_hash']}")
        selected_signals[row["signal_hash"]] = (long_active, short_active)
        row["pre_evaluation_rank"] = rank
        row["phase2_2023_hash_verified"] = True
        for split in ("test2024", "eval2025", "ytd2026"):
            row[split] = sim(market, dates, long_active, short_active, cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, split)
        row["passes_alpha_pool"] = bool(row["test2024"]["ratio"] >= 3.0 and row["eval2025"]["ratio"] >= 3.0 and row["test2024"]["trades"] >= 8 and row["eval2025"]["trades"] >= 8 and row["test2024"]["return_pct"] > 0.0 and row["eval2025"]["return_pct"] > 0.0)
        row["passes_live_grade"] = bool(row["passes_alpha_pool"] and row["ytd2026"]["ratio"] >= 5.0 and row["ytd2026"]["trades"] >= 6 and row["ytd2026"]["return_pct"] > 0.0)

    alpha_pool, live_grade = top10_promotions(selected)
    cost_stress: dict[str, Any] = {}
    for row in selected:
        long_active, short_active = selected_signals[row["signal_hash"]]
        cost_stress[row["signal_hash"]] = {}
        for bps in (6, 8, 10, 15):
            stressed_cfg = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
            cost_stress[row["signal_hash"]][str(bps)] = {split: sim(market, dates, long_active, short_active, stressed_cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, split) for split in ("test2024", "eval2025", "ytd2026")}

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "strict delayed dense-retrieval MOMENT PCA32 direct signed-utility execution; frozen PCA32; cosine kNN; no score quantiles/windows; next-bar entry; hold576; 0.5x; 6bp/side; full-window CAGR; strict intratrade MDD", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "model": model_metadata, "input": manifest["input"], "representation": manifest["representation"], "phase1_diagnostics": phase1_diagnostics, "phase2_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--context-hours", type=int, default=CONTEXT_HOURS)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--market-csv", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--funding-csv", default="")
    parser.add_argument("--premium-csv", default="")
    parser.add_argument("--exclude-from", default="2026-06-02")
    args = parser.parse_args()
    if not args.input_csv and not args.market_csv:
        parser.error("one of --input-csv or --market-csv is required")
    output = run(args)
    print(json.dumps({"tested_candidates": output["tested_candidates"], "selected": len(output["selected"]), "alpha_pool": len(output["alpha_pool_qualifiers"]), "live_grade": len(output["live_grade"])}, indent=2))


if __name__ == "__main__":
    main()
