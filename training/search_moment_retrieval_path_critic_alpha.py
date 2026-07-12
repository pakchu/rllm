"""Strict delayed dense-retrieval MOMENT PCA32 path critic.

This is a non-parametric causal analogue memory over frozen MOMENT PCA32 anchor
states.  It deliberately avoids gradient training: every prediction is a cosine
nearest-neighbor empirical path distribution built only from targets whose
48-hour executable path has matured before the current signal position.

Protocol is two-phase.  Phase 1 extracts/features/targets only pre-2024 signal
rows and freezes the 2023 Top-10 manifest.  Phase 2 reruns retrieval from
scratch on full features/targets only for selected specs, verifies the frozen
2023 executable path hashes, then computes 2024+ metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
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
from training.search_chronos2_zero_shot_alpha import ROLLING_WINDOWS, SCORE_QUANTILES
from training.search_invariant_groupdro_alpha import _score_quality
from training.search_moment_continual_probe_alpha import (
    assert_data_hashes_match_source,
    assert_pca_hashes_match_source,
    strict_validate_paths,
)
from training.search_moment_embedding_probe_alpha import CONTEXT_HOURS, EMBEDDING_VARIATES, MODEL_ID, MODEL_REVISION
from training.search_moment_mamba2_alpha import SEED, extract_pca32_for_phase, first_post_fit_index, seed_everything
from training.search_moment_mamba2_path_critic_alpha import (
    MIN_SCORE_HISTORY,
    PCA_DIMS,
    QUANTILES,
    SOURCE_MANIFEST_NAME_PREFIX,
    TARGET_COMPONENTS,
    UTILITY_SPECS,
    denormalize_sort_and_clamp_quantiles,
    executable_path_targets_48h,
    fit_admissible_mask,
    fit_component_scales,
    load_required_source_manifest,
    policy_masks,
    round_trip_unlevered_cost,
    select_distinct_top10,
    signed_utility_scores,
)
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash
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

K_VALUES = (64, 128)
REPRESENTATIONS = ("current", "current_mean8")
SIDES = ("long", "short", "both")
MEAN8_WINDOW = 8
READY_OFFSET_BARS = 1 + HOLD_BARS
PER_ANCHOR_DIAGNOSTIC_KEYS = {
    "pool_size_by_anchor",
    "kth_similarity_by_anchor",
    "ready_offset_by_anchor",
}


@dataclass(frozen=True)
class RetrievalSpec:
    representation: Literal["current", "current_mean8"]
    k: Literal[64, 128]

    @property
    def stream_prefix(self) -> str:
        return f"retrieval_pathcritic_{self.representation}_k{self.k}"


def build_query_representations(
    pca32: np.ndarray,
    valid_anchor_mask: np.ndarray,
    *,
    kind: str,
    mean_window: int = MEAN8_WINDOW,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build current or current+causal-mean8 representations.

    ``current_mean8`` concatenates the current PCA32 vector with the mean of the
    current and previous seven anchor PCA32 states.  All contributing states must
    be valid, so mutating future states cannot affect any prefix row.
    """
    x = np.asarray(pca32, dtype=np.float32)
    valid = np.asarray(valid_anchor_mask, dtype=bool)
    if x.ndim != 2 or x.shape[1] != 32:
        raise ValueError("expected PCA32 array with shape (n, 32)")
    if len(valid) != len(x):
        raise ValueError("valid_anchor_mask length mismatch")
    if kind == "current":
        reps = x.copy()
        reps[~valid] = np.nan
        return reps.astype(np.float32), valid.copy(), {"kind": kind, "dimensions": 32, "valid_rule": "current anchor PCA32 valid"}
    if kind != "current_mean8":
        raise ValueError(f"unknown retrieval representation: {kind}")
    w = int(mean_window)
    reps = np.full((len(x), 64), np.nan, dtype=np.float32)
    out_valid = np.zeros(len(x), dtype=bool)
    for i in range(w - 1, len(x)):
        sl = slice(i - w + 1, i + 1)
        if valid[sl].all() and np.isfinite(x[sl]).all():
            reps[i, :32] = x[i]
            reps[i, 32:] = x[sl].mean(axis=0)
            out_valid[i] = True
    return reps, out_valid, {
        "kind": kind,
        "dimensions": 64,
        "mean_window_anchors": w,
        "mean_rule": "causal mean over current and previous 7 anchor PCA32 states",
        "valid_rule": "current plus all 8 contributing anchor PCA32 states valid",
    }


def fit_standardizer(reps: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit = np.asarray(reps, dtype=np.float32)[np.asarray(fit_mask, dtype=bool)]
    if fit.size == 0 or not np.isfinite(fit).all():
        raise ValueError("fit representations contain no finite complete rows")
    mean = fit.mean(axis=0).astype(np.float32)
    std = fit.std(axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return mean, std


def standardize_l2(reps: np.ndarray, mean: np.ndarray, std: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (np.asarray(reps, dtype=np.float32) - np.asarray(mean, dtype=np.float32).reshape(1, -1)) / np.asarray(std, dtype=np.float32).reshape(1, -1)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(z).all(axis=1)
    out = np.zeros_like(z, dtype=np.float32)
    norms = np.linalg.norm(z[valid], axis=1)
    keep_local = norms > 1e-12
    valid_indices = np.flatnonzero(valid)
    keep_indices = valid_indices[keep_local]
    out[keep_indices] = z[keep_indices] / norms[keep_local, None]
    final_valid = np.zeros(len(z), dtype=bool)
    final_valid[keep_indices] = True
    return out, final_valid


def empirical_neighbor_quantiles(neighbor_targets: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> np.ndarray:
    y = np.asarray(neighbor_targets, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != len(TARGET_COMPONENTS):
        raise ValueError("neighbor_targets must have shape (n, 3)")
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("neighbor targets must be non-empty and finite")
    return np.quantile(y, np.asarray(quantiles, dtype=float), axis=0).T.astype(np.float32)


def ready_positions_for_targets(positions: np.ndarray, *, hold_bars: int = HOLD_BARS) -> np.ndarray:
    return np.asarray(positions, dtype=np.int64) + 1 + int(hold_bars)


def initial_fit_memory_mask(
    dates: pd.Series,
    positions: np.ndarray,
    fit_mask: np.ndarray,
    targets: np.ndarray,
    rep_valid_mask: np.ndarray,
    *,
    fit_end: str = "2023-01-01",
) -> np.ndarray:
    """Rows allowed in the initial memory before post-fit causal growth."""
    dates = pd.to_datetime(dates)
    ready = ready_positions_for_targets(positions)
    in_bounds = ready < len(dates)
    before_fit_end = np.zeros(len(ready), dtype=bool)
    if in_bounds.any():
        before_fit_end[in_bounds] = dates.iloc[ready[in_bounds]].to_numpy() < np.datetime64(pd.Timestamp(fit_end))
    return (
        np.asarray(fit_mask, dtype=bool)
        & np.asarray(rep_valid_mask, dtype=bool)
        & np.isfinite(np.asarray(targets, dtype=float)).all(axis=1)
        & before_fit_end
    )


def delayed_causal_retrieval_quantiles(
    normalized_reps: np.ndarray,
    rep_valid_mask: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    *,
    fit_memory_mask: np.ndarray,
    start_index: int,
    stop_index: int | None = None,
    k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Causal cosine kNN with delayed target availability.

    Before each row ``i`` is predicted, memory is exactly rows ``j`` with finite
    target and representation where either ``fit_memory_mask[j]`` is true or the
    executable path target is mature: ``positions[j]+1+576 <= positions[i]``.
    Row ``i`` itself is therefore never read unless its target had matured before
    its own signal, which cannot happen for positive holds.
    """
    x = np.asarray(normalized_reps, dtype=np.float32)
    valid = np.asarray(rep_valid_mask, dtype=bool)
    y = np.asarray(targets, dtype=np.float32)
    pos = np.asarray(positions, dtype=np.int64)
    n = len(pos)
    if x.shape[0] != n or y.shape[0] != n or valid.shape[0] != n:
        raise ValueError("retrieval input length mismatch")
    stop = n if stop_index is None else min(int(stop_index), n)
    k = int(k)
    preds = np.full((n, len(TARGET_COMPONENTS), len(QUANTILES)), np.nan, dtype=np.float32)
    finite_target = np.isfinite(y).all(axis=1)
    usable = valid & finite_target
    fit_memory = np.asarray(fit_memory_mask, dtype=bool) & usable
    ready = ready_positions_for_targets(pos)
    pool_sizes = np.zeros(n, dtype=np.int32)
    kth_sims = np.full(n, np.nan, dtype=np.float32)
    ready_offsets = np.full(n, READY_OFFSET_BARS, dtype=np.int32)
    for i in range(max(0, int(start_index)), stop):
        if not valid[i]:
            continue
        mature = usable & (ready <= pos[i])
        # Keep this explicit even though ready[i] > pos[i] under HOLD_BARS > 0.
        mature[i] = False
        memory = fit_memory | mature
        memory_indices = np.flatnonzero(memory)
        pool_sizes[i] = int(len(memory_indices))
        if len(memory_indices) < k:
            continue
        sims = x[memory_indices] @ x[i]
        if len(sims) > k:
            part = np.argpartition(sims, -k)[-k:]
            chosen_local = part[np.argsort(sims[part])[::-1]]
        else:
            chosen_local = np.argsort(sims)[::-1]
        chosen = memory_indices[chosen_local[:k]]
        kth_sims[i] = float(sims[chosen_local[min(k, len(chosen_local)) - 1]])
        preds[i] = empirical_neighbor_quantiles(y[chosen], QUANTILES)
    predicted = np.isfinite(preds).all(axis=(1, 2))
    diagnostics = {
        "k": k,
        "ready_offset_bars": READY_OFFSET_BARS,
        "initial_fit_memory_rows": int(fit_memory.sum()),
        "predicted_rows": int(predicted.sum()),
        "pool_size_by_anchor": pool_sizes.tolist(),
        "average_pool_size_predicted": float(pool_sizes[predicted].mean()) if predicted.any() else 0.0,
        "average_kth_similarity_predicted": float(np.nanmean(kth_sims[predicted])) if predicted.any() else None,
        "kth_similarity_by_anchor": [None if not np.isfinite(v) else float(v) for v in kth_sims],
        "ready_offset_by_anchor": ready_offsets.tolist(),
        "target_availability": "row j may enter memory only when positions[j]+1+576 <= current signal position; current row target is never read before prediction",
    }
    return preds, diagnostics


def compact_retrieval_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Drop repeated per-anchor arrays while retaining aggregate evidence."""
    return {
        key: value
        for key, value in diagnostics.items()
        if key not in PER_ANCHOR_DIAGNOSTIC_KEYS
    }


def retrieval_path_quantiles(
    pca32: np.ndarray,
    valid_anchor_mask: np.ndarray,
    targets: np.ndarray,
    dates: pd.Series,
    positions: np.ndarray,
    fit_mask: np.ndarray,
    spec: RetrievalSpec,
    *,
    start_index: int,
    stop_index: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reps, rep_valid, rep_meta = build_query_representations(pca32, valid_anchor_mask, kind=spec.representation)
    fit_for_rep = np.asarray(fit_mask, dtype=bool) & rep_valid & np.isfinite(np.asarray(targets, dtype=float)).all(axis=1)
    mean, std = fit_standardizer(reps, fit_for_rep)
    norm, norm_valid = standardize_l2(reps, mean, std, rep_valid)
    memory_mask = initial_fit_memory_mask(dates, positions, fit_for_rep, targets, norm_valid)
    preds, diag = delayed_causal_retrieval_quantiles(
        norm,
        norm_valid,
        targets,
        positions,
        fit_memory_mask=memory_mask,
        start_index=start_index,
        stop_index=stop_index,
        k=spec.k,
    )
    diag.update({
        "representation": rep_meta,
        "fit_standardization": "per-dimension mean/std fit on admissible 2020-2022 rows only, then L2 normalize",
        "standardizer_mean": mean.astype(float).tolist(),
        "standardizer_std": std.astype(float).tolist(),
        "fit_rows_for_standardizer": int(fit_for_rep.sum()),
    })
    return preds, diag


def selection_window_signal_hash(long_active: np.ndarray, short_active: np.ndarray, *, market: pd.DataFrame, dates: pd.Series) -> str:
    return effective_selection_signal_hash(market, dates, long_active, short_active, window=WINDOWS["holdout2023"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(SEED)
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

    rep, valid_mask, pca_metadata, embedding_metadata, model_metadata = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=True)
    fit_mask = fit_admissible_mask(dates, positions, phase1_targets, valid_mask, sequence_length=1)
    start_index = first_post_fit_index(dates, positions, fit_mask)
    stop_index = int(np.searchsorted(signal_dates.to_numpy(), np.datetime64(pd.Timestamp("2024-01-01")), side="left"))
    scales = fit_component_scales(phase1_targets, fit_mask)

    phase1_diagnostics: dict[str, Any] = {}
    phase1_scores: dict[str, np.ndarray] = {}
    raw_candidates: list[dict[str, Any]] = []
    for kind in REPRESENTATIONS:
        for k in K_VALUES:
            spec = RetrievalSpec(kind, k)  # type: ignore[arg-type]
            print(f"retrieving {spec.stream_prefix}", file=sys.stderr, flush=True)
            raw_q, diag = retrieval_path_quantiles(rep, valid_mask, phase1_targets, dates, positions, fit_mask, spec, start_index=start_index, stop_index=stop_index)
            raw_q[:start_index] = np.nan
            # Retrieval quantiles are already de-normalized path units; this call
            # provides the shared sort/MAE clamp helper with unit scales.
            q_pred = denormalize_sort_and_clamp_quantiles(raw_q, np.ones(len(TARGET_COMPONENTS), dtype=np.float32))
            streams, utility_meta = signed_utility_scores(q_pred, cfg)
            for utility_name, scores in streams.items():
                stream_id = f"{spec.stream_prefix}_{utility_name}"
                scores[:start_index] = np.nan
                phase1_scores[stream_id] = scores
                phase1_diagnostics[stream_id] = {
                    "retrieval": compact_retrieval_diagnostics(diag),
                    "utility": utility_name,
                    "utility_cost": utility_meta,
                    "first_policy_score_index": start_index,
                    "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all()),
                }
                for rolling_window in ROLLING_WINDOWS:
                    for quantile in SCORE_QUANTILES:
                        for side_policy in SIDES:
                            long_active, short_active = policy_masks(scores, positions, len(market), int(rolling_window), float(quantile), side_policy)
                            holdout = sim(market, dates, long_active, short_active, cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, "holdout2023")
                            if holdout["trades"] < 8 or holdout["return_pct"] <= 0.0:
                                continue
                            raw_candidates.append({
                                "stream_id": stream_id,
                                "retrieval_representation": kind,
                                "k": int(k),
                                "utility_score": utility_name,
                                "rolling_score_window_anchors": int(rolling_window),
                                "score_quantile": float(quantile),
                                "minimum_score_history": MIN_SCORE_HISTORY,
                                "side_policy": side_policy,
                                "hold_bars": HOLD_BARS,
                                "anchor_stride_bars": ANCHOR_STRIDE,
                                "holdout2023": holdout,
                                "signal_hash": selection_window_signal_hash(long_active, short_active, market=market, dates=dates),
                                "_long": long_active,
                                "_short": short_active,
                            })

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
        "algorithm": "strict delayed dense-retrieval path critic; no gradient training; cosine kNN empirical quantiles",
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), "anchor_stride_hours": 6, **embedding_metadata},
        "representation": {**pca_metadata, "retrieval_representations": list(REPRESENTATIONS)},
        "target": {**target_meta, "quantiles": list(QUANTILES), "k_values": list(K_VALUES)},
        "strict_protocol": {"phase1": "extract only signal dates <2024; generate path labels only when exit<2024; run retrieval only through signals<2024; fit/in-sample score prefix NaN", "memory": "initial memory only fit rows whose 48h path exits before 2023; after fit add/query only rows with ready_position=signal+1+576 <= current signal position", "selection": "2023 metrics only; positive return and >=8 trades; actual executable path hash de-dupe", "phase2": "after frozen manifest, extract full PCA only for selected specs, generate full path targets, rerun causal retrieval from scratch, assert frozen 2023 executable hashes unchanged, then compute OOS metrics/cost stress"},
        "utility": {"specs": [name for name, _mode, _lam in UTILITY_SPECS], "cost": {"round_trip_unlevered": round_trip_unlevered_cost(cfg), "formula": "2*(cfg.fee_rate+cfg.slippage_rate)", "expected_default": 0.0012}},
        "phase1_diagnostics": phase1_diagnostics,
        "top10": selected,
        "trial_counts": {"retrieval_representations": len(REPRESENTATIONS), "k_values": len(K_VALUES), "utility_scores": len(UTILITY_SPECS), "score_streams": len(phase1_scores), "rolling_windows": len(ROLLING_WINDOWS), "score_quantiles": len(SCORE_QUANTILES), "side_policies": len(SIDES), "total_policy_specs": len(phase1_scores) * len(ROLLING_WINDOWS) * len(SCORE_QUANTILES) * len(SIDES), "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    phase2_diagnostics: dict[str, Any] = {}
    phase2_scores: dict[str, np.ndarray] = {}
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
    else:
        targets_full, rep_full, valid_full, start_index_full = phase1_targets, rep, valid_mask, start_index

    for kind, k in sorted(selected_specs):
        spec = RetrievalSpec(kind, k)  # type: ignore[arg-type]
        raw_q, diag = retrieval_path_quantiles(rep_full, valid_full, targets_full, dates, positions, fit_mask_full, spec, start_index=start_index_full, stop_index=None)
        raw_q[:start_index_full] = np.nan
        q_pred = denormalize_sort_and_clamp_quantiles(raw_q, np.ones(len(TARGET_COMPONENTS), dtype=np.float32))
        streams, utility_meta = signed_utility_scores(q_pred, cfg)
        for utility_name, scores in streams.items():
            stream_id = f"{spec.stream_prefix}_{utility_name}"
            scores[:start_index_full] = np.nan
            phase2_scores[stream_id] = scores
            phase2_diagnostics[stream_id] = {
                "retrieval": compact_retrieval_diagnostics(diag),
                "score_quality": _score_quality(scores, targets_full[:, 0], masks),
                "utility_cost": utility_meta,
                "first_policy_score_index": start_index_full,
                "policy_scores_fit_prefix_nan": bool(
                    np.isnan(scores[:start_index_full]).all()
                ),
            }

    for rank, row in enumerate(selected, start=1):
        scores = phase2_scores[row["stream_id"]]
        long_active, short_active = policy_masks(scores, positions, len(market), int(row["rolling_score_window_anchors"]), float(row["score_quantile"]), row["side_policy"])
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
    for row in live_grade:
        long_active, short_active = selected_signals[row["signal_hash"]]
        cost_stress[row["signal_hash"]] = {}
        for bps in (6, 8, 10, 15):
            stressed_cfg = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
            cost_stress[row["signal_hash"]][str(bps)] = {split: sim(market, dates, long_active, short_active, stressed_cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, split) for split in ("test2024", "eval2025", "ytd2026")}

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "strict delayed dense-retrieval MOMENT PCA32 path critic; frozen PCA32; cosine kNN; next-bar entry; hold576; 0.5x; 6bp/side; full-window CAGR; strict intratrade MDD", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "model": model_metadata, "input": manifest["input"], "representation": manifest["representation"], "phase1_diagnostics": phase1_diagnostics, "phase2_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
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
