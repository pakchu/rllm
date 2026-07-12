"""Strict bounded-recency MOMENT PCA32 retrieval alpha evaluation.

Inputs are two frozen manifests: the MOMENT embedding source manifest and a
frozen direct-utility retrieval-family manifest.  This experiment derives its
base (representation, k, utility_score) family from that family manifest,
dedupes it, then reruns the same direct signed-utility policy with causal
retrieval memory optionally bounded to 365d or 730d of 5m bars.
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
from training.evaluate_moment_retrieval_direct_utility_alpha import DIRECT_POLICY_DESCRIPTION, direct_policy_masks
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
    SOURCE_MANIFEST_NAME_PREFIX,
    TARGET_COMPONENTS,
    RetrievalSpec,
    build_query_representations,
    compact_retrieval_diagnostics,
    denormalize_sort_and_clamp_quantiles,
    empirical_neighbor_quantiles,
    executable_path_targets_48h,
    fit_admissible_mask,
    fit_component_scales,
    fit_standardizer,
    load_required_source_manifest,
    ready_positions_for_targets,
    round_trip_unlevered_cost,
    select_distinct_top10,
    signed_utility_scores,
    standardize_l2,
)
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash
from training.search_tabicl_foundation_alpha import ANCHOR_STRIDE, HOLD_BARS, WINDOWS, _file_sha256, _git_head, anchor_dataset, split_mask_for_anchors, top10_promotions
from training.search_moment_mamba2_path_critic_alpha import QUANTILES, UTILITY_SPECS

RECENCY_WINDOWS: dict[str, int | None] = {"365d": 105120, "730d": 210240, "all": None}
RECENCY_SIDE_POLICIES: tuple[Literal["long", "both"], ...] = ("long", "both")
FAMILY_SOURCE_NAME_PREFIX = "moment_retrieval_direct_utility_top10_manifest"
READY_OFFSET_BARS = 1 + HOLD_BARS


def load_required_family_source_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if FAMILY_SOURCE_NAME_PREFIX not in p.name:
        raise ValueError(f"family source manifest name must include {FAMILY_SOURCE_NAME_PREFIX!r}: {p.name}")
    manifest = json.loads(p.read_text())
    if manifest.get("later_metrics_included") is not False:
        raise ValueError("family source manifest must have later_metrics_included=false")
    top = manifest.get("top10")
    if not isinstance(top, list) or not top:
        raise ValueError("family source manifest must contain a non-empty top10 list")
    return manifest


def derived_base_retrieval_specs(family_source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive unique retrieval specs only from frozen family-source Top10/Top6 rows."""
    allowed_utilities = {name for name, _mode, _lam in UTILITY_SPECS}
    seen: set[tuple[str, int, str]] = set()
    out: list[dict[str, Any]] = []
    for row in family_source_manifest.get("top10", []):
        try:
            representation = str(row["retrieval_representation"])
            k = int(row["k"])
            utility_score = str(row["utility_score"])
        except KeyError as exc:
            raise ValueError(f"family source top10 row missing {exc.args[0]}") from exc
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unsupported retrieval representation in family source: {representation}")
        if k not in K_VALUES:
            raise ValueError(f"unsupported k in family source: {k}")
        if utility_score not in allowed_utilities:
            raise ValueError(f"unsupported utility score in family source: {utility_score}")
        RetrievalSpec(representation, k)  # type: ignore[arg-type]
        key = (representation, k, utility_score)
        if key in seen:
            continue
        seen.add(key)
        out.append({"retrieval_representation": representation, "k": k, "utility_score": utility_score, "stream_id": f"retrieval_pathcritic_{representation}_k{k}_{utility_score}"})
    if not out:
        raise ValueError("family source produced no retrieval specs")
    return out


def verify_family_source_provenance(
    family_source_manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    data_hashes: dict[str, str | None],
) -> None:
    if family_source_manifest.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("family source does not reference the supplied MOMENT source hash")
    assert_data_hashes_match_source(data_hashes, family_source_manifest)


def recency_policy_family(base_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family: list[dict[str, Any]] = []
    for base in base_specs:
        for window_name, max_age in RECENCY_WINDOWS.items():
            for side_policy in RECENCY_SIDE_POLICIES:
                family.append({
                    **base,
                    "memory_window": window_name,
                    "max_age_bars": max_age,
                    "side_policy": side_policy,
                    "policy_rule": DIRECT_POLICY_DESCRIPTION,
                })
    return family


def delayed_causal_recency_retrieval_quantiles(
    normalized_reps: np.ndarray,
    rep_valid_mask: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    *,
    start_index: int,
    stop_index: int | None = None,
    k: int,
    max_age_bars: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Causal cosine kNN with delayed target availability and optional age bound."""
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
    ready = ready_positions_for_targets(pos)
    pool_sizes = np.zeros(n, dtype=np.int32)
    kth_sims = np.full(n, np.nan, dtype=np.float32)
    min_ages = np.full(n, np.nan, dtype=np.float32)
    max_ages = np.full(n, np.nan, dtype=np.float32)
    for i in range(max(0, int(start_index)), stop):
        if not valid[i]:
            continue
        age = pos[i] - pos
        memory = usable & (ready <= pos[i]) & (age >= READY_OFFSET_BARS)
        if max_age_bars is not None:
            memory &= age <= int(max_age_bars)
        memory[i] = False
        memory_indices = np.flatnonzero(memory)
        pool_sizes[i] = int(len(memory_indices))
        if len(memory_indices):
            mem_ages = age[memory_indices]
            min_ages[i] = float(mem_ages.min())
            max_ages[i] = float(mem_ages.max())
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
        "memory_window_max_age_bars": None if max_age_bars is None else int(max_age_bars),
        "ready_offset_bars": READY_OFFSET_BARS,
        "predicted_rows": int(predicted.sum()),
        "pool_size_by_anchor": pool_sizes.tolist(),
        "average_pool_size_predicted": float(pool_sizes[predicted].mean()) if predicted.any() else 0.0,
        "average_kth_similarity_predicted": float(np.nanmean(kth_sims[predicted])) if predicted.any() else None,
        "average_memory_min_age_bars_predicted": float(np.nanmean(min_ages[predicted])) if predicted.any() else None,
        "average_memory_max_age_bars_predicted": float(np.nanmean(max_ages[predicted])) if predicted.any() else None,
        "kth_similarity_by_anchor": [None if not np.isfinite(v) else float(v) for v in kth_sims],
        "target_availability": "row j may enter memory only when positions[j]+1+576 <= current signal position and current-position[j] <= max_age_bars when bounded",
    }
    return preds, diagnostics


def recency_retrieval_path_quantiles(
    pca32: np.ndarray,
    valid_anchor_mask: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    fit_mask: np.ndarray,
    spec: RetrievalSpec,
    *,
    start_index: int,
    stop_index: int | None = None,
    max_age_bars: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    reps, rep_valid, rep_meta = build_query_representations(pca32, valid_anchor_mask, kind=spec.representation)
    fit_for_rep = np.asarray(fit_mask, dtype=bool) & rep_valid & np.isfinite(np.asarray(targets, dtype=float)).all(axis=1)
    mean, std = fit_standardizer(reps, fit_for_rep)
    norm, norm_valid = standardize_l2(reps, mean, std, rep_valid)
    preds, diag = delayed_causal_recency_retrieval_quantiles(
        norm,
        norm_valid,
        targets,
        positions,
        start_index=start_index,
        stop_index=stop_index,
        k=spec.k,
        max_age_bars=max_age_bars,
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


def build_recency_candidates(
    phase_scores: dict[str, np.ndarray],
    policies: list[dict[str, Any]],
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    positions: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    for policy in policies:
        scores = phase_scores[policy["recency_stream_id"]]
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


def _score_streams_from_recency_retrieval(
    rep: np.ndarray,
    valid_mask: np.ndarray,
    targets: np.ndarray,
    positions: np.ndarray,
    fit_mask: np.ndarray,
    cfg: Config,
    *,
    start_index: int,
    stop_index: int | None,
    base_specs: list[dict[str, Any]],
    selected_recency_specs: set[tuple[str, int, str]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores_by_stream: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    utility_names = {s["utility_score"] for s in base_specs}
    unique_retrieval = sorted({(str(s["retrieval_representation"]), int(s["k"]), str(w), max_age) for s in base_specs for w, max_age in RECENCY_WINDOWS.items()})
    if selected_recency_specs is not None:
        unique_retrieval = [r for r in unique_retrieval if (r[0], r[1], r[2]) in selected_recency_specs]
    for kind, k, window_name, max_age in unique_retrieval:
        spec = RetrievalSpec(kind, k)  # type: ignore[arg-type]
        print(f"retrieving {spec.stream_prefix}_recency_{window_name}", file=sys.stderr, flush=True)
        raw_q, diag = recency_retrieval_path_quantiles(rep, valid_mask, targets, positions, fit_mask, spec, start_index=start_index, stop_index=stop_index, max_age_bars=max_age)
        raw_q[:start_index] = np.nan
        q_pred = denormalize_sort_and_clamp_quantiles(raw_q, np.ones(len(TARGET_COMPONENTS), dtype=np.float32))
        streams, utility_meta = signed_utility_scores(q_pred, cfg)
        for utility_name, scores in streams.items():
            if utility_name not in utility_names:
                continue
            stream_id = f"{spec.stream_prefix}_{utility_name}"
            recency_stream_id = f"{stream_id}_recency_{window_name}"
            scores[:start_index] = np.nan
            scores_by_stream[recency_stream_id] = scores
            diagnostics[recency_stream_id] = {
                "retrieval": {**compact_retrieval_diagnostics(diag), "memory_window": window_name},
                "utility": utility_name,
                "utility_cost": utility_meta,
                "first_policy_score_index": start_index,
                "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all()),
            }
    return scores_by_stream, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(SEED)
    strict_validate_paths(args.output, args.manifest_output, args.source_manifest)
    source_manifest = load_required_source_manifest(args.source_manifest, model_id=args.model_id, model_revision=args.model_revision)
    family_manifest = load_required_family_source_manifest(args.family_source_manifest)
    if source_manifest.get("later_metrics_included") is not False:
        raise ValueError("source manifest must have later_metrics_included=false")
    base_specs = derived_base_retrieval_specs(family_manifest)
    policies = recency_policy_family(base_specs)
    for p in policies:
        p["recency_stream_id"] = f"{p['stream_id']}_recency_{p['memory_window']}"
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
    source_manifest_sha256 = _file_sha256(args.source_manifest)
    verify_family_source_provenance(
        family_manifest,
        source_manifest_sha256=source_manifest_sha256,
        data_hashes=data_hashes,
    )

    rep, valid_mask, pca_metadata, embedding_metadata, model_metadata = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=True)
    fit_mask = fit_admissible_mask(dates, positions, phase1_targets, valid_mask, sequence_length=1)
    start_index = first_post_fit_index(dates, positions, fit_mask)
    stop_index = int(np.searchsorted(signal_dates.to_numpy(), np.datetime64(pd.Timestamp("2024-01-01")), side="left"))
    scales = fit_component_scales(phase1_targets, fit_mask)
    phase1_scores, phase1_diagnostics = _score_streams_from_recency_retrieval(rep, valid_mask, phase1_targets, positions, fit_mask, cfg, start_index=start_index, stop_index=stop_index, base_specs=base_specs)
    for stream_id, scores in phase1_scores.items():
        phase1_diagnostics[stream_id]["score_quality_prefreeze"] = _score_quality(scores, phase1_targets[:, 0], {k: masks[k] for k in ("fit2020_2022", "holdout2023")})
    raw_candidates = build_recency_candidates(phase1_scores, policies, market=market, dates=dates, positions=positions, cfg=cfg)
    selected, _phase1_signals = select_distinct_top10(raw_candidates)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_required_provenance": SOURCE_MANIFEST_NAME_PREFIX,
        "source_manifest_sha256": source_manifest_sha256,
        "family_source_manifest": str(Path(args.family_source_manifest).resolve()),
        "family_source_manifest_required_provenance": FAMILY_SOURCE_NAME_PREFIX,
        "family_source_manifest_sha256": _file_sha256(args.family_source_manifest),
        "family_source_top_rows": len(family_manifest.get("top10", [])),
        "model": model_metadata,
        "algorithm": "strict delayed dense-retrieval path utility with causal recency memory; no gradient training; cosine kNN empirical quantiles; direct signed-utility execution",
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), "anchor_stride_hours": 6, **embedding_metadata},
        "representation": {**pca_metadata, "base_retrieval_specs_from_family_source": base_specs},
        "target": {**target_meta},
        "strict_protocol": {
            "phase1": "extract only signal dates <2024; generate path labels only when exit<2024; run retrieval only through signals<2024; fit/in-sample score prefix NaN",
            "memory": "row j may enter memory only when positions[j]+1+576 <= current signal position and age_bars=current-position[j] is <= the recency max age; all means no age cap",
            "selection": "2023 metrics only; positive return and >=8 trades; actual executable path hash de-dupe; no rolling percentile/quantile gates",
            "phase2": "after frozen manifest, extract full PCA only for selected recency specs, generate full path targets, rerun causal retrieval from scratch, assert frozen 2023 executable hashes unchanged, then compute OOS metrics/diagnostics/cost stress",
        },
        "policy": {"rule": DIRECT_POLICY_DESCRIPTION, "side_policies": list(RECENCY_SIDE_POLICIES), "rolling_percentile_gates": False, "score_quantiles": [], "score_windows": [], "recency_windows": RECENCY_WINDOWS},
        "utility": {"specs": [name for name, _mode, _lam in UTILITY_SPECS], "cost": {"round_trip_unlevered": round_trip_unlevered_cost(cfg), "formula": "2*(cfg.fee_rate+cfg.slippage_rate)", "expected_default": 0.0012}},
        "phase1_diagnostics": phase1_diagnostics,
        "top10": selected,
        "trial_counts": {"base_specs_from_family_source": len(base_specs), "recency_windows": len(RECENCY_WINDOWS), "side_policies": len(RECENCY_SIDE_POLICIES), "total_policy_specs": len(policies), "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    phase2_diagnostics: dict[str, Any] = {}
    selected_recency_specs = {(str(r["retrieval_representation"]), int(r["k"]), str(r["memory_window"])) for r in selected}
    if selected_recency_specs:
        targets_full, _ = executable_path_targets_48h(market, positions)
        rep_full, valid_full, pca_full_metadata, _, _ = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=False)
        assert_pca_hashes_match_source(pca_full_metadata, source_manifest, dimensions=PCA_DIMS)
        fit_mask_full = fit_admissible_mask(dates, positions, targets_full, valid_full, sequence_length=1)
        start_index_full = first_post_fit_index(dates, positions, fit_mask_full)
        full_scales = fit_component_scales(targets_full, fit_mask_full)
        if not np.allclose(full_scales, scales, rtol=0.0, atol=1e-12):
            raise RuntimeError("fit-only path target scales changed after future target materialization")
        phase2_scores, phase2_diagnostics = _score_streams_from_recency_retrieval(rep_full, valid_full, targets_full, positions, fit_mask_full, cfg, start_index=start_index_full, stop_index=None, base_specs=base_specs, selected_recency_specs=selected_recency_specs)
        for stream_id, scores in phase2_scores.items():
            phase2_diagnostics[stream_id]["score_quality"] = _score_quality(scores, targets_full[:, 0], masks)
    else:
        targets_full = phase1_targets
        phase2_scores = phase1_scores

    for rank, row in enumerate(selected, start=1):
        scores = phase2_scores[row["recency_stream_id"]]
        long_active, short_active = direct_policy_masks(scores, positions, len(market), row["side_policy"])
        rerun_hash = selection_window_signal_hash(long_active, short_active, market=market, dates=dates)
        if rerun_hash != row["signal_hash"]:
            raise RuntimeError(f"selected 2023 executable hash changed for {row['recency_stream_id']}: {rerun_hash} != {row['signal_hash']}")
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

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "strict delayed dense-retrieval MOMENT PCA32 direct signed-utility execution with recency-bounded causal memory; frozen PCA32; cosine kNN; no score quantiles/windows; next-bar entry; hold576; 0.5x; 6bp/side; full-window CAGR; strict intratrade MDD", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "family_source_manifest": str(Path(args.family_source_manifest).resolve()), "model": model_metadata, "input": manifest["input"], "representation": manifest["representation"], "phase1_diagnostics": phase1_diagnostics, "phase2_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--family-source-manifest", required=True)
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
