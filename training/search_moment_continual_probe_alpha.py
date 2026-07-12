"""Frozen MOMENT PCA embeddings with strict delayed continual probes.

This experiment consumes a previously frozen MOMENT embedding manifest, recomputes
MOMENT summaries/PCA under the same pinned model, verifies PCA component hashes,
then searches a small predeclared continual-probe family.  Probe weights are
initially fit on 2020-2022 only.  From the first post-fit anchor onward each
prediction is made before the current sample is queued, and labels are released
for online updates only once the next-bar-entry plus 48h hold exit open is
observable.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_embedding_probe_alpha import fit_pca_representations, optional_file_sha256
from training.search_chronos2_zero_shot_alpha import ROLLING_WINDOWS, SCORE_QUANTILES, anchor_hour_indices, causal_hourly_frame
from training.search_invariant_groupdro_alpha import _score_quality
from training.search_moment_embedding_probe_alpha import (
    CONTEXT_HOURS,
    EMBEDDING_VARIATES,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SOURCE_URLS,
    _load_moment_pipeline,
    _model_commit_hash,
    extract_embedding_summaries,
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

SEED = 713
PCA_DIMS = (16, 32)
SIDES = ("long", "short", "both")
MIN_SCORE_HISTORY = 200


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    objective: Literal["classification", "regression"]
    hidden_dim: int | None
    lr: float
    replay: int
    initial_epochs: int


PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec("linear_cls_fast", "classification", None, 0.01, 32, 200),
    ProbeSpec("linear_cls_slow", "classification", None, 0.003, 128, 200),
    ProbeSpec("mlp_cls_fast", "classification", 32, 0.001, 32, 150),
    ProbeSpec("mlp_cls_slow", "classification", 32, 0.0003, 128, 150),
    ProbeSpec("linear_reg_slow", "regression", None, 0.003, 128, 200),
    ProbeSpec("mlp_reg_slow", "regression", 32, 0.0003, 128, 150),
)


def strict_validate_paths(output: str, manifest_output: str, source_manifest: str) -> None:
    validate_output_paths(output, manifest_output)
    resolved = {"output": Path(output).resolve(), "manifest": Path(manifest_output).resolve(), "source": Path(source_manifest).resolve()}
    if resolved["output"] == resolved["source"] or resolved["manifest"] == resolved["source"]:
        raise ValueError("source manifest must be distinct from output and frozen manifest paths")
    if not Path(source_manifest).exists():
        raise FileNotFoundError(f"source manifest not found: {source_manifest}")
    for label in ("output", "manifest"):
        if resolved[label].exists():
            raise FileExistsError(f"refusing to overwrite existing {label}: {resolved[label]}")


def load_and_verify_source_manifest(path: str, *, model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("later_metrics_included") is not False:
        raise ValueError("source manifest must have later_metrics_included=false")
    model = manifest.get("model", {})
    if model.get("id") != model_id or model.get("revision") != model_revision:
        raise ValueError("source manifest MOMENT model id/revision does not match pinned constants")
    return manifest


def assert_pca_hashes_match_source(pca_metadata: dict[str, Any], source_manifest: dict[str, Any], dimensions: tuple[int, ...] = PCA_DIMS) -> None:
    source_repr = source_manifest.get("representation", {})
    for dim in dimensions:
        key = f"pca{dim}"
        expected = source_repr.get(key, {}).get("components_sha256")
        actual = pca_metadata.get(key, {}).get("components_sha256")
        if not expected:
            raise ValueError(f"source manifest missing {key} component hash")
        if actual != expected:
            raise ValueError(f"PCA component hash mismatch for {key}: recomputed {actual}, source {expected}")


def assert_data_hashes_match_source(
    data_hashes: dict[str, str | None], source_manifest: dict[str, Any]
) -> None:
    source_hashes = source_manifest.get("data_sha256", {})
    for name, actual in data_hashes.items():
        expected = source_hashes.get(name)
        if expected != actual:
            raise ValueError(
                f"source data hash mismatch for {name}: current {actual}, source {expected}"
            )


def label_ready_positions(signal_positions: np.ndarray, *, entry_delay_bars: int = 1, hold_bars: int = HOLD_BARS) -> np.ndarray:
    return np.asarray(signal_positions, dtype=np.int64) + int(entry_delay_bars) + int(hold_bars)


def fit_target_calibration(targets: np.ndarray, fit_mask: np.ndarray) -> dict[str, float]:
    fit_values = np.asarray(targets, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    finite = fit_values[np.isfinite(fit_values)]
    if len(finite) < 3:
        raise ValueError("not enough finite fit targets")
    low, high = np.quantile(finite, (0.30, 0.70))
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return {"low": float(low), "high": float(high), "scale": scale, "clip_std": 3.0}


def make_class_labels(targets: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    labels = np.ones(len(targets), dtype=np.int64)
    labels[np.asarray(targets, dtype=float) <= calibration["low"]] = 0
    labels[np.asarray(targets, dtype=float) >= calibration["high"]] = 2
    return labels


def make_reg_targets(targets: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    y = np.asarray(targets, dtype=float) / float(calibration["scale"])
    return np.clip(y, -float(calibration["clip_std"]), float(calibration["clip_std"])).astype(np.float32)


def fit_standardizer(matrix: np.ndarray, fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit_rows = np.asarray(matrix, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    mean = np.nanmean(fit_rows, axis=0).astype(np.float32)
    std = np.nanstd(fit_rows, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def apply_standardizer(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (np.asarray(matrix, dtype=np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ContinualProbe:
    def __init__(self, input_dim: int, spec: ProbeSpec, *, seed: int = SEED):
        torch.manual_seed(seed)
        self.spec = spec
        if spec.hidden_dim is None:
            self.net = nn.Linear(input_dim, 3 if spec.objective == "classification" else 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, spec.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(spec.hidden_dim),
                nn.Linear(spec.hidden_dim, 3 if spec.objective == "classification" else 1),
            )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=spec.lr, weight_decay=0.01)
        self.loss = nn.CrossEntropyLoss() if spec.objective == "classification" else nn.HuberLoss()

    def train_batch(self, x: np.ndarray, y: np.ndarray, *, epochs: int = 1) -> None:
        if len(x) == 0:
            return
        xt = torch.as_tensor(x, dtype=torch.float32)
        if self.spec.objective == "classification":
            yt = torch.as_tensor(y, dtype=torch.long)
        else:
            yt = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.net.train()
        for _ in range(int(epochs)):
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.net(xt)
            loss = self.loss(pred, yt)
            loss.backward()
            self.optimizer.step()

    def score_one(self, x: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.as_tensor(x[None, :], dtype=torch.float32))
            if self.spec.objective == "classification":
                probs = torch.softmax(out[0], dim=0)
                return float(probs[2] - probs[0])
            return float(out.reshape(-1)[0])


def _fit_initial_probe(x: np.ndarray, targets: np.ndarray, fit_mask: np.ndarray, spec: ProbeSpec, calibration: dict[str, float], *, seed: int = SEED) -> ContinualProbe:
    probe = ContinualProbe(x.shape[1], spec, seed=seed)
    fit_targets = np.asarray(targets, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    y = (
        make_class_labels(fit_targets, calibration)
        if spec.objective == "classification"
        else make_reg_targets(fit_targets, calibration)
    )
    probe.train_batch(x[fit_mask], y, epochs=spec.initial_epochs)
    return probe


def delayed_continual_scores(
    probe: ContinualProbe,
    *,
    matrix: np.ndarray,
    targets: np.ndarray,
    signal_positions: np.ndarray,
    ready_positions: np.ndarray,
    stream_start_index: int,
    calibration: dict[str, float],
    replay_init_indices: np.ndarray | None = None,
    stop_before_signal_position: int | None = None,
    event_log: list[tuple[str, int, tuple[int, ...]]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict-before-queue streaming with labels materialized only on release."""
    matrix = np.asarray(matrix, dtype=np.float32)
    targets = np.asarray(targets, dtype=float)
    signal_positions = np.asarray(signal_positions, dtype=np.int64)
    ready_positions = np.asarray(ready_positions, dtype=np.int64)
    if np.any(np.diff(signal_positions) <= 0) or np.any(np.diff(ready_positions) <= 0):
        raise ValueError("signal and ready positions must be strictly increasing")
    if np.any(ready_positions <= signal_positions):
        raise ValueError("labels must mature after signal")

    scores = np.full(len(signal_positions), np.nan, dtype=np.float32)
    pending: deque[tuple[int, int]] = deque()
    replay: deque[int] = deque(maxlen=probe.spec.replay)
    if replay_init_indices is not None:
        for idx in np.asarray(replay_init_indices, dtype=np.int64):
            if idx < stream_start_index and ready_positions[idx] <= signal_positions[stream_start_index]:
                replay.append(int(idx))

    label_reads: list[int] = []
    queued = 0
    updates = 0
    def released_targets(indices: np.ndarray) -> np.ndarray:
        released_values = targets[np.asarray(indices, dtype=np.int64)]
        if probe.spec.objective == "classification":
            return make_class_labels(released_values, calibration)
        return make_reg_targets(released_values, calibration)

    for index in range(int(stream_start_index), len(signal_positions)):
        signal_pos = int(signal_positions[index])
        if stop_before_signal_position is not None and signal_pos >= int(stop_before_signal_position):
            break
        while pending and pending[0][0] <= signal_pos:
            _ready_pos, released_index = pending.popleft()
            label_reads.append(released_index)
            replay.append(released_index)
            batch_indices = np.fromiter(replay, dtype=np.int64)
            probe.train_batch(matrix[batch_indices], released_targets(batch_indices), epochs=1)
            updates += 1
            if event_log is not None:
                event_log.append(("learn", released_index, tuple(int(i) for i in batch_indices)))

        if event_log is not None:
            event_log.append(("predict", index, tuple(int(i) for i in replay)))
        scores[index] = probe.score_one(matrix[index])
        pending.append((int(ready_positions[index]), int(index)))
        queued += 1
        if event_log is not None:
            event_log.append(("queue", index, ()))

    return scores, {
        "stream_start_index": int(stream_start_index),
        "queued_samples": int(queued),
        "online_updates": int(updates),
        "pending_after_stop": int(len(pending)),
        "label_read_indices": label_reads,
        "replay_capacity": int(probe.spec.replay),
        "predict_before_current_label_queued": True,
        "label_ready_offset_bars": int(1 + HOLD_BARS),
    }


def selection_window_signal_hash(long_active: np.ndarray, short_active: np.ndarray, *, market: pd.DataFrame, dates: pd.Series) -> str:
    return effective_selection_signal_hash(market, dates, long_active, short_active, window=WINDOWS["holdout2023"])


def _policy_masks(scores: np.ndarray, positions: np.ndarray, market_size: int, rolling_window: int, quantile: float, side_policy: str) -> tuple[np.ndarray, np.ndarray]:
    low, high = causal_rolling_thresholds(scores, window=rolling_window, quantile=quantile, min_periods=MIN_SCORE_HISTORY)
    return signed_dynamic_policy_masks(scores, positions, market_size, side_policy=side_policy, low_thresholds=low, high_thresholds=high)


def _select_distinct_top10(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    candidates.sort(key=lambda row: (row["holdout2023"]["ratio"], row["holdout2023"]["return_pct"], row["holdout2023"]["trades"]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in candidates:
        signal_hash = candidate["signal_hash"]
        if signal_hash in selected_signals:
            candidate.pop("_long", None); candidate.pop("_short", None)
            continue
        long_active = candidate.pop("_long")
        short_active = candidate.pop("_short")
        selected.append(candidate)
        selected_signals[signal_hash] = (long_active, short_active)
        if len(selected) == 10:
            break
    return selected, selected_signals


def _jsonable_probe_spec(spec: ProbeSpec) -> dict[str, Any]:
    return {"name": spec.name, "objective": spec.objective, "hidden_dim": spec.hidden_dim, "lr": spec.lr, "recent_replay": spec.replay, "initial_epochs": spec.initial_epochs, "optimizer": "AdamW", "weight_decay": 0.01}


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict_validate_paths(args.output, args.manifest_output, args.source_manifest)
    source_manifest = load_and_verify_source_manifest(args.source_manifest, model_id=args.model_id, model_revision=args.model_revision)
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    input_csv = args.market_csv or args.input_csv
    cfg = Config(input_csv=input_csv, output=args.output, funding_csv=args.funding_csv, premium_csv=args.premium_csv, exclude_from=args.exclude_from)
    data_hashes = {
        "market": _file_sha256(input_csv),
        "funding": optional_file_sha256(args.funding_csv),
        "premium": optional_file_sha256(args.premium_csv),
    }
    assert_data_hashes_match_source(data_hashes, source_manifest)
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    positions, targets, _ = anchor_dataset(market, pd.DataFrame(index=market.index))
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    ready_positions = label_ready_positions(positions)
    masks = {name: split_mask_for_anchors(dates, positions, *bounds) for name, bounds in WINDOWS.items()}
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)

    pipeline = _load_moment_pipeline(args.model_id, args.model_revision)
    model_context = int(getattr(pipeline.config, "seq_len", CONTEXT_HOURS))
    if int(args.context_hours) != model_context:
        raise ValueError(f"MOMENT context must match pretrained seq_len={model_context}, got {args.context_hours}")
    model_commit = _model_commit_hash(pipeline)
    if model_commit is not None and model_commit != args.model_revision:
        raise ValueError("loaded MOMENT model revision does not match pinned revision")
    summaries, valid_anchor_indices, embedding_metadata = extract_embedding_summaries(pipeline, hourly, hour_indices, context_hours=args.context_hours, chunk_size=args.chunk_size, batch_size=args.batch_size, device=args.device)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    valid_mask = np.zeros(len(positions), dtype=bool); valid_mask[valid_anchor_indices] = True
    pca_representations, pca_metadata = fit_pca_representations(summaries, masks["fit2020_2022"], valid_mask, dimensions=PCA_DIMS)
    assert_pca_hashes_match_source(pca_metadata, source_manifest, PCA_DIMS)

    fit_valid_mask = masks["fit2020_2022"] & valid_mask
    fit_indices = np.flatnonzero(fit_valid_mask)
    if len(fit_indices) == 0:
        raise ValueError("no valid fit anchors")
    stream_start_index = int(fit_indices[-1] + 1)
    if stream_start_index >= len(positions):
        raise ValueError("no post-fit anchors for continual stream")
    calibration = fit_target_calibration(targets, fit_valid_mask)

    cutoff_2024_pos = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64("2024-01-01"), side="left"))
    raw_candidates: list[dict[str, Any]] = []
    phase1_diagnostics: dict[str, Any] = {}
    for rep_name, rep in pca_representations.items():
        mean, std = fit_standardizer(rep, fit_valid_mask)
        x = apply_standardizer(rep, mean, std)
        replay_init = fit_indices[ready_positions[fit_indices] <= positions[stream_start_index]][-max(s.replay for s in PROBE_SPECS):]
        for spec in PROBE_SPECS:
            stream_id = f"moment_{rep_name}_{spec.name}"
            print(f"phase1 streaming {stream_id}", file=sys.stderr, flush=True)
            probe = _fit_initial_probe(x, targets, fit_valid_mask, spec, calibration)
            scores, diag = delayed_continual_scores(probe, matrix=x, targets=targets, signal_positions=positions, ready_positions=ready_positions, stream_start_index=stream_start_index, calibration=calibration, replay_init_indices=replay_init[-spec.replay:], stop_before_signal_position=cutoff_2024_pos)
            scores[~valid_mask] = np.nan
            phase1_diagnostics[stream_id] = {"representation": rep_name, "standardizer_fit_samples": int(fit_valid_mask.sum()), "probe": _jsonable_probe_spec(spec), "stream": {k: v for k, v in diag.items() if k != "label_read_indices"}}
            for rolling_window in ROLLING_WINDOWS:
                for quantile in SCORE_QUANTILES:
                    for side_policy in SIDES:
                        long_active, short_active = _policy_masks(scores, positions, len(market), rolling_window, quantile, side_policy)
                        holdout = sim(market, dates, long_active, short_active, cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, "holdout2023")
                        if holdout["trades"] < 8 or holdout["return_pct"] <= 0.0:
                            continue
                        raw_candidates.append({
                            "stream_id": stream_id,
                            "representation": rep_name,
                            "probe_spec": spec.name,
                            "objective": spec.objective,
                            "rolling_score_window_anchors": rolling_window,
                            "score_quantile": quantile,
                            "minimum_score_history": MIN_SCORE_HISTORY,
                            "side_policy": side_policy,
                            "hold_bars": HOLD_BARS,
                            "anchor_stride_bars": ANCHOR_STRIDE,
                            "holdout2023": holdout,
                            "signal_hash": selection_window_signal_hash(long_active, short_active, market=market, dates=dates),
                            "_long": long_active,
                            "_short": short_active,
                        })

    selected, _phase1_selected_signals = _select_distinct_top10(raw_candidates)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "source_manifest_sha256": _file_sha256(args.source_manifest),
        "model": {"id": args.model_id, "revision": args.model_revision, "loaded_commit_hash": model_commit, "source_urls": MODEL_SOURCE_URLS, "momentfm_version": importlib.metadata.version("momentfm")},
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), **embedding_metadata},
        "representation": pca_metadata,
        "target": {"name": "next_48h_open_to_open_return", "fit_only_tail_thresholds": {"low": calibration["low"], "high": calibration["high"]}, "fit_only_regression_scale": calibration["scale"], "regression_clip_std": calibration["clip_std"]},
        "strict_protocol": {"fit_window": "2020-2022 only for standardizer, thresholds/scale, initial weights", "streaming": "predict before queue; release only ready_position <= current signal position; queued payload is sample index only", "phase1": "signal dates <2024 only; Top-10 frozen before 2024+ diagnostics", "phase2": "selected algorithms rerun after manifest write and 2023 executable hashes asserted"},
        "probe_family": [_jsonable_probe_spec(s) for s in PROBE_SPECS],
        "top10": selected,
        "trial_counts": {"representations": len(PCA_DIMS), "probe_specs": len(PROBE_SPECS), "score_streams": len(PCA_DIMS) * len(PROBE_SPECS), "rolling_windows": len(ROLLING_WINDOWS), "score_quantiles": len(SCORE_QUANTILES), "side_policies": len(SIDES), "total_policy_specs": len(PCA_DIMS) * len(PROBE_SPECS) * len(ROLLING_WINDOWS) * len(SCORE_QUANTILES) * len(SIDES), "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    unique_specs = {(row["representation"], row["probe_spec"]) for row in selected}
    phase2_diagnostics: dict[str, Any] = {}
    phase2_scores: dict[str, np.ndarray] = {}
    spec_by_name = {s.name: s for s in PROBE_SPECS}
    for rep_name, spec_name in sorted(unique_specs):
        spec = spec_by_name[spec_name]
        rep = pca_representations[rep_name]
        mean, std = fit_standardizer(rep, fit_valid_mask)
        x = apply_standardizer(rep, mean, std)
        replay_init = fit_indices[ready_positions[fit_indices] <= positions[stream_start_index]][-spec.replay:]
        print(f"phase2 streaming moment_{rep_name}_{spec.name}", file=sys.stderr, flush=True)
        probe = _fit_initial_probe(x, targets, fit_valid_mask, spec, calibration)
        scores, diag = delayed_continual_scores(probe, matrix=x, targets=targets, signal_positions=positions, ready_positions=ready_positions, stream_start_index=stream_start_index, calibration=calibration, replay_init_indices=replay_init)
        scores[~valid_mask] = np.nan
        stream_id = f"moment_{rep_name}_{spec.name}"
        phase2_scores[stream_id] = scores
        phase2_diagnostics[stream_id] = {
            "stream": {k: v for k, v in diag.items() if k != "label_read_indices"},
            "score_quality": _score_quality(scores, targets, masks),
        }

    for rank, row in enumerate(selected, start=1):
        scores = phase2_scores[row["stream_id"]]
        long_active, short_active = _policy_masks(scores, positions, len(market), int(row["rolling_score_window_anchors"]), float(row["score_quantile"]), row["side_policy"])
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

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "frozen MOMENT PCA continual probes; fit-only 2020-2022; strict 48h delayed online updates; 2023 Top-10 frozen before 2024+ metrics", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "model": manifest["model"], "input": manifest["input"], "representation": pca_metadata, "phase1_model_diagnostics": phase1_diagnostics, "phase2_model_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
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
