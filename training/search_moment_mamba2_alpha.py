"""Strict-causal MOMENT PCA32 sequence Mamba2 alpha search.

A tiny HuggingFace ``transformers.Mamba2Model`` consumes sequences of frozen
MOMENT PCA32 anchor embeddings.  MOMENT/PCA/data hashes must match a frozen
source manifest.  Model weights and policy thresholds are selected only from
2020-2023 information; 2024+ features/metrics are computed only after the
Top-10 manifest has been written.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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

import torch
from torch import nn
import torch.nn.functional as F

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_embedding_probe_alpha import fit_pca_representations, optional_file_sha256
from training.search_chronos2_zero_shot_alpha import ROLLING_WINDOWS, SCORE_QUANTILES, anchor_hour_indices, causal_hourly_frame
from training.search_invariant_groupdro_alpha import _score_quality, half_year_environments, environment_risk_objective
from training.search_moment_continual_probe_alpha import (
    assert_data_hashes_match_source,
    assert_pca_hashes_match_source,
    load_and_verify_source_manifest,
    strict_validate_paths,
)
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
from training.search_river_online_alpha import causal_rolling_thresholds
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
PCA_DIMS = (32,)
SEQUENCE_LENGTHS = (32, 64)
HORIZONS: dict[str, int] = {"12h": 144, "48h": 576, "7d": 2016}
OBJECTIVES = ("erm", "vrex1", "groupdro")
SCORE_TRANSFORMS = ("h48", "mean_12_48", "mean_all", "consensus_all")
SIDES = ("long", "short", "both")
MIN_SCORE_HISTORY = 200


@dataclass(frozen=True)
class MambaSpec:
    sequence_length: int
    objective: Literal["erm", "vrex1", "groupdro"]

    @property
    def stream_prefix(self) -> str:
        return f"mamba2_seq{self.sequence_length}_{self.objective}"


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def tiny_mamba2_config() -> Any:
    from transformers import Mamba2Config

    return Mamba2Config(
        hidden_size=32,
        num_hidden_layers=1,
        state_size=16,
        conv_kernel=4,
        expand=2,
        head_dim=16,
        num_heads=4,
        n_groups=4,
        use_cache=False,
        chunk_size=64,
        vocab_size=1,
    )


class TinyMamba2Regressor(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        from transformers import Mamba2Model

        self.config = tiny_mamba2_config()
        self.input_norm = nn.LayerNorm(32)
        self.dropout = nn.Dropout(float(dropout))
        self.backbone = Mamba2Model(self.config)
        self.head = nn.Linear(32, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.input_norm(inputs))
        out = self.backbone(inputs_embeds=x, use_cache=False)
        return self.head(out.last_hidden_state[:, -1, :])


def mamba2_transformers_metadata(model: TinyMamba2Regressor | None = None) -> dict[str, Any]:
    import transformers

    config = tiny_mamba2_config() if model is None else model.config
    return {
        "library": "transformers",
        "version": getattr(transformers, "__version__", "unknown"),
        "git_version": getattr(transformers, "__git_version__", None),
        "mamba2_config": config.to_dict(),
        "input": "inputs_embeds: PCA32 anchor sequence ending at current anchor",
    }


def multi_horizon_open_returns(
    market: pd.DataFrame,
    positions: np.ndarray,
    horizons: dict[str, int] = HORIZONS,
    *,
    available_before_position: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Next-open to future-open log returns for all horizons.

    Anchor position ``p`` enters at ``p+1`` open and horizon ``h`` exits at
    ``p+1+h`` open.  Out-of-bounds or non-positive opens produce NaN.
    """
    positions = np.asarray(positions, dtype=np.int64)
    opens = market["open"].to_numpy(float)
    y = np.full((len(positions), len(horizons)), np.nan, dtype=np.float32)
    details: dict[str, Any] = {"entry_offset_bars": 1, "horizons": horizons.copy(), "columns": list(horizons)}
    for col, (_name, bars) in enumerate(horizons.items()):
        entry = positions + 1
        exit_pos = entry + int(bars)
        upper_bound = (
            len(opens)
            if available_before_position is None
            else min(len(opens), int(available_before_position))
        )
        in_bounds = (entry < upper_bound) & (exit_pos < upper_bound)
        valid = np.zeros(len(positions), dtype=bool)
        valid[in_bounds] = (
            np.isfinite(opens[entry[in_bounds]])
            & np.isfinite(opens[exit_pos[in_bounds]])
            & (opens[entry[in_bounds]] > 0)
            & (opens[exit_pos[in_bounds]] > 0)
        )
        y[valid, col] = np.log(opens[exit_pos[valid]] / opens[entry[valid]]).astype(np.float32)
    return y, details


def fit_horizon_scales(targets: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit = np.asarray(targets, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    scale = np.nanstd(fit, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return scale


def normalize_targets_no_mean(targets: np.ndarray, scales: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    y = np.asarray(targets, dtype=np.float32) / np.asarray(scales, dtype=np.float32).reshape(1, -1)
    return np.clip(y, -float(clip_std), float(clip_std)).astype(np.float32)


def sequence_valid_mask(valid_anchor_mask: np.ndarray, sequence_length: int) -> np.ndarray:
    valid_anchor_mask = np.asarray(valid_anchor_mask, dtype=bool)
    output = np.zeros(len(valid_anchor_mask), dtype=bool)
    n = int(sequence_length)
    for i in range(n - 1, len(valid_anchor_mask)):
        output[i] = bool(valid_anchor_mask[i - n + 1 : i + 1].all())
    return output


def build_sequences(representation: np.ndarray, valid_anchor_mask: np.ndarray, sequence_length: int, indices: np.ndarray) -> np.ndarray:
    rep = np.asarray(representation, dtype=np.float32)
    valid = np.asarray(valid_anchor_mask, dtype=bool)
    n = int(sequence_length)
    indices = np.asarray(indices, dtype=np.int64)
    out = np.empty((len(indices), n, rep.shape[1]), dtype=np.float32)
    for row, i in enumerate(indices):
        start = int(i) - n + 1
        if start < 0 or not valid[start : int(i) + 1].all():
            raise ValueError(f"invalid causal sequence ending at anchor {int(i)}")
        out[row] = rep[start : int(i) + 1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_admissible_mask(
    dates: pd.Series,
    positions: np.ndarray,
    targets: np.ndarray,
    valid_anchor_mask: np.ndarray,
    sequence_length: int,
    *,
    fit_start: str = "2020-01-01",
    fit_end: str = "2023-01-01",
) -> np.ndarray:
    dates = pd.to_datetime(dates)
    positions = np.asarray(positions, dtype=np.int64)
    signal_dates = dates.iloc[positions].to_numpy()
    max_horizon = max(HORIZONS.values())
    exit_positions = positions + 1 + max_horizon
    exit_in_bounds = exit_positions < len(dates)
    exit_before_end = np.zeros(len(positions), dtype=bool)
    if exit_in_bounds.any():
        exit_dates = dates.iloc[exit_positions[exit_in_bounds]].to_numpy()
        exit_before_end[exit_in_bounds] = exit_dates < np.datetime64(pd.Timestamp(fit_end))
    start = np.datetime64(pd.Timestamp(fit_start))
    end = np.datetime64(pd.Timestamp(fit_end))
    seq_valid = sequence_valid_mask(valid_anchor_mask, sequence_length)
    finite_targets = np.isfinite(np.asarray(targets, dtype=float)).all(axis=1)
    return (signal_dates >= start) & (signal_dates < end) & exit_before_end & seq_valid & finite_targets


def first_post_fit_index(dates: pd.Series, positions: np.ndarray, fit_mask: np.ndarray) -> int:
    """First anchor strictly after the fit/in-sample region; scores before it stay NaN."""
    del dates
    fit_indices = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    return int(fit_indices[-1] + 1) if len(fit_indices) else 0


def train_mamba_model(
    representation: np.ndarray,
    valid_anchor_mask: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    signal_dates: pd.Series,
    spec: MambaSpec,
    scales: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[TinyMamba2Regressor, dict[str, Any]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_indices = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    if len(fit_indices) < 8:
        raise ValueError("not enough fit sequences for Mamba2 training")
    x = torch.as_tensor(build_sequences(representation, valid_anchor_mask, spec.sequence_length, fit_indices), dtype=torch.float32, device=device)
    y_np = normalize_targets_no_mean(targets[fit_indices], scales)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    env_all = half_year_environments(pd.to_datetime(signal_dates).reset_index(drop=True))
    env = torch.as_tensor(env_all[fit_indices], dtype=torch.long, device=device)
    unique_envs = torch.unique(env, sorted=True)
    model = TinyMamba2Regressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    group_weights: torch.Tensor | None = None
    losses: list[float] = []
    model.train()
    for _epoch in range(80):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        env_losses = torch.stack([F.mse_loss(pred[env == e], y[env == e]) for e in unique_envs])
        if spec.objective == "erm":
            loss = env_losses.mean()
        elif spec.objective == "vrex1":
            loss, group_weights = environment_risk_objective(env_losses, objective="vrex", vrex_penalty=1.0, group_weights=group_weights)
        elif spec.objective == "groupdro":
            loss, group_weights = environment_risk_objective(env_losses, objective="groupdro", groupdro_eta=0.05, group_weights=group_weights)
        else:
            raise ValueError(spec.objective)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return model, {
        "fit_sequences": int(len(fit_indices)),
        "epochs": 80,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.01)",
        "gradient_clip_norm": 1.0,
        "objective": spec.objective,
        "final_loss": losses[-1],
        "environment_count": int(len(unique_envs)),
        "device": str(device),
    }


def predict_mamba_scores(
    model: TinyMamba2Regressor,
    representation: np.ndarray,
    valid_anchor_mask: np.ndarray,
    sequence_length: int,
    *,
    start_index: int,
    stop_index: int | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    seq_valid = sequence_valid_mask(valid_anchor_mask, sequence_length)
    stop = len(seq_valid) if stop_index is None else min(int(stop_index), len(seq_valid))
    eligible = np.flatnonzero(seq_valid & (np.arange(len(seq_valid)) >= int(start_index)) & (np.arange(len(seq_valid)) < stop))
    scores = np.full((len(seq_valid), 3), np.nan, dtype=np.float32)
    if len(eligible) == 0:
        return scores
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for s in range(0, len(eligible), int(batch_size)):
            idx = eligible[s : s + int(batch_size)]
            x = torch.as_tensor(build_sequences(representation, valid_anchor_mask, sequence_length, idx), dtype=torch.float32, device=device)
            scores[idx] = model(x).detach().cpu().numpy().astype(np.float32)
    return scores


def transform_score_streams(predictions: np.ndarray, *, prefix: str) -> dict[str, np.ndarray]:
    p = np.asarray(predictions, dtype=np.float32)
    finite_all = np.isfinite(p).all(axis=1)
    mean_12_48 = np.full(len(p), np.nan, dtype=np.float32)
    mean_all = np.full(len(p), np.nan, dtype=np.float32)
    mean_12_48[finite_all] = p[finite_all, :2].mean(axis=1)
    mean_all[finite_all] = p[finite_all].mean(axis=1)
    signs = np.sign(p)
    consensus = np.full(len(p), np.nan, dtype=np.float32)
    agree = finite_all & np.all(signs == signs[:, [0]], axis=1)
    consensus[finite_all & ~agree] = 0.0
    consensus[agree] = mean_all[agree]
    return {
        f"{prefix}_h48": p[:, 1].copy(),
        f"{prefix}_mean_12_48": mean_12_48,
        f"{prefix}_mean_all": mean_all,
        f"{prefix}_consensus_all": consensus,
    }


def selection_window_signal_hash(long_active: np.ndarray, short_active: np.ndarray, *, market: pd.DataFrame, dates: pd.Series) -> str:
    return effective_selection_signal_hash(market, dates, long_active, short_active, window=WINDOWS["holdout2023"])


def policy_masks(scores: np.ndarray, positions: np.ndarray, market_size: int, rolling_window: int, quantile: float, side_policy: str) -> tuple[np.ndarray, np.ndarray]:
    low, high = causal_rolling_thresholds(scores, window=rolling_window, quantile=quantile, min_periods=MIN_SCORE_HISTORY)
    return signed_dynamic_policy_masks(scores, positions, market_size, side_policy=side_policy, low_thresholds=low, high_thresholds=high)


def select_distinct_top10(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    candidates.sort(key=lambda r: (r["holdout2023"]["ratio"], r["holdout2023"]["return_pct"], r["holdout2023"]["trades"]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in candidates:
        h = c["signal_hash"]
        if h in selected_signals:
            c.pop("_long", None); c.pop("_short", None)
            continue
        long_active = c.pop("_long")
        short_active = c.pop("_short")
        selected.append(c)
        selected_signals[h] = (long_active, short_active)
        if len(selected) == 10:
            break
    return selected, selected_signals


def extract_pca32_for_phase(args: argparse.Namespace, market: pd.DataFrame, dates: pd.Series, positions: np.ndarray, masks: dict[str, np.ndarray], source_manifest: dict[str, Any], *, before_2024_only: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)
    pipeline = _load_moment_pipeline(args.model_id, args.model_revision)
    model_context = int(getattr(pipeline.config, "seq_len", CONTEXT_HOURS))
    if int(args.context_hours) != model_context:
        raise ValueError(f"MOMENT context must match pretrained seq_len={model_context}, got {args.context_hours}")
    model_commit = _model_commit_hash(pipeline)
    if model_commit is not None and model_commit != args.model_revision:
        raise ValueError("loaded MOMENT model revision does not match pinned revision")
    if before_2024_only:
        phase_mask = pd.to_datetime(dates.iloc[positions]).to_numpy() < np.datetime64(pd.Timestamp("2024-01-01"))
        phase_hour_indices = hour_indices.copy()
        phase_hour_indices[~phase_mask] = -10**12
    else:
        phase_hour_indices = hour_indices
    summaries, valid_anchor_indices, embedding_metadata = extract_embedding_summaries(
        pipeline,
        hourly,
        phase_hour_indices,
        context_hours=args.context_hours,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    valid_mask = np.zeros(len(positions), dtype=bool)
    valid_mask[valid_anchor_indices] = True
    pca_reps, pca_metadata = fit_pca_representations(summaries, masks["fit2020_2022"], valid_mask, dimensions=PCA_DIMS)
    assert_pca_hashes_match_source(pca_metadata, source_manifest, dimensions=PCA_DIMS)
    model_metadata = {
        "id": args.model_id,
        "revision": args.model_revision,
        "loaded_commit_hash": model_commit,
        "source_urls": MODEL_SOURCE_URLS,
        "momentfm_version": importlib.metadata.version("momentfm"),
        "loader": "MOMENTPipeline.from_pretrained(model_kwargs={'task_name':'embedding'}); init(); cuda(); eval()",
    }
    return pca_reps["pca32"], valid_mask, pca_metadata, embedding_metadata, model_metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    strict_validate_paths(args.output, args.manifest_output, args.source_manifest)
    source_manifest = load_and_verify_source_manifest(args.source_manifest, model_id=args.model_id, model_revision=args.model_revision)
    for name, bounds in WINDOWS.items():
        state_sim.W[name] = bounds
    input_csv = args.market_csv or args.input_csv
    cfg = Config(input_csv=input_csv, output=args.output, funding_csv=args.funding_csv, premium_csv=args.premium_csv, exclude_from=args.exclude_from)
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    positions, _, _ = anchor_dataset(market, pd.DataFrame(index=market.index))
    signal_dates = dates.iloc[positions].reset_index(drop=True)
    masks = {name: split_mask_for_anchors(dates, positions, *bounds) for name, bounds in WINDOWS.items()}
    cutoff_2024_pos = int(
        np.searchsorted(
            dates.to_numpy(dtype="datetime64[ns]"),
            np.datetime64("2024-01-01"),
            side="left",
        )
    )
    phase1_targets, target_meta = multi_horizon_open_returns(
        market,
        positions,
        available_before_position=cutoff_2024_pos,
    )
    data_hashes = {"market": _file_sha256(input_csv), "funding": optional_file_sha256(args.funding_csv), "premium": optional_file_sha256(args.premium_csv)}
    assert_data_hashes_match_source(data_hashes, source_manifest)

    # Phase 1: no 2024+ feature extraction or inference.
    rep, valid_mask, pca_metadata, embedding_metadata, model_metadata = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=True)
    phase1_models: dict[tuple[int, str], TinyMamba2Regressor] = {}
    phase1_diagnostics: dict[str, Any] = {}
    phase1_scores: dict[str, np.ndarray] = {}
    raw_candidates: list[dict[str, Any]] = []
    for seq_len in SEQUENCE_LENGTHS:
        fit_mask = fit_admissible_mask(
            dates, positions, phase1_targets, valid_mask, seq_len
        )
        start_index = first_post_fit_index(dates, positions, fit_mask)
        stop_index = int(np.searchsorted(signal_dates.to_numpy(), np.datetime64(pd.Timestamp("2024-01-01")), side="left"))
        scales = fit_horizon_scales(phase1_targets, fit_mask)
        for objective in OBJECTIVES:
            spec = MambaSpec(seq_len, objective)  # type: ignore[arg-type]
            print(f"training {spec.stream_prefix}", file=sys.stderr, flush=True)
            model, diag = train_mamba_model(
                rep,
                valid_mask,
                phase1_targets,
                fit_mask,
                signal_dates,
                spec,
                scales,
            )
            phase1_models[(seq_len, objective)] = model
            pred = predict_mamba_scores(model, rep, valid_mask, seq_len, start_index=start_index, stop_index=stop_index, batch_size=args.predict_batch_size)
            pred[:start_index] = np.nan
            streams = transform_score_streams(pred, prefix=spec.stream_prefix)
            for stream_id, scores in streams.items():
                scores[:start_index] = np.nan
                phase1_scores[stream_id] = scores
                phase1_diagnostics[stream_id] = {"training": diag, "sequence_length": seq_len, "objective": objective, "fit_horizon_scales_no_mean": scales.tolist(), "first_policy_score_index": start_index, "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all()), "score_quality_prefreeze": _score_quality(scores, phase1_targets[:, 1], {k: masks[k] for k in ("fit2020_2022", "holdout2023")})}
                for rolling_window in ROLLING_WINDOWS:
                    for quantile in SCORE_QUANTILES:
                        for side_policy in SIDES:
                            long_active, short_active = policy_masks(scores, positions, len(market), int(rolling_window), float(quantile), side_policy)
                            holdout = sim(market, dates, long_active, short_active, cfg, HOLD_BARS, ANCHOR_STRIDE, 10.0, 10.0, "holdout2023")
                            if holdout["trades"] < 8 or holdout["return_pct"] <= 0.0:
                                continue
                            raw_candidates.append({
                                "stream_id": stream_id,
                                "sequence_length": seq_len,
                                "objective": objective,
                                "score_transform": stream_id.removeprefix(spec.stream_prefix + "_"),
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
        "source_manifest_sha256": _file_sha256(args.source_manifest),
        "model": model_metadata,
        "state_space_model": mamba2_transformers_metadata(),
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), "anchor_stride_hours": 6, "sequence_lengths": list(SEQUENCE_LENGTHS), **embedding_metadata},
        "representation": pca_metadata,
        "target": {**target_meta, "normalization": "divide each horizon by fit-only std; no mean subtraction; clip +/-3"},
        "strict_protocol": {"phase1": "extract/embed and infer only signal dates <2024; train only admissible 2020-2022 rows; policy scores remain NaN through fit prefix", "selection": "2023 metrics only; positive return and >=8 trades; no future diagnostics in manifest", "phase2": "after manifest write, extract all anchors, infer selected specs, assert 2023 executable hash unchanged, then compute 2024+ metrics"},
        "top10": selected,
        "trial_counts": {"sequence_lengths": len(SEQUENCE_LENGTHS), "objectives": len(OBJECTIVES), "score_transforms": len(SCORE_TRANSFORMS), "score_streams": len(phase1_scores), "rolling_windows": len(ROLLING_WINDOWS), "score_quantiles": len(SCORE_QUANTILES), "side_policies": len(SIDES), "total_policy_specs": len(phase1_scores) * len(ROLLING_WINDOWS) * len(SCORE_QUANTILES) * len(SIDES), "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    phase2_diagnostics: dict[str, Any] = {}
    phase2_scores: dict[str, np.ndarray] = {}
    targets_full, _ = multi_horizon_open_returns(market, positions)
    unique_specs = {(int(row["sequence_length"]), str(row["objective"])) for row in selected}
    if unique_specs:
        rep_full, valid_full, pca_full_metadata, _, _ = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=False)
        assert_pca_hashes_match_source(pca_full_metadata, source_manifest, dimensions=PCA_DIMS)
    else:
        rep_full, valid_full = rep, valid_mask
    for seq_len, objective in sorted(unique_specs):
        fit_mask = fit_admissible_mask(
            dates, positions, targets_full, valid_full, seq_len
        )
        start_index = first_post_fit_index(dates, positions, fit_mask)
        model = phase1_models[(seq_len, objective)]
        pred = predict_mamba_scores(model, rep_full, valid_full, seq_len, start_index=start_index, stop_index=None, batch_size=args.predict_batch_size)
        pred[:start_index] = np.nan
        streams = transform_score_streams(pred, prefix=f"mamba2_seq{seq_len}_{objective}")
        for stream_id, scores in streams.items():
            scores[:start_index] = np.nan
            phase2_scores[stream_id] = scores
            phase2_diagnostics[stream_id] = {"score_quality": _score_quality(scores, targets_full[:, 1], masks), "first_policy_score_index": start_index, "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all())}

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

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "strict MOMENT PCA32 tiny Mamba2; fit-only 2020-2022; 2023 Top-10 frozen before 2024+ features/metrics; next-bar entry; hold576; 6bp/side; full-window CAGR; strict intratrade MDD", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "model": model_metadata, "state_space_model": mamba2_transformers_metadata(), "input": manifest["input"], "representation": pca_metadata, "phase1_model_diagnostics": phase1_diagnostics, "phase2_model_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
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
    parser.add_argument("--predict-batch-size", type=int, default=128)
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
