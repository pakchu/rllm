"""Strict risk-aware distributional MOMENT PCA32 Mamba2 path critic.

This search reuses the frozen MOMENT/PCA32 sequence machinery from
``search_moment_mamba2_alpha`` but predicts a 48h executable path distribution:
future simple return plus long/short adverse excursion quantiles.  Selection is
strictly two-phase: 2023 Top-10 candidates are frozen before any 2024+ feature,
target, or diagnostic is produced.
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

import torch
from torch import nn

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import signed_dynamic_policy_masks
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_chronos2_embedding_probe_alpha import optional_file_sha256
from training.search_chronos2_zero_shot_alpha import ROLLING_WINDOWS, SCORE_QUANTILES
from training.search_invariant_groupdro_alpha import _score_quality, environment_risk_objective, half_year_environments
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
)
from training.search_moment_mamba2_alpha import (
    SEED,
    build_sequences,
    extract_pca32_for_phase,
    first_post_fit_index,
    mamba2_transformers_metadata,
    seed_everything,
    sequence_valid_mask,
    tiny_mamba2_config,
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

PCA_DIMS = (32,)
SEQUENCE_LENGTH = 32
SEQUENCE_LENGTHS = (SEQUENCE_LENGTH,)
OBJECTIVES = ("erm", "vrex1", "groupdro")
TARGET_COMPONENTS = ("simple_return", "long_mae", "short_mae")
QUANTILES = (0.25, 0.50, 0.75)
UTILITY_SPECS = (
    ("median_lambda0.5", "median", 0.5),
    ("median_lambda1.0", "median", 1.0),
    ("conservative_lambda0.5", "conservative", 0.5),
    ("conservative_lambda1.0", "conservative", 1.0),
)
SIDES = ("long", "short", "both")
MIN_SCORE_HISTORY = 200
SOURCE_MANIFEST_NAME_PREFIX = "moment_embedding_probe_top10_manifest"


@dataclass(frozen=True)
class PathCriticSpec:
    sequence_length: int
    objective: Literal["erm", "vrex1", "groupdro"]

    @property
    def stream_prefix(self) -> str:
        return f"pathcritic_mamba2_seq{self.sequence_length}_{self.objective}"


class TinyMamba2PathCritic(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        from transformers import Mamba2Model

        self.config = tiny_mamba2_config()
        self.input_norm = nn.LayerNorm(32)
        self.dropout = nn.Dropout(float(dropout))
        self.backbone = Mamba2Model(self.config)
        self.head = nn.Linear(32, len(TARGET_COMPONENTS) * len(QUANTILES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.input_norm(inputs))
        out = self.backbone(inputs_embeds=x, use_cache=False)
        raw = self.head(out.last_hidden_state[:, -1, :])
        return raw.reshape(-1, len(TARGET_COMPONENTS), len(QUANTILES))


def path_critic_transformers_metadata(model: TinyMamba2PathCritic | None = None) -> dict[str, Any]:
    meta = mamba2_transformers_metadata(None)
    config = tiny_mamba2_config() if model is None else model.config
    meta.update(
        {
            "mamba2_config": config.to_dict(),
            "head": "Linear(32, 3 components x 3 quantiles)",
            "quantiles": list(QUANTILES),
            "components": list(TARGET_COMPONENTS),
            "input": "inputs_embeds: PCA32 anchor sequence length 32 ending at current anchor",
        }
    )
    return meta


def load_required_source_manifest(path: str, *, model_id: str = MODEL_ID, model_revision: str = MODEL_REVISION) -> dict[str, Any]:
    """Load the frozen MOMENT embedding source manifest and verify PCA32 provenance."""
    manifest = load_and_verify_source_manifest(path, model_id=model_id, model_revision=model_revision)
    if SOURCE_MANIFEST_NAME_PREFIX not in Path(path).name:
        # Keep the guard explicit: this experiment must be rooted in the frozen
        # embedding-probe manifest rather than a later derived experiment.
        raise ValueError(
            "source manifest must be the frozen moment_embedding_probe_top10_manifest provenance"
        )
    if "pca32" not in manifest.get("representation", {}):
        raise ValueError("source manifest missing PCA32 representation provenance")
    return manifest


def executable_path_targets_48h(
    market: pd.DataFrame,
    positions: np.ndarray,
    *,
    available_before_position: int | None = None,
    hold_bars: int = HOLD_BARS,
    entry_delay_bars: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return 48h path labels aligned with the simulator's executable path.

    Anchor ``p`` enters at ``open[p+1]`` and exits at ``open[p+1+576]``.  MAE is
    computed over bars ``range(p+1, p+1+576)`` and therefore excludes the exit
    bar, matching strict intratrade simulator semantics.  Rows whose exit is not
    strictly before ``available_before_position`` are NaN.
    """
    positions = np.asarray(positions, dtype=np.int64)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    y = np.full((len(positions), len(TARGET_COMPONENTS)), np.nan, dtype=np.float32)
    upper_bound = len(opens) if available_before_position is None else min(len(opens), int(available_before_position))
    for row, p in enumerate(positions):
        entry_pos = int(p) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        if entry_pos < 0 or exit_pos >= upper_bound:
            continue
        entry = opens[entry_pos]
        exit_price = opens[exit_pos]
        path_lows = lows[entry_pos:exit_pos]
        path_highs = highs[entry_pos:exit_pos]
        if (
            not np.isfinite(entry)
            or not np.isfinite(exit_price)
            or entry <= 0.0
            or len(path_lows) != int(hold_bars)
            or not np.isfinite(path_lows).all()
            or not np.isfinite(path_highs).all()
        ):
            continue
        simple_return = exit_price / entry - 1.0
        long_mae = max(0.0, 1.0 - float(np.min(path_lows)) / entry)
        short_mae = max(0.0, float(np.max(path_highs)) / entry - 1.0)
        y[row] = (simple_return, long_mae, short_mae)
    meta = {
        "components": list(TARGET_COMPONENTS),
        "entry": "open[p+1]",
        "exit": f"open[p+1+{int(hold_bars)}]",
        "intratrade_bars": f"range(p+1, p+1+{int(hold_bars)}) exclusive exit",
        "invalid_rows": "NaN when entry/exit/path is non-finite, non-positive entry, out-of-bounds, or exit not before cutoff",
    }
    return y, meta


def fit_component_scales(targets: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    fit = np.asarray(targets, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    scales = np.nanstd(fit, axis=0).astype(np.float32)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    return scales


def normalize_path_targets_no_mean(targets: np.ndarray, scales: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    y = np.asarray(targets, dtype=np.float32) / np.asarray(scales, dtype=np.float32).reshape(1, -1)
    y[:, 0] = np.clip(y[:, 0], -float(clip_std), float(clip_std))
    y[:, 1:] = np.clip(y[:, 1:], 0.0, float(clip_std))
    return y.astype(np.float32)


def pinball_loss(predictions: torch.Tensor, targets: torch.Tensor, quantiles: tuple[float, ...] = QUANTILES) -> torch.Tensor:
    """Mean pinball loss over rows, path components, and quantiles."""
    q = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device).view(1, 1, -1)
    error = targets.unsqueeze(-1) - predictions
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def fit_admissible_mask(
    dates: pd.Series,
    positions: np.ndarray,
    targets: np.ndarray,
    valid_anchor_mask: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    *,
    fit_start: str = "2020-01-01",
    fit_end: str = "2023-01-01",
    hold_bars: int = HOLD_BARS,
) -> np.ndarray:
    dates = pd.to_datetime(dates)
    positions = np.asarray(positions, dtype=np.int64)
    signal_dates = dates.iloc[positions].to_numpy()
    exit_positions = positions + 1 + int(hold_bars)
    exit_in_bounds = exit_positions < len(dates)
    exit_before_end = np.zeros(len(positions), dtype=bool)
    if exit_in_bounds.any():
        exit_dates = dates.iloc[exit_positions[exit_in_bounds]].to_numpy()
        exit_before_end[exit_in_bounds] = exit_dates < np.datetime64(pd.Timestamp(fit_end))
    start = np.datetime64(pd.Timestamp(fit_start))
    end = np.datetime64(pd.Timestamp(fit_end))
    seq_valid = sequence_valid_mask(valid_anchor_mask, int(sequence_length))
    finite_targets = np.isfinite(np.asarray(targets, dtype=float)).all(axis=1)
    return (signal_dates >= start) & (signal_dates < end) & exit_before_end & seq_valid & finite_targets


def train_path_critic_model(
    representation: np.ndarray,
    valid_anchor_mask: np.ndarray,
    targets: np.ndarray,
    fit_mask: np.ndarray,
    signal_dates: pd.Series,
    spec: PathCriticSpec,
    scales: np.ndarray,
    *,
    seed: int = SEED,
) -> tuple[TinyMamba2PathCritic, dict[str, Any]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_indices = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    if len(fit_indices) < 8:
        raise ValueError("not enough fit sequences for Mamba2 path-critic training")
    x = torch.as_tensor(build_sequences(representation, valid_anchor_mask, spec.sequence_length, fit_indices), dtype=torch.float32, device=device)
    y_np = normalize_path_targets_no_mean(targets[fit_indices], scales)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    env_all = half_year_environments(pd.to_datetime(signal_dates).reset_index(drop=True))
    env = torch.as_tensor(env_all[fit_indices], dtype=torch.long, device=device)
    unique_envs = torch.unique(env, sorted=True)
    model = TinyMamba2PathCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    group_weights: torch.Tensor | None = None
    losses: list[float] = []
    model.train()
    for _epoch in range(80):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        env_losses = torch.stack([pinball_loss(pred[env == e], y[env == e]) for e in unique_envs])
        if spec.objective == "erm":
            loss = env_losses.mean()
        elif spec.objective == "vrex1":
            loss, group_weights = environment_risk_objective(env_losses, objective="vrex", vrex_penalty=1.0, group_weights=group_weights)
        elif spec.objective == "groupdro":
            loss, group_weights = environment_risk_objective(env_losses, objective="groupdro", groupdro_eta=0.05, group_weights=group_weights)
        else:  # pragma: no cover
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
        "loss": "pinball averaged over 3 components x quantiles",
        "final_loss": losses[-1],
        "environment_count": int(len(unique_envs)),
        "device": str(device),
    }


def predict_path_quantiles(
    model: nn.Module,
    representation: np.ndarray,
    valid_anchor_mask: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    *,
    start_index: int,
    stop_index: int | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    seq_valid = sequence_valid_mask(valid_anchor_mask, int(sequence_length))
    stop = len(seq_valid) if stop_index is None else min(int(stop_index), len(seq_valid))
    eligible = np.flatnonzero(seq_valid & (np.arange(len(seq_valid)) >= int(start_index)) & (np.arange(len(seq_valid)) < stop))
    preds = np.full((len(seq_valid), len(TARGET_COMPONENTS), len(QUANTILES)), np.nan, dtype=np.float32)
    if len(eligible) == 0:
        return preds
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for s in range(0, len(eligible), int(batch_size)):
            idx = eligible[s : s + int(batch_size)]
            x = torch.as_tensor(build_sequences(representation, valid_anchor_mask, int(sequence_length), idx), dtype=torch.float32, device=device)
            out = model(x).detach().cpu().numpy().astype(np.float32)
            preds[idx] = out.reshape(-1, len(TARGET_COMPONENTS), len(QUANTILES))
    return preds


def denormalize_sort_and_clamp_quantiles(predictions: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """De-normalize, sort per component, and clamp MAE risks nonnegative."""
    q = np.asarray(predictions, dtype=np.float32) * np.asarray(scales, dtype=np.float32).reshape(1, -1, 1)
    q = np.sort(q, axis=2)
    q[:, 1:, :] = np.maximum(q[:, 1:, :], 0.0)
    return q.astype(np.float32)


def round_trip_unlevered_cost(cfg: Config) -> float:
    return 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate))


def signed_utility_scores(quantiles: np.ndarray, cfg: Config) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build signed long/short utility streams from de-normalized path quantiles."""
    q = np.asarray(quantiles, dtype=np.float32)
    cost = round_trip_unlevered_cost(cfg)
    out: dict[str, np.ndarray] = {}
    for name, mode, lam in UTILITY_SPECS:
        lam = float(lam)
        if mode == "median":
            long_utility = q[:, 0, 1] - cost - lam * q[:, 1, 1]
            short_utility = -q[:, 0, 1] - cost - lam * q[:, 2, 1]
        elif mode == "conservative":
            long_utility = q[:, 0, 0] - cost - lam * q[:, 1, 2]
            short_utility = -q[:, 0, 2] - cost - lam * q[:, 2, 2]
        else:  # pragma: no cover
            raise ValueError(mode)
        scores = np.zeros(len(q), dtype=np.float32)
        finite = np.isfinite(long_utility) & np.isfinite(short_utility)
        long_take = finite & (long_utility > 0.0) & (long_utility >= short_utility)
        short_take = finite & (short_utility > 0.0) & (short_utility > long_utility)
        scores[~finite] = np.nan
        scores[long_take] = long_utility[long_take].astype(np.float32)
        scores[short_take] = -short_utility[short_take].astype(np.float32)
        out[name] = scores
    return out, {"round_trip_unlevered_cost": cost, "cost_formula": "2*(cfg.fee_rate+cfg.slippage_rate)", "expected_default": 0.0012}


def policy_masks(scores: np.ndarray, positions: np.ndarray, market_size: int, rolling_window: int, quantile: float, side_policy: str) -> tuple[np.ndarray, np.ndarray]:
    low, high = causal_rolling_thresholds(scores, window=rolling_window, quantile=quantile, min_periods=MIN_SCORE_HISTORY)
    return signed_dynamic_policy_masks(scores, positions, market_size, side_policy=side_policy, low_thresholds=low, high_thresholds=high)


def selection_window_signal_hash(long_active: np.ndarray, short_active: np.ndarray, *, market: pd.DataFrame, dates: pd.Series) -> str:
    return effective_selection_signal_hash(market, dates, long_active, short_active, window=WINDOWS["holdout2023"])


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


def run(args: argparse.Namespace) -> dict[str, Any]:
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

    # Phase 1: no 2024+ feature extraction, target materialization, inference, or diagnostics.
    rep, valid_mask, pca_metadata, embedding_metadata, model_metadata = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=True)
    phase1_models: dict[str, TinyMamba2PathCritic] = {}
    phase1_diagnostics: dict[str, Any] = {}
    phase1_scores: dict[str, np.ndarray] = {}
    raw_candidates: list[dict[str, Any]] = []
    fit_mask = fit_admissible_mask(dates, positions, phase1_targets, valid_mask, SEQUENCE_LENGTH)
    start_index = first_post_fit_index(dates, positions, fit_mask)
    stop_index = int(np.searchsorted(signal_dates.to_numpy(), np.datetime64(pd.Timestamp("2024-01-01")), side="left"))
    scales = fit_component_scales(phase1_targets, fit_mask)
    for objective in OBJECTIVES:
        spec = PathCriticSpec(SEQUENCE_LENGTH, objective)  # type: ignore[arg-type]
        print(f"training {spec.stream_prefix}", file=sys.stderr, flush=True)
        model, diag = train_path_critic_model(rep, valid_mask, phase1_targets, fit_mask, signal_dates, spec, scales)
        phase1_models[objective] = model
        raw_pred = predict_path_quantiles(model, rep, valid_mask, SEQUENCE_LENGTH, start_index=start_index, stop_index=stop_index, batch_size=args.predict_batch_size)
        raw_pred[:start_index] = np.nan
        q_pred = denormalize_sort_and_clamp_quantiles(raw_pred, scales)
        streams, utility_meta = signed_utility_scores(q_pred, cfg)
        for utility_name, scores in streams.items():
            stream_id = f"{spec.stream_prefix}_{utility_name}"
            scores[:start_index] = np.nan
            phase1_scores[stream_id] = scores
            phase1_diagnostics[stream_id] = {
                "training": diag,
                "sequence_length": SEQUENCE_LENGTH,
                "objective": objective,
                "utility": utility_name,
                "target_component_scales_no_mean": scales.tolist(),
                "utility_cost": utility_meta,
                "first_policy_score_index": start_index,
                "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index]).all()),
                "score_quality_prefreeze": _score_quality(scores, phase1_targets[:, 0], {k: masks[k] for k in ("fit2020_2022", "holdout2023")}),
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
                            "sequence_length": SEQUENCE_LENGTH,
                            "objective": objective,
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
        "state_space_model": path_critic_transformers_metadata(),
        "input": {"frequency": "1h completed candles", "context_hours": args.context_hours, "embedding_variates": list(EMBEDDING_VARIATES), "anchor_stride_hours": 6, "sequence_lengths": list(SEQUENCE_LENGTHS), **embedding_metadata},
        "representation": pca_metadata,
        "target": {**target_meta, "normalization": "divide each component by fit-only std; no mean subtraction; return clip +/-3std; MAE clip 0..3std", "quantiles": list(QUANTILES)},
        "strict_protocol": {"phase1": "extract/embed and infer only signal dates <2024; generate only labels whose exit <2024; train only admissible 2020-2022 rows; policy scores remain NaN through fit prefix", "selection": "2023 metrics only; positive return and >=8 trades; actual executable path hash de-dupe", "phase2": "after manifest write, extract full anchors only for selected objectives, infer, assert frozen 2023 executable hashes unchanged, then compute OOS strict metrics/diagnostics/cost stress"},
        "utility": {"specs": [name for name, _mode, _lam in UTILITY_SPECS], "cost": {"round_trip_unlevered": round_trip_unlevered_cost(cfg), "formula": "2*(cfg.fee_rate+cfg.slippage_rate)", "expected_default": 0.0012}},
        "top10": selected,
        "trial_counts": {"sequence_lengths": len(SEQUENCE_LENGTHS), "objectives": len(OBJECTIVES), "utility_scores": len(UTILITY_SPECS), "score_streams": len(phase1_scores), "rolling_windows": len(ROLLING_WINDOWS), "score_quantiles": len(SCORE_QUANTILES), "side_policies": len(SIDES), "total_policy_specs": len(phase1_scores) * len(ROLLING_WINDOWS) * len(SCORE_QUANTILES) * len(SIDES), "eligible_holdout_candidates": len(raw_candidates), "distinct_top10": len(selected)},
        "data_sha256": data_hashes,
    }
    manifest_path = Path(args.manifest_output)
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    phase2_diagnostics: dict[str, Any] = {}
    phase2_scores: dict[str, np.ndarray] = {}
    unique_objectives = {str(row["objective"]) for row in selected}
    if unique_objectives:
        targets_full, _ = executable_path_targets_48h(market, positions)
        rep_full, valid_full, pca_full_metadata, _, _ = extract_pca32_for_phase(args, market, dates, positions, masks, source_manifest, before_2024_only=False)
        assert_pca_hashes_match_source(pca_full_metadata, source_manifest, dimensions=PCA_DIMS)
    else:
        targets_full, rep_full, valid_full = phase1_targets, rep, valid_mask
    for objective in sorted(unique_objectives):
        fit_mask_full = fit_admissible_mask(dates, positions, targets_full, valid_full, SEQUENCE_LENGTH)
        start_index_full = first_post_fit_index(dates, positions, fit_mask_full)
        full_scales = fit_component_scales(targets_full, fit_mask_full)
        if not np.allclose(full_scales, scales, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "fit-only path target scales changed after future target materialization"
            )
        model = phase1_models[objective]
        raw_pred = predict_path_quantiles(model, rep_full, valid_full, SEQUENCE_LENGTH, start_index=start_index_full, stop_index=None, batch_size=args.predict_batch_size)
        raw_pred[:start_index_full] = np.nan
        q_pred = denormalize_sort_and_clamp_quantiles(raw_pred, full_scales)
        streams, utility_meta = signed_utility_scores(q_pred, cfg)
        for utility_name, scores in streams.items():
            stream_id = f"pathcritic_mamba2_seq{SEQUENCE_LENGTH}_{objective}_{utility_name}"
            scores[:start_index_full] = np.nan
            phase2_scores[stream_id] = scores
            phase2_diagnostics[stream_id] = {"score_quality": _score_quality(scores, targets_full[:, 0], masks), "utility_cost": utility_meta, "first_policy_score_index": start_index_full, "policy_scores_fit_prefix_nan": bool(np.isnan(scores[:start_index_full]).all())}

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

    output = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "protocol": "strict MOMENT PCA32 tiny Mamba2 distributional path critic; fit-only 2020-2022; 2023 Top-10 frozen before 2024+ features/metrics; next-bar entry; hold576; 6bp/side; full-window CAGR; strict intratrade MDD", "manifest": str(manifest_path), "source_manifest": str(Path(args.source_manifest).resolve()), "model": model_metadata, "state_space_model": path_critic_transformers_metadata(), "input": manifest["input"], "representation": pca_metadata, "phase1_model_diagnostics": phase1_diagnostics, "phase2_model_diagnostics": phase2_diagnostics, "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()}, "tested_candidates": len(raw_candidates), "selected": selected, "alpha_pool_qualifiers": alpha_pool, "live_grade": live_grade, "cost_stress_bps_per_side": cost_stress}
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
