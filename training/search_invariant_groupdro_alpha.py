"""Cross-environment invariant tail-classification alpha search.

Feature directions are admitted only when their Spearman sign agrees in each
of 2020, 2021, and 2022.  Low-capacity linear/MLP classifiers are then trained
with ERM, V-REx, or Group-DRO across six half-year environments.  The model
predicts short/flat/long 48-hour return tails.  2023 ranks and freezes a
distinct Top-10 before 2024+ strict trading metrics are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim
from training.search_river_contextual_utility_alpha import (
    effective_selection_signal_hash,
)
from training.search_river_online_alpha import validate_output_paths
from training.search_tabicl_foundation_alpha import (
    ANCHOR_STRIDE,
    HOLD_BARS,
    WINDOWS,
    _file_sha256,
    _git_head,
    anchor_dataset,
    feature_groups,
    policy_masks,
    split_mask_for_anchors,
    top10_promotions,
)


TAIL_QUANTILES = (0.30, 0.70)
FEATURE_SET_SIZES = (8, 16, 24)
SCORE_QUANTILES = (0.70, 0.80, 0.90, 0.95)
SEED = 713

MODEL_SPECS = (
    ("linear_erm", "linear", "erm", 0.0),
    ("linear_vrex1", "linear", "vrex", 1.0),
    ("linear_vrex10", "linear", "vrex", 10.0),
    ("linear_groupdro", "linear", "groupdro", 0.0),
    ("mlp_erm", "mlp", "erm", 0.0),
    ("mlp_vrex1", "mlp", "vrex", 1.0),
    ("mlp_vrex10", "mlp", "vrex", 10.0),
    ("mlp_groupdro", "mlp", "groupdro", 0.0),
)


def _finite_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3 or np.nanstd(x[finite]) < 1e-12:
        return 0.0
    value = spearmanr(x[finite], y[finite]).statistic
    return float(value) if np.isfinite(value) else 0.0


def stable_feature_ranking(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    targets: np.ndarray,
    signal_dates: pd.Series,
    fit_mask: np.ndarray,
    *,
    years: Sequence[int] = (2020, 2021, 2022),
    correlation_cap: float = 0.95,
    minimum_samples_per_year: int = 100,
) -> list[dict[str, Any]]:
    """Rank sign-consistent train-only features and remove near duplicates."""
    matrix = np.asarray(matrix, dtype=float)
    targets = np.asarray(targets, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    date_years = pd.to_datetime(signal_dates).dt.year.to_numpy()
    raw: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        correlations: list[float] = []
        valid = True
        for year in years:
            mask = fit_mask & (date_years == int(year))
            finite = mask & np.isfinite(matrix[:, index]) & np.isfinite(targets)
            if int(finite.sum()) < int(minimum_samples_per_year):
                valid = False
                break
            correlations.append(_finite_spearman(matrix[finite, index], targets[finite]))
        if not valid or any(abs(value) < 1e-12 for value in correlations):
            continue
        signs = np.sign(correlations)
        if not np.all(signs == signs[0]):
            continue
        raw.append(
            {
                "index": index,
                "name": str(name),
                "direction": int(signs[0]),
                "year_spearman": {
                    str(year): float(value)
                    for year, value in zip(years, correlations, strict=True)
                },
                "minimum_abs_spearman": float(min(abs(value) for value in correlations)),
                "mean_abs_spearman": float(np.mean(np.abs(correlations))),
            }
        )
    raw.sort(
        key=lambda row: (
            row["minimum_abs_spearman"],
            row["mean_abs_spearman"],
            row["name"],
        ),
        reverse=True,
    )

    fit_rows = matrix[fit_mask]
    medians = np.nanmedian(fit_rows, axis=0)
    clean_fit = np.where(np.isfinite(fit_rows), fit_rows, medians)
    selected: list[dict[str, Any]] = []
    for candidate in raw:
        candidate_values = clean_fit[:, candidate["index"]]
        duplicate_of: str | None = None
        for existing in selected:
            existing_values = clean_fit[:, existing["index"]]
            correlation = np.corrcoef(candidate_values, existing_values)[0, 1]
            if np.isfinite(correlation) and abs(float(correlation)) >= float(
                correlation_cap
            ):
                duplicate_of = str(existing["name"])
                break
        if duplicate_of is None:
            selected.append(candidate)
    return selected


def feature_set_definitions(
    ranking: list[dict[str, Any]],
    sizes: Sequence[int] = FEATURE_SET_SIZES,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    seen_lengths: set[int] = set()
    for requested_size in sizes:
        actual_size = min(int(requested_size), len(ranking))
        if actual_size <= 0 or actual_size in seen_lengths:
            continue
        seen_lengths.add(actual_size)
        output[f"stable{actual_size}"] = ranking[:actual_size]
    return output


def tail_labels(
    targets: np.ndarray,
    fit_mask: np.ndarray,
    *,
    low_quantile: float = TAIL_QUANTILES[0],
    high_quantile: float = TAIL_QUANTILES[1],
) -> tuple[np.ndarray, tuple[float, float]]:
    fit_values = np.asarray(targets, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    low, high = np.quantile(fit_values[np.isfinite(fit_values)], (low_quantile, high_quantile))
    labels = np.ones(len(targets), dtype=np.int64)
    labels[np.asarray(targets) <= low] = 0
    labels[np.asarray(targets) >= high] = 2
    return labels, (float(low), float(high))


def half_year_environments(signal_dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(signal_dates)
    keys = dates.dt.year.to_numpy() * 2 + (dates.dt.month.to_numpy() > 6).astype(int)
    _, environments = np.unique(keys, return_inverse=True)
    return environments.astype(np.int64)


def environment_risk_objective(
    environment_losses: torch.Tensor,
    *,
    objective: str,
    vrex_penalty: float = 0.0,
    group_weights: torch.Tensor | None = None,
    groupdro_eta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if objective == "erm":
        return environment_losses.mean(), group_weights
    if objective == "vrex":
        loss = environment_losses.mean() + float(vrex_penalty) * environment_losses.var(
            unbiased=False
        )
        return loss, group_weights
    if objective == "groupdro":
        if group_weights is None:
            group_weights = torch.ones_like(environment_losses) / len(environment_losses)
        with torch.no_grad():
            group_weights = group_weights * torch.exp(
                float(groupdro_eta) * environment_losses.detach()
            )
            group_weights = group_weights / group_weights.sum()
        return torch.sum(group_weights * environment_losses), group_weights
    raise ValueError(f"unsupported objective: {objective}")


class TailClassifier(torch.nn.Module):
    def __init__(self, input_size: int, architecture: str):
        super().__init__()
        if architecture == "linear":
            self.network = torch.nn.Linear(input_size, 3)
        elif architecture == "mlp":
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, 32),
                torch.nn.GELU(),
                torch.nn.LayerNorm(32),
                torch.nn.Linear(32, 3),
            )
        else:
            raise ValueError(f"unsupported architecture: {architecture}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _standardize(
    matrix: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_rows = np.asarray(matrix, dtype=float)[np.asarray(fit_mask, dtype=bool)]
    median = np.nanmedian(fit_rows, axis=0)
    scale = np.nanstd(fit_rows, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    clean = np.where(np.isfinite(matrix), matrix, median)
    standardized = np.clip((clean - median) / scale, -10.0, 10.0)
    return standardized.astype(np.float32), median, scale


def train_tail_classifier(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    environments: np.ndarray,
    *,
    architecture: str,
    objective: str,
    vrex_penalty: float,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standardized, median, scale = _standardize(matrix, fit_mask)
    fit_indices = np.flatnonzero(fit_mask)
    x_train = torch.from_numpy(standardized[fit_indices]).to(device)
    y_train = torch.from_numpy(np.asarray(labels, dtype=np.int64)[fit_indices]).to(device)
    env_train = torch.from_numpy(np.asarray(environments, dtype=np.int64)[fit_indices]).to(
        device
    )
    unique_environments = torch.unique(env_train, sorted=True)
    class_counts = torch.bincount(y_train, minlength=3).float().clamp_min(1.0)
    class_weights = (class_counts.sum() / class_counts)
    class_weights = class_weights / class_weights.mean()

    model = TailClassifier(matrix.shape[1], architecture).to(device)
    learning_rate = 0.01 if architecture == "linear" else 0.003
    epochs = 400 if architecture == "linear" else 300
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    group_weights: torch.Tensor | None = None
    final_environment_losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        environment_losses = torch.stack(
            [
                F.cross_entropy(
                    logits[env_train == environment],
                    y_train[env_train == environment],
                    weight=class_weights,
                )
                for environment in unique_environments
            ]
        )
        loss, group_weights = environment_risk_objective(
            environment_losses,
            objective=objective,
            vrex_penalty=vrex_penalty,
            group_weights=group_weights,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_environment_losses = environment_losses.detach().cpu().tolist()

    model.eval()
    with torch.inference_mode():
        probabilities = []
        batch_size = 4096
        for start in range(0, len(standardized), batch_size):
            batch = torch.from_numpy(standardized[start : start + batch_size]).to(device)
            probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
        probabilities_array = np.concatenate(probabilities, axis=0)
        train_prediction = probabilities_array[fit_indices].argmax(axis=1)
    state_bytes = b"".join(
        tensor.detach().cpu().numpy().tobytes()
        for tensor in model.state_dict().values()
    )
    diagnostics = {
        "device": str(device),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "class_counts": class_counts.detach().cpu().int().tolist(),
        "class_weights": class_weights.detach().cpu().tolist(),
        "environment_losses": final_environment_losses,
        "final_loss_mean": float(np.mean(final_environment_losses)),
        "final_loss_variance": float(np.var(final_environment_losses)),
        "group_weights": (
            group_weights.detach().cpu().tolist() if group_weights is not None else None
        ),
        "train_accuracy": float(
            np.mean(train_prediction == np.asarray(labels)[fit_indices])
        ),
        "scaler_median": median.tolist(),
        "scaler_scale": scale.tolist(),
        "state_dict_sha256": hashlib.sha256(state_bytes).hexdigest(),
    }
    return probabilities_array[:, 2] - probabilities_array[:, 0], diagnostics


def _score_quality(
    scores: np.ndarray,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for name, mask in masks.items():
        finite = mask & np.isfinite(scores) & np.isfinite(targets)
        output[name] = {
            "samples": int(finite.sum()),
            "spearman": (
                _finite_spearman(scores[finite], targets[finite])
                if int(finite.sum()) >= 3
                else None
            ),
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_output_paths(args.output, args.manifest_output)
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
    if not feature_sets:
        raise RuntimeError("no sign-stable train-only feature set found")
    labels, label_thresholds = tail_labels(targets, fit_mask)
    environments = half_year_environments(signal_dates)

    raw_candidates: list[dict[str, Any]] = []
    model_diagnostics: dict[str, Any] = {}
    for feature_set_name, feature_rows in feature_sets.items():
        indices = [int(row["index"]) for row in feature_rows]
        matrix = full_matrix[:, indices]
        for model_id, architecture, objective, vrex_penalty in MODEL_SPECS:
            full_model_id = f"{model_id}_{feature_set_name}"
            print(
                f"training {full_model_id} ({len(indices)} features)",
                file=sys.stderr,
                flush=True,
            )
            scores, diagnostics = train_tail_classifier(
                matrix,
                labels,
                fit_mask,
                environments,
                architecture=architecture,
                objective=objective,
                vrex_penalty=vrex_penalty,
            )
            model_diagnostics[full_model_id] = {
                "architecture": architecture,
                "objective": objective,
                "vrex_penalty": vrex_penalty,
                "feature_set": feature_set_name,
                "feature_names": [row["name"] for row in feature_rows],
                "training": diagnostics,
                "score_quality": _score_quality(scores, targets, masks),
            }
            holdout_scores = scores[masks["holdout2023"]]
            for quantile in SCORE_QUANTILES:
                low_threshold, high_threshold = np.quantile(
                    holdout_scores[np.isfinite(holdout_scores)],
                    (1.0 - quantile, quantile),
                )
                for side_policy in ("long", "short", "both"):
                    long_active, short_active = policy_masks(
                        scores,
                        positions,
                        len(market),
                        side_policy=side_policy,
                        low_threshold=(
                            float(low_threshold)
                            if side_policy in {"short", "both"}
                            else None
                        ),
                        high_threshold=(
                            float(high_threshold)
                            if side_policy in {"long", "both"}
                            else None
                        ),
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
                            "model_id": full_model_id,
                            "architecture": architecture,
                            "objective": objective,
                            "vrex_penalty": vrex_penalty,
                            "feature_set": feature_set_name,
                            "feature_names": [row["name"] for row in feature_rows],
                            "score_quantile": quantile,
                            "score_thresholds": [
                                float(low_threshold),
                                float(high_threshold),
                            ],
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
        long_active = candidate.pop("_long")
        short_active = candidate.pop("_short")
        selected.append(candidate)
        selected_signals[signal_hash] = (long_active, short_active)
        if len(selected) == 10:
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "selection_policy": (
            "train-only sign-stable features; fixed ERM/V-REx/Group-DRO algorithms; "
            "2023 distinct executed-path Top-10 frozen before 2024+ metrics"
        ),
        "target": {
            "name": "three_class_next_48h_return_tail",
            "fit_quantiles": list(TAIL_QUANTILES),
            "fit_thresholds": list(label_thresholds),
            "classes": ["short_tail", "flat_middle", "long_tail"],
        },
        "environment_definition": "six half-years in fit 2020-2022",
        "feature_ranking": ranking,
        "feature_sets": {
            name: [row["name"] for row in rows]
            for name, rows in feature_sets.items()
        },
        "top10": selected,
        "trial_counts": {
            "model_specs": len(MODEL_SPECS),
            "feature_sets": len(feature_sets),
            "policy_specs_per_model": len(SCORE_QUANTILES) * 3,
            "total_policy_specs": len(MODEL_SPECS)
            * len(feature_sets)
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
        "protocol": (
            "train-only sign-stable feature selection; 3-class 48h tail models trained "
            "across six half-year environments with ERM/V-REx/Group-DRO; 2023 score "
            "threshold and executed-path Top-10 freeze; 2024/2025/2026 OOS; next-bar "
            "entry; 6bp/side; full-window CAGR; strict intratrade MDD"
        ),
        "manifest": str(manifest_path),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "target": manifest["target"],
        "feature_ranking": ranking,
        "feature_sets": manifest["feature_sets"],
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
