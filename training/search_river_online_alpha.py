# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
#   "pandas==2.3.3",
#   "river==0.25.0",
#   "scipy==1.18.0",
# ]
# ///
"""Leak-safe prequential alpha search with River and delayed labels.

Every 6-hour anchor is predicted before its own example is queued for learning.
An example becomes learnable only when its next-bar-entry, 48-hour exit open is
observable.  Trading thresholds are rolling quantiles of *prior* predictions.
The 2023 holdout ranks a frozen Top-10 before any 2024+ trading metric is run.

Run this Python>=3.11-only experiment without changing the project environment:

    uv run --isolated --no-project --python 3.12 --script \
      training/search_river_online_alpha.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim
from training.search_tabicl_foundation_alpha import (
    ANCHOR_STRIDE,
    HOLD_BARS,
    WINDOWS,
    _file_sha256,
    _git_head,
    _signal_hash,
    anchor_dataset,
    feature_groups,
    split_mask_for_anchors,
    top10_promotions,
)


ROLLING_WINDOWS = (720, 1460)  # roughly 180 and 365 days at 6-hour anchors
SCORE_QUANTILES = (0.70, 0.80, 0.85, 0.90, 0.95)
MIN_MODEL_UPDATES = 300
MIN_SCORE_HISTORY = 200
TARGET_CLIP = 0.20
SEED = 713

MODEL_SPECS = (
    ("arf_compact", "arf", "compact"),
    ("arf_price", "arf", "price"),
    ("arf_full", "arf", "full"),
    ("hatr_compact", "hatr", "compact"),
    ("hatr_full", "hatr", "full"),
    ("linear_compact", "linear", "compact"),
    ("linear_full", "linear", "full"),
)


class OnlineRegressor(Protocol):
    def predict_one(self, x: dict[str, float]) -> float | None: ...

    def learn_one(self, x: dict[str, float], y: float) -> Any: ...


def label_ready_positions(
    signal_positions: np.ndarray,
    *,
    entry_delay_bars: int = 1,
    hold_bars: int = HOLD_BARS,
) -> np.ndarray:
    """Return the first bar position at which each forward label is observable."""
    return np.asarray(signal_positions, dtype=np.int64) + int(entry_delay_bars) + int(
        hold_bars
    )


def _row_as_dict(feature_names: Sequence[str], row: np.ndarray) -> dict[str, float]:
    values = np.asarray(row, dtype=float)
    return {
        name: float(value) if np.isfinite(value) else 0.0
        for name, value in zip(feature_names, values, strict=True)
    }


def delayed_online_predictions(
    model: OnlineRegressor,
    *,
    feature_names: Sequence[str],
    matrix: np.ndarray,
    targets: np.ndarray,
    signal_positions: np.ndarray,
    ready_positions: np.ndarray,
    min_completed_updates: int = MIN_MODEL_UPDATES,
    target_clip: float = TARGET_CLIP,
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    """Run predict-before-learn streaming with labels released at their exit open.

    Pending examples are ordered because both signal and ready positions are
    monotonic.  At a new anchor, all labels whose exit open is already known are
    learned first.  The current prediction is then produced, and only afterwards
    is the current example placed in the pending queue.
    """
    matrix = np.asarray(matrix, dtype=float)
    targets = np.asarray(targets, dtype=float)
    signal_positions = np.asarray(signal_positions, dtype=np.int64)
    ready_positions = np.asarray(ready_positions, dtype=np.int64)
    n = len(signal_positions)
    if matrix.shape[0] != n or len(targets) != n or len(ready_positions) != n:
        raise ValueError("stream arrays must have identical row counts")
    if np.any(np.diff(signal_positions) <= 0) or np.any(np.diff(ready_positions) <= 0):
        raise ValueError("signal and label-ready positions must be strictly increasing")
    if np.any(ready_positions <= signal_positions):
        raise ValueError("labels must become observable after their signal")

    predictions = np.full(n, np.nan, dtype=float)
    pending: deque[tuple[int, dict[str, float], float]] = deque()
    completed_updates = 0
    first_prediction_position: int | None = None
    last_learned_ready_position: int | None = None

    for index, signal_pos in enumerate(signal_positions):
        while pending and pending[0][0] <= int(signal_pos):
            ready_pos, old_x, old_y = pending.popleft()
            model.learn_one(old_x, old_y)
            completed_updates += 1
            last_learned_ready_position = ready_pos

        current_x = _row_as_dict(feature_names, matrix[index])
        if completed_updates >= int(min_completed_updates):
            prediction = model.predict_one(current_x)
            if prediction is not None and np.isfinite(float(prediction)):
                predictions[index] = float(prediction)
                if first_prediction_position is None:
                    first_prediction_position = int(signal_pos)

        clipped_target = float(np.clip(targets[index], -target_clip, target_clip))
        pending.append((int(ready_positions[index]), current_x, clipped_target))

    return predictions, {
        "scheduled_samples": n,
        "completed_updates_before_last_prediction": completed_updates,
        "pending_samples_after_last_prediction": len(pending),
        "first_prediction_position": first_prediction_position,
        "last_learned_ready_position": last_learned_ready_position,
        "target_clip_abs": float(target_clip),
    }


def causal_rolling_thresholds(
    scores: np.ndarray,
    *,
    window: int,
    quantile: float,
    min_periods: int = MIN_SCORE_HISTORY,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute low/high thresholds from prior predictions only."""
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0)")
    history = pd.Series(np.asarray(scores, dtype=float)).shift(1)
    rolling = history.rolling(int(window), min_periods=int(min_periods))
    low = rolling.quantile(1.0 - float(quantile)).to_numpy(float)
    high = rolling.quantile(float(quantile)).to_numpy(float)
    return low, high


def dynamic_policy_masks(
    scores: np.ndarray,
    positions: np.ndarray,
    size: int,
    *,
    side_policy: str,
    low_thresholds: np.ndarray,
    high_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map causal per-anchor score thresholds to executable signal bars."""
    scores = np.asarray(scores, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    low_thresholds = np.asarray(low_thresholds, dtype=float)
    high_thresholds = np.asarray(high_thresholds, dtype=float)
    if not (
        len(scores)
        == len(positions)
        == len(low_thresholds)
        == len(high_thresholds)
    ):
        raise ValueError("score, position, and threshold lengths must match")
    if side_policy not in {"long", "short", "both"}:
        raise ValueError(f"unsupported side policy: {side_policy}")

    long_active = np.zeros(int(size), dtype=bool)
    short_active = np.zeros(int(size), dtype=bool)
    if side_policy in {"long", "both"}:
        active = np.isfinite(scores) & np.isfinite(high_thresholds) & (
            scores >= high_thresholds
        )
        long_active[positions[active]] = True
    if side_policy in {"short", "both"}:
        active = np.isfinite(scores) & np.isfinite(low_thresholds) & (
            scores <= low_thresholds
        )
        short_active[positions[active]] = True
    return long_active, short_active


def selection_window_signal_hash(
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    positions: np.ndarray,
    selection_mask: np.ndarray,
) -> str:
    """Hash only selection-window anchor decisions for pre-OOS de-duplication."""
    positions = np.asarray(positions, dtype=np.int64)
    selection_mask = np.asarray(selection_mask, dtype=bool)
    if len(positions) != len(selection_mask):
        raise ValueError("positions and selection mask must have identical lengths")
    selected_positions = positions[selection_mask]
    return _signal_hash(
        np.asarray(long_active, dtype=bool)[selected_positions],
        np.asarray(short_active, dtype=bool)[selected_positions],
    )


def validate_output_paths(output: str, manifest_output: str) -> None:
    if Path(output).resolve() == Path(manifest_output).resolve():
        raise ValueError("output and frozen manifest paths must be different")


def _make_model(model_kind: str) -> OnlineRegressor:
    from river import drift, forest, linear_model, optim, preprocessing, tree

    if model_kind == "arf":
        return forest.ARFRegressor(
            n_models=10,
            max_features="sqrt",
            drift_detector=drift.ADWIN(delta=0.001),
            warning_detector=drift.ADWIN(delta=0.01),
            grace_period=50,
            seed=SEED,
        )
    if model_kind == "hatr":
        return tree.HoeffdingAdaptiveTreeRegressor(
            grace_period=100,
            drift_detector=drift.ADWIN(delta=0.002),
            seed=SEED,
        )
    if model_kind == "linear":
        return preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(0.01),
            l2=0.001,
        )
    raise KeyError(model_kind)


def _model_change_counts(model: OnlineRegressor) -> dict[str, int | None]:
    drift_count = getattr(model, "n_drifts_detected", None)
    warning_count = getattr(model, "n_warnings_detected", None)
    return {
        "drifts_detected": int(drift_count()) if callable(drift_count) else None,
        "warnings_detected": int(warning_count()) if callable(warning_count) else None,
    }


def _score_diagnostics(
    scores: np.ndarray,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, int | float | None]]:
    from scipy.stats import spearmanr

    diagnostics: dict[str, dict[str, int | float | None]] = {}
    for split, split_mask in masks.items():
        finite = split_mask & np.isfinite(scores) & np.isfinite(targets)
        count = int(finite.sum())
        if count < 3:
            diagnostics[split] = {
                "samples": count,
                "spearman": None,
                "direction_accuracy": None,
            }
            continue
        correlation = spearmanr(scores[finite], targets[finite]).statistic
        diagnostics[split] = {
            "samples": count,
            "spearman": float(correlation) if np.isfinite(correlation) else None,
            "direction_accuracy": float(
                np.mean(np.sign(scores[finite]) == np.sign(targets[finite]))
            ),
        }
    return diagnostics


def _select_distinct_top10(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    candidates.sort(
        key=lambda row: (
            row["holdout2023"]["ratio"],
            row["holdout2023"]["return_pct"],
            row["holdout2023"]["trades"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in candidates:
        signal_hash = candidate["signal_hash"]
        if signal_hash in selected_signals:
            continue
        long_active = candidate.pop("_long")
        short_active = candidate.pop("_short")
        selected.append(candidate)
        selected_signals[signal_hash] = (long_active, short_active)
        if len(selected) == 10:
            break
    return selected, selected_signals


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
    groups = feature_groups(features)
    positions, targets, _ = anchor_dataset(market, features)
    ready_positions = label_ready_positions(positions)
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    anchor_matrices = {
        group_name: features.iloc[positions][columns].to_numpy(float)
        for group_name, columns in groups.items()
    }

    raw_candidates: list[dict[str, Any]] = []
    model_diagnostics: dict[str, Any] = {}
    for model_index, (model_id, model_kind, group_name) in enumerate(MODEL_SPECS, start=1):
        print(
            f"[{model_index}/{len(MODEL_SPECS)}] streaming {model_id} "
            f"({len(groups[group_name])} features)",
            file=sys.stderr,
            flush=True,
        )
        model = _make_model(model_kind)
        scores, stream_diagnostics = delayed_online_predictions(
            model,
            feature_names=groups[group_name],
            matrix=anchor_matrices[group_name],
            targets=targets,
            signal_positions=positions,
            ready_positions=ready_positions,
        )
        model_diagnostics[model_id] = {
            "model_kind": model_kind,
            "feature_group": group_name,
            "feature_count": len(groups[group_name]),
            "feature_names": groups[group_name],
            "stream": stream_diagnostics,
            "concept_change": _model_change_counts(model),
            "score_quality": _score_diagnostics(scores, targets, masks),
        }

        for rolling_window in ROLLING_WINDOWS:
            for quantile in SCORE_QUANTILES:
                low_thresholds, high_thresholds = causal_rolling_thresholds(
                    scores,
                    window=rolling_window,
                    quantile=quantile,
                )
                for side_policy in ("long", "short", "both"):
                    long_active, short_active = dynamic_policy_masks(
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
                            "model_id": model_id,
                            "model_kind": model_kind,
                            "feature_group": group_name,
                            "rolling_score_window_anchors": rolling_window,
                            "score_quantile": quantile,
                            "minimum_score_history": MIN_SCORE_HISTORY,
                            "side_policy": side_policy,
                            "hold_bars": HOLD_BARS,
                            "anchor_stride_bars": ANCHOR_STRIDE,
                            "holdout2023": holdout,
                            "signal_hash": selection_window_signal_hash(
                                long_active,
                                short_active,
                                positions=positions,
                                selection_mask=masks["holdout2023"],
                            ),
                            "_long": long_active,
                            "_short": short_active,
                        }
                    )

    selected, selected_signals = _select_distinct_top10(raw_candidates)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "river_version": importlib.metadata.version("river"),
        "selection_policy": (
            "rank distinct profitable candidates on 2023 full-window strict CAGR/MDD; "
            "freeze Top-10 before computing 2024+ trading metrics"
        ),
        "online_protocol": {
            "predict_before_current_example_is_queued": True,
            "label_ready_offset_bars": 1 + HOLD_BARS,
            "learn_only_when_ready_position_lte_current_signal_position": True,
            "rolling_threshold_uses_shifted_prior_scores": True,
            "minimum_completed_updates": MIN_MODEL_UPDATES,
            "minimum_prior_scores_for_threshold": MIN_SCORE_HISTORY,
            "target_clip_abs_log_return": TARGET_CLIP,
        },
        "top10": selected,
        "trial_counts": {
            "model_specs": len(MODEL_SPECS),
            "rolling_windows": len(ROLLING_WINDOWS),
            "score_quantiles": len(SCORE_QUANTILES),
            "side_policies": 3,
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
        "source": "https://github.com/online-ml/river",
        "river_version": importlib.metadata.version("river"),
        "protocol": (
            "River prequential regressors; causal features at 6h anchors; predict first; "
            "release labels only after next-bar-entry plus 48h exit open; causal rolling "
            "score quantiles; 2023 ranks frozen Top-10; 2024/2025/2026 prequential OOS; "
            "next-bar execution; 6bp/side; full-window CAGR; strict intratrade MDD"
        ),
        "manifest": str(manifest_path),
        "feature_groups": groups,
        "model_diagnostics": model_diagnostics,
        "target": {
            "name": "next_48h_open_to_open_log_return",
            "entry_delay_bars": 1,
            "hold_bars": HOLD_BARS,
            "label_ready_offset_bars": 1 + HOLD_BARS,
            "fixed_clip_abs_log_return": TARGET_CLIP,
        },
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
