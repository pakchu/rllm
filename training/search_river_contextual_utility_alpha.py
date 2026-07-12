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
"""Delayed multi-head contextual action-utility alpha search.

Instead of regressing one noisy future return, this experiment learns four
counterfactual outcomes that become observable after the 48-hour path ends:
long net return, short net return, long MAE loss, and short MAE loss.  A causal
policy chooses the side with greater predicted risk-adjusted utility and may
abstain.  The 2023 selection-window Top-10 is frozen before 2024+ metrics.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim
from training.search_river_online_alpha import (
    MIN_MODEL_UPDATES,
    MIN_SCORE_HISTORY,
    _make_model,
    _model_change_counts,
    _score_diagnostics,
    _select_distinct_top10,
    delayed_online_predictions,
    label_ready_positions,
    validate_output_paths,
)
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


MODEL_SPECS = (
    ("utility_arf_compact", "arf", "compact"),
    ("utility_arf_full", "arf", "full"),
    ("utility_hatr_compact", "hatr", "compact"),
    ("utility_hatr_full", "hatr", "full"),
)
UTILITY_PENALTIES = (0.25, 0.50, 1.00)
ROLLING_WINDOWS = (720, 1460)
GATE_QUANTILES = (0.70, 0.80, 0.90, 0.95)
SIDE_POLICIES = ("both", "long", "short")
TARGET_CLIP = 0.20


def executable_path_targets(
    market: pd.DataFrame,
    positions: np.ndarray,
    cfg: Config,
    *,
    hold_bars: int = HOLD_BARS,
    entry_delay_bars: int = 1,
) -> dict[str, np.ndarray]:
    """Build exact delayed-entry net-return and adverse-excursion components."""
    positions = np.asarray(positions, dtype=np.int64)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    long_net = np.full(len(positions), np.nan, dtype=float)
    short_net = np.full(len(positions), np.nan, dtype=float)
    long_mae_loss = np.full(len(positions), np.nan, dtype=float)
    short_mae_loss = np.full(len(positions), np.nan, dtype=float)
    leverage = float(cfg.leverage)
    side_cost = (float(cfg.fee_rate) + float(cfg.slippage_rate)) * leverage

    for index, signal_pos in enumerate(positions):
        entry_pos = int(signal_pos) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        entry = opens[entry_pos]
        exit_price = opens[exit_pos]
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0.0:
            continue
        held_high = np.nanmax(highs[entry_pos:exit_pos])
        held_low = np.nanmin(lows[entry_pos:exit_pos])
        long_gross = exit_price / entry - 1.0
        short_gross = 1.0 - exit_price / entry
        long_net[index] = (1.0 - side_cost) ** 2 * (
            1.0 + leverage * long_gross
        ) - 1.0
        short_net[index] = (1.0 - side_cost) ** 2 * (
            1.0 + leverage * short_gross
        ) - 1.0
        long_mae_loss[index] = leverage * max(0.0, 1.0 - held_low / entry)
        short_mae_loss[index] = leverage * max(0.0, held_high / entry - 1.0)

    return {
        "long_net": long_net,
        "short_net": short_net,
        "long_mae_loss": long_mae_loss,
        "short_mae_loss": short_mae_loss,
    }


def contextual_utility_scores(
    component_predictions: dict[str, np.ndarray],
    *,
    mae_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_score = np.asarray(component_predictions["long_net"], dtype=float) - float(
        mae_penalty
    ) * np.maximum(
        0.0, np.asarray(component_predictions["long_mae_loss"], dtype=float)
    )
    short_score = np.asarray(component_predictions["short_net"], dtype=float) - float(
        mae_penalty
    ) * np.maximum(
        0.0, np.asarray(component_predictions["short_mae_loss"], dtype=float)
    )
    best_score = np.maximum(long_score, short_score)
    return long_score, short_score, best_score


def causal_gate_thresholds(
    best_scores: np.ndarray,
    *,
    rolling_window: int | None,
    quantile: float | None,
    min_periods: int = MIN_SCORE_HISTORY,
) -> np.ndarray:
    """Return fixed-zero or prior-score rolling thresholds."""
    scores = np.asarray(best_scores, dtype=float)
    if rolling_window is None or quantile is None:
        return np.zeros(len(scores), dtype=float)
    history = pd.Series(scores).shift(1)
    return (
        history.rolling(int(rolling_window), min_periods=int(min_periods))
        .quantile(float(quantile))
        .to_numpy(float)
    )


def utility_policy_masks(
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    thresholds: np.ndarray,
    positions: np.ndarray,
    size: int,
    *,
    side_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the higher-utility side, with flat as a zero-utility action."""
    long_scores = np.asarray(long_scores, dtype=float)
    short_scores = np.asarray(short_scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    positions = np.asarray(positions, dtype=np.int64)
    if not (
        len(long_scores) == len(short_scores) == len(thresholds) == len(positions)
    ):
        raise ValueError("score, threshold, and position lengths must match")
    if side_policy not in SIDE_POLICIES:
        raise ValueError(f"unsupported side policy: {side_policy}")

    if side_policy == "both":
        policy_scores = np.maximum(long_scores, short_scores)
    elif side_policy == "long":
        policy_scores = long_scores
    else:
        policy_scores = short_scores
    active = (
        np.isfinite(long_scores)
        & np.isfinite(short_scores)
        & np.isfinite(thresholds)
        & (policy_scores > 0.0)
        & (policy_scores >= thresholds)
    )
    if side_policy == "both":
        choose_long = active & (long_scores >= short_scores)
        choose_short = active & (short_scores > long_scores)
    elif side_policy == "long":
        choose_long = active
        choose_short = np.zeros(len(active), dtype=bool)
    else:
        choose_long = np.zeros(len(active), dtype=bool)
        choose_short = active

    long_active = np.zeros(int(size), dtype=bool)
    short_active = np.zeros(int(size), dtype=bool)
    long_active[positions[choose_long]] = True
    short_active[positions[choose_short]] = True
    return long_active, short_active


def policy_score_for_side(
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    *,
    side_policy: str,
) -> np.ndarray:
    if side_policy == "both":
        return np.maximum(long_scores, short_scores)
    if side_policy == "long":
        return np.asarray(long_scores, dtype=float)
    if side_policy == "short":
        return np.asarray(short_scores, dtype=float)
    raise ValueError(f"unsupported side policy: {side_policy}")


def effective_selection_signal_hash(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: tuple[str, str],
    hold_bars: int = HOLD_BARS,
    stride_bars: int = ANCHOR_STRIDE,
    minimum_signal_position: int = 143,
) -> str:
    """Hash the non-overlapping executable decisions inside the selection window."""
    start, end = map(pd.Timestamp, window)
    window_mask = np.asarray((dates >= start) & (dates < end), dtype=bool)
    positions = np.arange(
        int(minimum_signal_position),
        len(market) - int(hold_bars) - 2,
        int(stride_bars),
        dtype=np.int64,
    )
    candidate = positions[
        window_mask[positions]
        & (np.asarray(long_active, dtype=bool)[positions] | np.asarray(short_active, dtype=bool)[positions])
    ]
    effective_long = np.zeros(len(market), dtype=bool)
    effective_short = np.zeros(len(market), dtype=bool)
    next_allowed = 0
    for signal_pos in candidate:
        if signal_pos < next_allowed:
            continue
        is_long = bool(long_active[signal_pos]) and not bool(short_active[signal_pos])
        is_short = bool(short_active[signal_pos]) and not bool(long_active[signal_pos])
        if not is_long and not is_short:
            continue
        exit_pos = int(signal_pos) + 1 + int(hold_bars)
        if exit_pos >= len(market) or not window_mask[exit_pos]:
            continue
        if is_long:
            effective_long[signal_pos] = True
        else:
            effective_short[signal_pos] = True
        next_allowed = exit_pos + 1
    return _signal_hash(effective_long[positions], effective_short[positions])


def _gate_specs() -> list[tuple[str, int | None, float | None]]:
    specs: list[tuple[str, int | None, float | None]] = [("positive", None, None)]
    for rolling_window in ROLLING_WINDOWS:
        for quantile in GATE_QUANTILES:
            specs.append(
                (f"rolling_{rolling_window}_q{quantile:.2f}", rolling_window, quantile)
            )
    return specs


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
    positions, forward_return, _ = anchor_dataset(market, features)
    ready_positions = label_ready_positions(positions)
    outcome_targets = executable_path_targets(market, positions, cfg)
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
    gate_specs = _gate_specs()

    for model_index, (model_id, model_kind, group_name) in enumerate(MODEL_SPECS, start=1):
        print(
            f"[{model_index}/{len(MODEL_SPECS)}] streaming {model_id} "
            f"({len(groups[group_name])} features x 4 outcomes)",
            file=sys.stderr,
            flush=True,
        )
        component_predictions: dict[str, np.ndarray] = {}
        component_diagnostics: dict[str, Any] = {}
        for component_name, component_target in outcome_targets.items():
            model = _make_model(model_kind)
            predictions, stream_diagnostics = delayed_online_predictions(
                model,
                feature_names=groups[group_name],
                matrix=anchor_matrices[group_name],
                targets=component_target,
                signal_positions=positions,
                ready_positions=ready_positions,
                target_clip=TARGET_CLIP,
            )
            component_predictions[component_name] = predictions
            component_diagnostics[component_name] = {
                "stream": stream_diagnostics,
                "concept_change": _model_change_counts(model),
                "score_quality": _score_diagnostics(
                    predictions, component_target, masks
                ),
            }
        model_diagnostics[model_id] = {
            "model_kind": model_kind,
            "feature_group": group_name,
            "feature_count": len(groups[group_name]),
            "feature_names": groups[group_name],
            "components": component_diagnostics,
        }

        for mae_penalty in UTILITY_PENALTIES:
            long_scores, short_scores, _ = contextual_utility_scores(
                component_predictions,
                mae_penalty=mae_penalty,
            )
            for side_policy in SIDE_POLICIES:
                policy_scores = policy_score_for_side(
                    long_scores,
                    short_scores,
                    side_policy=side_policy,
                )
                for gate_id, rolling_window, gate_quantile in gate_specs:
                    thresholds = causal_gate_thresholds(
                        policy_scores,
                        rolling_window=rolling_window,
                        quantile=gate_quantile,
                    )
                    long_active, short_active = utility_policy_masks(
                        long_scores,
                        short_scores,
                        thresholds,
                        positions,
                        len(market),
                        side_policy=side_policy,
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
                            "mae_penalty": mae_penalty,
                            "gate_id": gate_id,
                            "rolling_score_window_anchors": rolling_window,
                            "gate_quantile": gate_quantile,
                            "minimum_score_history": (
                                MIN_SCORE_HISTORY if rolling_window is not None else 0
                            ),
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

    selected, selected_signals = _select_distinct_top10(raw_candidates)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "river_version": importlib.metadata.version("river"),
        "selection_policy": (
            "rank distinct profitable contextual utility policies on 2023 strict "
            "full-window CAGR/MDD; freeze Top-10 before 2024+ metrics"
        ),
        "online_protocol": {
            "counterfactual_components": list(outcome_targets),
            "predict_before_current_example_is_queued": True,
            "label_ready_offset_bars": 1 + HOLD_BARS,
            "learn_only_after_path_exit_open_is_observable": True,
            "rolling_gate_uses_shifted_prior_scores": True,
            "minimum_completed_updates": MIN_MODEL_UPDATES,
            "flat_action_utility": 0.0,
        },
        "top10": selected,
        "trial_counts": {
            "model_specs": len(MODEL_SPECS),
            "utility_penalties": len(UTILITY_PENALTIES),
            "gate_specs": len(gate_specs),
            "side_policies": len(SIDE_POLICIES),
            "total_policy_specs": len(MODEL_SPECS)
            * len(UTILITY_PENALTIES)
            * len(gate_specs)
            * len(SIDE_POLICIES),
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
            "River delayed multi-head contextual action utility; predict long/short net "
            "return and MAE only from causal features; release all counterfactual labels "
            "after next-bar-entry plus 48h path; flat utility zero; shifted rolling gate; "
            "2023 Top-10 freeze; 2024/2025/2026 prequential OOS; 6bp/side; full-window "
            "CAGR; strict intratrade MDD"
        ),
        "manifest": str(manifest_path),
        "feature_groups": groups,
        "model_diagnostics": model_diagnostics,
        "target": {
            "components": list(outcome_targets),
            "entry_delay_bars": 1,
            "hold_bars": HOLD_BARS,
            "label_ready_offset_bars": 1 + HOLD_BARS,
            "target_clip_abs": TARGET_CLIP,
            "flat_action_utility": 0.0,
        },
        "forward_return_reference_sample_count": int(np.isfinite(forward_return).sum()),
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
