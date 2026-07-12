"""Leak-safe tabular-foundation-model alpha search on fixed 6-hour anchors.

TabICLv2 and conventional tree baselines are trained on 2020-2022 only.
Candidate score thresholds are ranked on the 2023 internal holdout.  A Top-10
manifest is written before Test 2024, Eval 2025, or 2026 metrics are computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import training.search_bidirectional_state_alpha as state_sim
from preprocessing.market_features import build_market_feature_frame
from training.long_regime_combo_scan import _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_bidirectional_state_alpha import Config, sim


WINDOWS = {
    "fit2020_2022": ("2020-01-01", "2023-01-01"),
    "holdout2023": ("2023-01-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
}
HOLD_BARS = 576
ANCHOR_STRIDE = 72


COMPACT_FEATURES = (
    "range_vol",
    "trend_12",
    "trend_24",
    "trend_96",
    "rsi_norm",
    "mfi_norm",
    "bb_z",
    "range_pos",
    "close_zscore_48",
    "return_zscore_48",
    "body_to_range",
    "shadow_imbalance",
    "volume_zscore",
    "window_drawdown",
    "taker_imbalance",
    "funding_rate",
    "funding_zscore",
    "premium_index_zscore",
    "premium_index_change",
    "oi_change",
    "oi_zscore",
    "dxy_zscore",
    "dxy_momentum",
    "kimchi_premium_zscore",
    "kimchi_premium_change",
    "usdkrw_zscore",
    "usdkrw_momentum",
    "rex_144_range_width_pct",
    "rex_144_range_pos",
    "rex_576_range_width_pct",
    "rex_576_range_pos",
    "rex_2016_range_width_pct",
    "rex_2016_range_pos",
    "rex_8640_range_width_pct",
    "rex_8640_range_pos",
    "htf_4h_return_1",
    "htf_4h_range_pos",
    "htf_1d_return_1",
    "htf_1d_range_pos",
    "htf_3d_return_1",
    "htf_3d_range_pos",
    "htf_1w_return_1",
    "htf_1w_range_pos",
    "weekly_return_1w",
    "weekly_range_pos",
    "quote_vol_rel_1d_30d",
    "trades_rel_1d_30d",
    "volume_rel_1d_30d",
    "dollar_flow_rel_4h_30d",
    "premium_abs_z",
    "funding_abs_z",
    "interest_score",
)


def feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    availability = {
        column
        for column in features.columns
        if column.endswith("_available")
        or column in {"external_any_available", "binance_aux_any_available"}
    }
    full = [column for column in features.columns if column not in availability]
    external_prefixes = (
        "funding",
        "premium",
        "oi_",
        "dxy",
        "kimchi",
        "usdkrw",
        "btckrw",
        "interest_score",
    )
    price = [
        column
        for column in full
        if not any(column == prefix or column.startswith(prefix) for prefix in external_prefixes)
    ]
    compact = [column for column in COMPACT_FEATURES if column in features.columns]
    return {"compact": compact, "price": price, "full": full}


def anchor_dataset(
    market: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    positions = np.arange(143, len(market) - HOLD_BARS - 2, ANCHOR_STRIDE, dtype=np.int64)
    opens = market["open"].to_numpy(float)
    entry = positions + 1
    exit_pos = entry + HOLD_BARS
    valid = (
        np.isfinite(opens[entry])
        & np.isfinite(opens[exit_pos])
        & (opens[entry] > 0.0)
        & (opens[exit_pos] > 0.0)
    )
    positions = positions[valid]
    target = np.log(opens[positions + 1 + HOLD_BARS] / opens[positions + 1])
    dates = pd.to_datetime(market["date"])
    return positions, target, dates


def split_mask_for_anchors(
    dates: pd.Series,
    positions: np.ndarray,
    start: str,
    end: str,
) -> np.ndarray:
    signal_dates = dates.iloc[positions].to_numpy()
    exit_dates = dates.iloc[positions + 1 + HOLD_BARS].to_numpy()
    start_time = np.datetime64(pd.Timestamp(start))
    end_time = np.datetime64(pd.Timestamp(end))
    return (signal_dates >= start_time) & (signal_dates < end_time) & (exit_dates < end_time)


def _fit_predict(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
) -> np.ndarray:
    if model_name == "tabiclv2":
        from tabicl import TabICLRegressor

        model = TabICLRegressor(
            n_estimators=4,
            batch_size=4,
            device="cuda",
            use_amp="auto",
            random_state=713,
            verbose=False,
        )
    elif model_name == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=713,
            ),
        )
    elif model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=400,
                min_samples_leaf=8,
                max_features=0.7,
                n_jobs=-1,
                random_state=713,
            ),
        )
    else:
        raise KeyError(model_name)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(predict_x), dtype=float)


def policy_masks(
    scores: np.ndarray,
    positions: np.ndarray,
    size: int,
    *,
    side_policy: str,
    low_threshold: float | None,
    high_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    long_active = np.zeros(size, dtype=bool)
    short_active = np.zeros(size, dtype=bool)
    finite = np.isfinite(scores)
    if side_policy in {"long", "both"} and high_threshold is not None:
        chosen = positions[finite & (scores >= high_threshold)]
        long_active[chosen] = True
    if side_policy in {"short", "both"} and low_threshold is not None:
        chosen = positions[finite & (scores <= low_threshold)]
        short_active[chosen] = True
    return long_active, short_active


def top10_promotions(selected: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = selected[:10]
    alpha = [row for row in eligible if row.get("passes_alpha_pool", False)]
    live = [row for row in eligible if row.get("passes_live_grade", False)]
    return alpha, live


def _signal_hash(long_active: np.ndarray, short_active: np.ndarray) -> str:
    packed = np.r_[np.packbits(long_active), np.packbits(short_active)]
    return hashlib.sha256(packed.tobytes()).hexdigest()[:16]


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args: argparse.Namespace) -> dict:
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
    positions, target, _ = anchor_dataset(market, features)
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    fit_mask = masks["fit2020_2022"]
    holdout_mask = masks["holdout2023"]
    post_fit_mask = ~fit_mask
    fit_positions = positions[fit_mask]
    post_fit_positions = positions[post_fit_mask]
    fit_target = target[fit_mask]
    target_low, target_high = np.quantile(fit_target, (0.01, 0.99))
    fit_target = np.clip(fit_target, target_low, target_high)
    zero = np.zeros(len(market), dtype=bool)
    model_specs = (
        ("tabiclv2", "compact"),
        ("tabiclv2", "price"),
        ("tabiclv2", "full"),
        ("histgb", "compact"),
        ("histgb", "full"),
        ("extra_trees", "compact"),
        ("extra_trees", "full"),
    )
    model_outputs: dict[str, dict] = {}
    raw_candidates: list[dict] = []
    policy_specs = []
    for quantile in (0.70, 0.80, 0.85, 0.90, 0.95):
        policy_specs.append(("long", None, quantile))
        policy_specs.append(("short", 1.0 - quantile, None))
        policy_specs.append(("both", 1.0 - quantile, quantile))

    for model_name, group_name in model_specs:
        columns = groups[group_name]
        matrix = features[columns].to_numpy(float)
        matrix[~np.isfinite(matrix)] = np.nan
        train_x = matrix[fit_positions]
        predict_x = matrix[post_fit_positions]
        predictions = _fit_predict(model_name, train_x, fit_target, predict_x)
        score_by_position = np.full(len(market), np.nan, dtype=float)
        score_by_position[post_fit_positions] = predictions
        holdout_scores = score_by_position[positions[holdout_mask]]
        holdout_target = target[holdout_mask]
        finite_holdout = np.isfinite(holdout_scores) & np.isfinite(holdout_target)
        if finite_holdout.sum() < 20:
            continue
        from scipy.stats import spearmanr

        model_id = f"{model_name}_{group_name}"
        model_outputs[model_id] = {
            "model_name": model_name,
            "feature_group": group_name,
            "feature_count": len(columns),
            "feature_names": columns,
            "fit_samples": int(fit_mask.sum()),
            "holdout_samples": int(holdout_mask.sum()),
            "holdout_spearman": float(
                spearmanr(holdout_scores[finite_holdout], holdout_target[finite_holdout]).statistic
            ),
            "holdout_direction_accuracy": float(
                np.mean(
                    np.sign(holdout_scores[finite_holdout])
                    == np.sign(holdout_target[finite_holdout])
                )
            ),
        }
        for side_policy, low_quantile, high_quantile in policy_specs:
            low_threshold = (
                float(np.quantile(holdout_scores[finite_holdout], low_quantile))
                if low_quantile is not None
                else None
            )
            high_threshold = (
                float(np.quantile(holdout_scores[finite_holdout], high_quantile))
                if high_quantile is not None
                else None
            )
            long_active, short_active = policy_masks(
                score_by_position[positions],
                positions,
                len(market),
                side_policy=side_policy,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
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
                    "model_name": model_name,
                    "feature_group": group_name,
                    "side_policy": side_policy,
                    "low_score_quantile": low_quantile,
                    "high_score_quantile": high_quantile,
                    "low_score_threshold": low_threshold,
                    "high_score_threshold": high_threshold,
                    "hold_bars": HOLD_BARS,
                    "anchor_stride_bars": ANCHOR_STRIDE,
                    "holdout2023": holdout,
                    "signal_hash": _signal_hash(long_active, short_active),
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
    selected: list[dict] = []
    seen_signals: set[str] = set()
    selected_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in raw_candidates:
        if row["signal_hash"] in seen_signals:
            continue
        seen_signals.add(row["signal_hash"])
        if len(selected) >= 10:
            continue
        long_active = row.pop("_long")
        short_active = row.pop("_short")
        selected.append(row)
        selected_signals[row["signal_hash"]] = (long_active, short_active)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "top10": selected,
        "trial_counts": {
            "model_specs": len(model_specs),
            "policy_specs_per_model": len(policy_specs),
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
    stress = {}
    for row in live_grade:
        long_active, short_active = selected_signals[row["signal_hash"]]
        stress[row["signal_hash"]] = {}
        for bps in (6, 8, 10, 15):
            stressed = replace(cfg, fee_rate=max(0.0, bps / 10000 - cfg.slippage_rate))
            stress[row["signal_hash"]][str(bps)] = {
                split: sim(
                    market,
                    dates,
                    long_active,
                    short_active,
                    stressed,
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
            "TabICLv2/tabular baselines fit 2020-2022 on causal 6h anchors; 2023 score thresholds "
            "and Top-10 rank; manifest frozen before later evaluation; next-bar entry; hold576; "
            "6bp/side; full-window CAGR; strict intratrade MDD"
        ),
        "source": "https://github.com/soda-inria/tabicl",
        "manifest": str(manifest_path),
        "feature_groups": groups,
        "model_diagnostics": model_outputs,
        "target": {
            "name": "next_48h_open_to_open_log_return",
            "train_clip_quantiles": [0.01, 0.99],
            "train_clip_values": [float(target_low), float(target_high)],
        },
        "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "tested_candidates": len(raw_candidates),
        "selected": selected,
        "alpha_pool_qualifiers": alpha_pool,
        "live_grade": live_grade,
        "cost_stress_bps_per_side": stress,
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
