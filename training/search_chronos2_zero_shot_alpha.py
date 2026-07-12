"""Chronos-2 zero-shot multivariate/covariate BTC alpha search.

The 120M time-series foundation model receives only completed hourly history.
At each six-hour signal anchor it forecasts 48 hourly log-price steps using
past-only volume, flow, range, funding, premium, and FX covariates.  No market
sample is used to fine-tune the foundation model.  Causal rolling score
percentiles are ranked on 2023 and frozen before 2024+ strict metrics.
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
import torch
from scipy.stats import spearmanr

import training.search_bidirectional_state_alpha as state_sim
from training.evaluate_invariant_ensemble_uncertainty import (
    signed_dynamic_policy_masks,
)
from training.long_regime_combo_scan import _load_market
from training.search_bidirectional_state_alpha import Config, sim
from training.search_river_contextual_utility_alpha import (
    effective_selection_signal_hash,
)
from training.search_river_online_alpha import (
    causal_rolling_thresholds,
    validate_output_paths,
)
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


MODEL_ID = "amazon/chronos-2"
CONTEXT_HOURS = 720
PREDICTION_HOURS = 48
ROLLING_WINDOWS = (720, 1460)
SCORE_QUANTILES = (0.70, 0.80, 0.90, 0.95)
COVARIATE_COLUMNS = (
    "log_quote_volume",
    "taker_imbalance",
    "hourly_range",
    "funding_rate",
    "premium_index",
    "dxy_zscore",
    "kimchi_premium_zscore",
    "usdkrw_zscore",
)


def causal_hourly_frame(market: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5m candle-open timestamps to completed hourly observations."""
    frame = market.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    quote = frame["quote_asset_volume"].astype(float)
    buy = frame["taker_buy_quote"].astype(float)
    hourly = pd.DataFrame(
        {
            "open": frame["open"].astype(float).resample(
                "1h", closed="left", label="right"
            ).first(),
            "high": frame["high"].astype(float).resample(
                "1h", closed="left", label="right"
            ).max(),
            "low": frame["low"].astype(float).resample(
                "1h", closed="left", label="right"
            ).min(),
            "close": frame["close"].astype(float).resample(
                "1h", closed="left", label="right"
            ).last(),
            "quote": quote.resample("1h", closed="left", label="right").sum(),
            "buy": buy.resample("1h", closed="left", label="right").sum(),
        }
    )
    for column in (
        "funding_rate",
        "premium_index",
        "dxy_zscore",
        "kimchi_premium_zscore",
        "usdkrw_zscore",
    ):
        values = (
            frame[column].astype(float)
            if column in frame.columns
            else pd.Series(0.0, index=frame.index)
        )
        hourly[column] = values.resample(
            "1h", closed="left", label="right"
        ).last()
    hourly = hourly.dropna(subset=["open", "high", "low", "close", "quote", "buy"])
    hourly["log_close"] = np.log(hourly["close"].clip(lower=1e-12))
    hourly["log_quote_volume"] = np.log1p(hourly["quote"].clip(lower=0.0))
    hourly["taker_imbalance"] = (
        2.0 * hourly["buy"] / hourly["quote"].replace(0.0, np.nan) - 1.0
    ).clip(-1.0, 1.0)
    hourly["hourly_range"] = (
        (hourly["high"] - hourly["low"]) / hourly["close"].replace(0.0, np.nan)
    )
    hourly.loc[:, COVARIATE_COLUMNS] = (
        hourly.loc[:, COVARIATE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
    )
    return hourly


def anchor_hour_indices(
    market_dates: pd.Series,
    positions: np.ndarray,
    hourly_index: pd.DatetimeIndex,
    *,
    bar_minutes: int = 5,
) -> np.ndarray:
    """Map each signal close time to its latest completed hourly observation."""
    decision_times = pd.to_datetime(market_dates.iloc[positions]).to_numpy() + np.timedelta64(
        int(bar_minutes), "m"
    )
    return hourly_index.to_numpy().searchsorted(decision_times, side="right") - 1


def build_chronos_inputs(
    hourly: pd.DataFrame,
    hour_indices: np.ndarray,
    *,
    context_hours: int = CONTEXT_HOURS,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    target = hourly["log_close"].to_numpy(np.float32)
    covariates = {
        column: hourly[column].to_numpy(np.float32) for column in COVARIATE_COLUMNS
    }
    valid_anchor_indices = np.flatnonzero(hour_indices + 1 >= int(context_hours))
    inputs: list[dict[str, Any]] = []
    for anchor_index in valid_anchor_indices:
        end = int(hour_indices[anchor_index]) + 1
        start = end - int(context_hours)
        inputs.append(
            {
                "target": target[start:end],
                "past_covariates": {
                    name: values[start:end] for name, values in covariates.items()
                },
            }
        )
    return inputs, valid_anchor_indices


def forecast_score_streams(
    predictions: list[torch.Tensor],
    valid_anchor_indices: np.ndarray,
    entry_log_prices: np.ndarray,
    quantiles: list[float],
    total_anchors: int,
) -> dict[str, np.ndarray]:
    quantile_index = {round(float(q), 8): index for index, q in enumerate(quantiles)}
    q10 = quantile_index[0.1]
    q50 = quantile_index[0.5]
    q90 = quantile_index[0.9]
    streams = {
        name: np.full(int(total_anchors), np.nan, dtype=float)
        for name in (
            "median_terminal",
            "central_terminal",
            "quantile_mean_terminal",
            "median_path_mean",
            "median_24h_48h_consensus",
            "terminal_interval_snr",
        )
    }
    for prediction, anchor_index in zip(predictions, valid_anchor_indices, strict=True):
        values = prediction.detach().cpu().numpy()[0]
        entry = float(entry_log_prices[anchor_index])
        median_path = values[q50]
        q10_terminal = float(values[q10, -1] - entry)
        q50_terminal = float(values[q50, -1] - entry)
        q90_terminal = float(values[q90, -1] - entry)
        streams["median_terminal"][anchor_index] = q50_terminal
        streams["central_terminal"][anchor_index] = (
            q10_terminal + q90_terminal
        ) / 2.0
        streams["quantile_mean_terminal"][anchor_index] = float(
            values[q10 : q90 + 1, -1].mean() - entry
        )
        streams["median_path_mean"][anchor_index] = float(median_path.mean() - entry)
        streams["median_24h_48h_consensus"][anchor_index] = float(
            (median_path[23] + median_path[-1]) / 2.0 - entry
        )
        streams["terminal_interval_snr"][anchor_index] = q50_terminal / max(
            1e-4, q90_terminal - q10_terminal
        )
    return streams


def _score_quality(
    scores: np.ndarray,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for split, split_mask in masks.items():
        finite = split_mask & np.isfinite(scores) & np.isfinite(targets)
        count = int(finite.sum())
        correlation = (
            spearmanr(scores[finite], targets[finite]).statistic if count >= 3 else np.nan
        )
        output[split] = {
            "samples": count,
            "spearman": float(correlation) if np.isfinite(correlation) else None,
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
    positions, targets, _ = anchor_dataset(market, pd.DataFrame(index=market.index))
    masks = {
        name: split_mask_for_anchors(dates, positions, *bounds)
        for name, bounds in WINDOWS.items()
    }
    hourly = causal_hourly_frame(market)
    hour_indices = anchor_hour_indices(dates, positions, hourly.index)
    inputs, valid_anchor_indices = build_chronos_inputs(
        hourly,
        hour_indices,
        context_hours=args.context_hours,
    )
    entry_log_prices = np.log(market["open"].to_numpy(float)[positions + 1])

    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda")
    batch_count = 0

    def progress() -> None:
        nonlocal batch_count
        batch_count += 1
        if batch_count % 25 == 0:
            print(f"chronos batches completed: {batch_count}", file=sys.stderr, flush=True)

    predictions = pipeline.predict(
        inputs,
        prediction_length=PREDICTION_HOURS,
        batch_size=args.batch_size,
        context_length=args.context_hours,
        cross_learning=False,
        after_batch=progress,
    )
    score_streams = forecast_score_streams(
        predictions,
        valid_anchor_indices,
        entry_log_prices,
        pipeline.quantiles,
        len(positions),
    )
    model_commit = getattr(pipeline.model.config, "_commit_hash", None)
    del predictions
    del pipeline
    torch.cuda.empty_cache()

    diagnostics = {
        stream_name: _score_quality(scores, targets, masks)
        for stream_name, scores in score_streams.items()
    }
    raw_candidates: list[dict[str, Any]] = []
    for stream_name, scores in score_streams.items():
        for rolling_window in ROLLING_WINDOWS:
            for quantile in SCORE_QUANTILES:
                low_thresholds, high_thresholds = causal_rolling_thresholds(
                    scores,
                    window=rolling_window,
                    quantile=quantile,
                )
                for side_policy in ("long", "short", "both"):
                    long_active, short_active = signed_dynamic_policy_masks(
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
                            "stream_name": stream_name,
                            "rolling_score_window_anchors": rolling_window,
                            "score_quantile": quantile,
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
        selected.append(candidate)
        selected_signals[signal_hash] = (
            candidate.pop("_long"),
            candidate.pop("_short"),
        )
        if len(selected) == 10:
            break

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head_before_experiment_commit": _git_head(),
        "selection_window": WINDOWS["holdout2023"],
        "later_metrics_included": False,
        "model": {
            "id": args.model_id,
            "revision": model_commit,
            "chronos_version": importlib.metadata.version("chronos-forecasting"),
            "parameters": "120M",
            "fine_tuned_on_repo_data": False,
            "cross_learning": False,
        },
        "input": {
            "frequency": "1h completed candles",
            "context_hours": args.context_hours,
            "prediction_hours": PREDICTION_HOURS,
            "target": "log_close",
            "past_covariates": list(COVARIATE_COLUMNS),
            "anchor_count": len(positions),
            "forecasted_anchor_count": len(valid_anchor_indices),
        },
        "selection_policy": (
            "zero-shot Chronos-2 fixed forecast transforms; shifted rolling score "
            "percentiles; 2023 executed-path Top-10 freeze before 2024+ metrics"
        ),
        "top10": selected,
        "trial_counts": {
            "score_streams": len(score_streams),
            "rolling_windows": len(ROLLING_WINDOWS),
            "score_quantiles": len(SCORE_QUANTILES),
            "side_policies": 3,
            "total_policy_specs": len(score_streams)
            * len(ROLLING_WINDOWS)
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
        "source": "https://github.com/amazon-science/chronos-forecasting",
        "protocol": manifest["selection_policy"]
        + "; completed-hour causal input; next-bar 5m entry; hold576; 6bp/side; "
        "full-window CAGR; strict intratrade MDD",
        "manifest": str(manifest_path),
        "model": manifest["model"],
        "input": manifest["input"],
        "score_diagnostics": diagnostics,
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
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--context-hours", type=int, default=CONTEXT_HOURS)
    parser.add_argument("--batch-size", type=int, default=256)
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
