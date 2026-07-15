#!/usr/bin/env python3
"""Select a causal direction model for the frozen orthogonal book event clock.

The pressure-sign policy is intentionally not reused.  Model families and
regularization are ranked on generic next-day direction in 2021/2022, then the
single selected specification is tested on the frozen 2023 book event clock.
No 2024+ row is loaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_cross_collateral_near_pressure_oos import validate_selection_manifest
from training.preregister_cross_collateral_liquidity_void_refill import lagged_robust_zscore
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.select_cross_collateral_near_pressure_pre2024 import (
    Config as ClockConfig,
    EXPECTED_SELECTED as CLOCK_SPEC,
    event_mask,
    load_sources as load_clock_sources,
    raw_pressure,
    resolve_existing,
    sha256,
)


DEFAULT_OUTPUT = "results/cross_collateral_event_direction_pre2024_2026-07-16.json"
DEFAULT_DOCS = "docs/cross-collateral-event-direction-pre2024-2026-07-16.md"
MODEL_CS = (0.03, 0.1, 0.3, 1.0)
CONFIDENCE_MARGINS = (0.0, 0.03, 0.06)
HOLD_BARS = int(CLOCK_SPEC["hold_bars"])

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend_state": (
        "ret_12",
        "ret_48",
        "ret_144",
        "ret_288",
        "ret_576",
        "ret_2016",
        "range_pos_144",
        "range_pos_576",
        "range_pos_2016",
        "realized_vol_288",
    ),
    "trend_flow": (
        "ret_12",
        "ret_48",
        "ret_144",
        "ret_288",
        "ret_576",
        "ret_2016",
        "range_pos_144",
        "range_pos_576",
        "range_pos_2016",
        "realized_vol_288",
        "taker_12",
        "taker_48",
        "taker_144",
        "volume_z_48",
        "volume_z_288",
        "clv_12",
    ),
    "trend_carry": (
        "ret_12",
        "ret_48",
        "ret_144",
        "ret_288",
        "ret_576",
        "ret_2016",
        "range_pos_144",
        "range_pos_576",
        "range_pos_2016",
        "realized_vol_288",
        "funding_rate_lag1",
        "funding_z_2016",
    ),
    "weak_combo": (
        "ret_12",
        "ret_48",
        "ret_144",
        "ret_288",
        "ret_576",
        "ret_2016",
        "range_pos_144",
        "range_pos_576",
        "range_pos_2016",
        "realized_vol_288",
        "taker_12",
        "taker_48",
        "taker_144",
        "volume_z_48",
        "volume_z_288",
        "clv_12",
        "funding_rate_lag1",
        "funding_z_2016",
    ),
}

GENERIC_FOLDS = {
    "oos_2021": ("2020-01-01", "2021-01-01", "2021-01-01", "2022-01-01"),
    "oos_2022": ("2020-01-01", "2022-01-01", "2022-01-01", "2023-01-01"),
}
EVENT_WINDOWS = {
    "h1_2023": ("2023-01-01", "2023-07-01"),
    "h2_2023": ("2023-07-01", "2024-01-01"),
    "q1_2023": ("2023-01-01", "2023-04-01"),
    "q2_2023": ("2023-04-01", "2023-07-01"),
    "q3_2023": ("2023-07-01", "2023-10-01"),
    "q4_2023": ("2023-10-01", "2024-01-01"),
    "full_2023": ("2023-01-01", "2024-01-01"),
}


@dataclass(frozen=True)
class Config:
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    clock_manifest: str = (
        "results/cross_collateral_near_pressure_pre2024_manifest_2026-07-16.json"
    )


def exact_factor(trade: Trade, cfg: ExecutionConfig) -> float:
    cost = 1.0 - float(cfg.leverage) * float(cfg.fee_rate + cfg.slippage_rate)
    return float(cost * trade.price_factor * trade.funding_factor * cost)


def load_context(cfg: Config) -> dict[str, Any]:
    selection_manifest, _ = validate_selection_manifest(cfg.clock_manifest)
    clock_cfg = ClockConfig(output="/tmp/no_write.json", manifest_output="/tmp/no_write.json", docs_output="")
    shells, credibility, selection_market, _, source = load_clock_sources(clock_cfg)
    market_manifest_path = resolve_existing(clock_cfg.market_manifest)
    market_manifest = json.loads(market_manifest_path.read_text(encoding="utf-8"))
    market_path = resolve_existing(market_manifest["combined_output"])
    if sha256(market_path) != selection_manifest["source"]["market_sha256"]:
        raise RuntimeError("direction market differs from the frozen clock market")
    market = pd.read_csv(market_path, compression="infer", parse_dates=["date"])
    market = market[(market["date"] >= "2020-01-01") & (market["date"] < "2024-01-01")]
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    intervals = market["date"].diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise RuntimeError("direction market is not a complete 5-minute grid")
    selection_slice = market[(market["date"] >= "2023-01-01") & (market["date"] < "2024-01-01")]
    if not np.array_equal(selection_slice["date"].to_numpy(), selection_market["date"].to_numpy()):
        raise RuntimeError("direction and clock timestamps differ in 2023")

    funding_path = resolve_existing(clock_cfg.funding_csv)
    funding = pd.read_csv(funding_path, compression="infer")[["date", "funding_rate"]]
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="raise")
    funding = funding[(funding["date"] >= "2020-01-01") & (funding["date"] < "2024-01-01")]
    funding = funding.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    execution_cfg = ExecutionConfig(
        input_csv=str(market_path),
        metrics_csv="",
        funding_csv=str(funding_path),
        output="/tmp/no_write_direction.json",
        manifest_output="/tmp/no_write_direction_manifest.json",
        exclude_from="2024-01-01",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    engine = ExecutionEngine(market, funding, execution_cfg)
    features = build_features(market, funding)
    labels = build_daily_labels(market, features, engine, execution_cfg)

    weights = (1.0, 0.5, 0.0, 0.0, 0.0)
    venue_scores = []
    for venue in ("um", "cm"):
        venue_scores.append(
            lagged_robust_zscore(
                raw_pressure(
                    shells,
                    credibility,
                    venue=venue,
                    weights=weights,
                    credibility_weighted=False,
                ),
                window=clock_cfg.robust_window_bars,
                minimum=clock_cfg.robust_min_periods,
            )
        )
    clock_score = (venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)
    clock_score = clock_score.where(shells["source_complete"].astype(bool) & clock_score.notna())
    onset, _ = event_mask(clock_score, float(CLOCK_SPEC["threshold"]))
    offset = int(selection_slice.index[0])
    event_positions = np.flatnonzero(onset).astype(np.int64) + offset
    return {
        "market": market,
        "funding": funding,
        "features": features,
        "labels": labels,
        "engine": engine,
        "execution_cfg": execution_cfg,
        "event_positions": event_positions,
        "source": {
            **source,
            "market_sha256": sha256(market_path),
            "funding_sha256": sha256(funding_path),
            "clock_manifest_sha256": sha256(resolve_existing(cfg.clock_manifest)),
        },
    }


def build_features(market: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="raise")
    high = pd.to_numeric(market["high"], errors="raise")
    low = pd.to_numeric(market["low"], errors="raise")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="raise")
    taker = pd.to_numeric(market["taker_buy_quote"], errors="raise")
    log_close = np.log(close.where(close > 0.0))
    values: dict[str, pd.Series] = {}
    for window in (12, 48, 144, 288, 576, 2016):
        values[f"ret_{window}"] = log_close - log_close.shift(window)
    for window in (144, 576, 2016):
        rolling_high = high.rolling(window, min_periods=window).max()
        rolling_low = low.rolling(window, min_periods=window).min()
        values[f"range_pos_{window}"] = (
            (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
        ) * 2.0 - 1.0
    one_bar = log_close.diff()
    values["realized_vol_288"] = one_bar.rolling(288, min_periods=288).std(ddof=0)
    imbalance = 2.0 * taker / quote.replace(0.0, np.nan) - 1.0
    for window in (12, 48, 144):
        values[f"taker_{window}"] = imbalance.rolling(window, min_periods=window).mean()
    log_quote = np.log1p(quote)
    for window in (48, 288):
        mean = log_quote.rolling(window, min_periods=window).mean()
        std = log_quote.rolling(window, min_periods=window).std(ddof=0)
        values[f"volume_z_{window}"] = (log_quote - mean) / std.replace(0.0, np.nan)
    candle_range = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / candle_range
    values["clv_12"] = clv.rolling(12, min_periods=12).mean()

    funding_lag = pd.merge_asof(
        market[["date"]], funding, on="date", direction="backward", allow_exact_matches=True
    )["funding_rate"].shift(1)
    values["funding_rate_lag1"] = funding_lag
    mean = funding_lag.rolling(2016, min_periods=576).mean()
    std = funding_lag.rolling(2016, min_periods=576).std(ddof=0)
    values["funding_z_2016"] = (funding_lag - mean) / std.replace(0.0, np.nan)
    frame = pd.DataFrame(values)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_daily_labels(
    market: pd.DataFrame,
    features: pd.DataFrame,
    engine: ExecutionEngine,
    cfg: ExecutionConfig,
) -> dict[str, Any]:
    dates = pd.to_datetime(market["date"])
    finite = np.isfinite(features[list(FEATURE_GROUPS["weak_combo"])].to_numpy(float)).all(axis=1)
    daily = dates.dt.hour.eq(0) & dates.dt.minute.eq(0)
    positions = np.flatnonzero(daily.to_numpy(bool) & finite)
    kept: list[int] = []
    targets: list[int] = []
    advantages: list[float] = []
    trades: dict[int, dict[int, Trade]] = {}
    for signal in positions:
        long_trade = engine.trade_at(int(signal), 1, HOLD_BARS, 1_000_000, 1_000_000)
        short_trade = engine.trade_at(int(signal), -1, HOLD_BARS, 1_000_000, 1_000_000)
        if long_trade is None or short_trade is None:
            continue
        long_factor = exact_factor(long_trade, cfg)
        short_factor = exact_factor(short_trade, cfg)
        kept.append(int(signal))
        targets.append(int(long_factor >= short_factor))
        advantages.append(abs(long_factor - short_factor))
        trades[int(signal)] = {1: long_trade, -1: short_trade}
    return {
        "positions": np.asarray(kept, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.int8),
        "weight": np.clip(np.asarray(advantages, dtype=float), 1e-5, 0.10),
        "trades": trades,
    }


def fit_model(
    context: dict[str, Any], feature_names: tuple[str, ...], c_value: float, *, end: str
) -> Pipeline:
    positions = context["labels"]["positions"]
    dates = pd.to_datetime(context["market"]["date"]).iloc[positions]
    fit = dates.lt(pd.Timestamp(end) - pd.Timedelta(minutes=5 * (HOLD_BARS + 1))).to_numpy(bool)
    x = context["features"].loc[positions[fit], list(feature_names)].to_numpy(float)
    y = context["labels"]["target"][fit]
    weights = context["labels"]["weight"][fit]
    if len(np.unique(y)) != 2:
        raise RuntimeError("direction fit lost one class")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=float(c_value), max_iter=2_000, solver="lbfgs")),
        ]
    )
    model.fit(x, y, model__sample_weight=weights)
    return model


def predicted_schedule(
    context: dict[str, Any],
    model: Pipeline,
    feature_names: tuple[str, ...],
    positions: np.ndarray,
    *,
    start: str,
    end: str,
    margin: float,
) -> list[Trade]:
    dates = pd.to_datetime(context["market"]["date"])
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    positions = np.asarray(positions, dtype=np.int64)
    positions = positions[period[positions]]
    finite = np.isfinite(
        context["features"].loc[positions, list(feature_names)].to_numpy(float)
    ).all(axis=1)
    positions = positions[finite]
    if not len(positions):
        return []
    probabilities = model.predict_proba(
        context["features"].loc[positions, list(feature_names)].to_numpy(float)
    )[:, 1]
    trades: list[Trade] = []
    next_allowed = 0
    for signal, probability in zip(positions, probabilities, strict=True):
        signal = int(signal)
        if signal < next_allowed or abs(float(probability) - 0.5) < float(margin):
            continue
        side = 1 if probability >= 0.5 else -1
        trade = context["engine"].trade_at(signal, side, HOLD_BARS, 1_000_000, 1_000_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def slim_stats(trades: list[Trade], context: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    return {
        **equity_stats(trades, start=start, end=end, cfg=context["execution_cfg"]),
        "schedule_hash": _schedule_hash(trades),
    }


def evaluate_generic_cell(context: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    names = FEATURE_GROUPS[cell["feature_group"]]
    result = {**cell, "folds": {}}
    positions = context["labels"]["positions"]
    for fold, (_, fit_end, start, end) in GENERIC_FOLDS.items():
        model = fit_model(context, names, float(cell["c_value"]), end=fit_end)
        trades = predicted_schedule(
            context,
            model,
            names,
            positions,
            start=start,
            end=end,
            margin=float(cell["confidence_margin"]),
        )
        result["folds"][fold] = slim_stats(trades, context, start, end)
    rows = list(result["folds"].values())
    result["support_pass"] = all(
        row["trades"] >= 80 and row["longs"] >= 15 and row["shorts"] >= 15 for row in rows
    )
    result["worst_ratio"] = min(row["cagr_to_strict_mdd"] for row in rows)
    result["worst_return"] = min(row["absolute_return_pct"] for row in rows)
    return result


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"feature_group": group, "c_value": c_value, "confidence_margin": margin}
        for group in FEATURE_GROUPS
        for c_value in MODEL_CS
        for margin in CONFIDENCE_MARGINS
    ]


def select_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["support_pass"] and row["worst_return"] > 0.0]
    pool = eligible or [row for row in rows if row["support_pass"]] or rows
    return max(
        pool,
        key=lambda row: (
            row["worst_ratio"],
            row["worst_return"],
            -len(FEATURE_GROUPS[row["feature_group"]]),
            -float(row["c_value"]),
            -float(row["confidence_margin"]),
        ),
    )


def render_docs(payload: dict[str, Any]) -> str:
    selected = payload["selected"]
    lines = [
        "# Cross-collateral event direction pre-2024 selection",
        "",
        "The orthogonal book-pressure onset is used only as an event clock. Direction comes from a "
        "regularized weak-signal model selected on generic 2021/2022 annual OOS folds, never from "
        "the pressure sign.",
        "",
        f"Selected: `{selected['feature_group']}`, C={selected['c_value']}, "
        f"confidence margin={selected['confidence_margin']}.",
        "",
        "## Generic direction folds",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in selected["folds"].items():
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-clock 2023 test",
            "",
            "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stats in payload["event_stats"].items():
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(["", f"Verdict: **{payload['verdict']}**", ""])
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    context = load_context(cfg)
    cells = [evaluate_generic_cell(context, cell) for cell in candidate_grid()]
    selected = select_cell(cells)
    names = FEATURE_GROUPS[selected["feature_group"]]
    final_model = fit_model(context, names, float(selected["c_value"]), end="2023-01-01")
    event_stats: dict[str, dict[str, Any]] = {}
    for name, (start, end) in EVENT_WINDOWS.items():
        trades = predicted_schedule(
            context,
            final_model,
            names,
            context["event_positions"],
            start=start,
            end=end,
            margin=float(selected["confidence_margin"]),
        )
        event_stats[name] = slim_stats(trades, context, start, end)
    half_pass = all(
        event_stats[name]["absolute_return_pct"] > 0.0
        and event_stats[name]["cagr_to_strict_mdd"] >= 2.5
        and event_stats[name]["trades"] >= 30
        for name in ("h1_2023", "h2_2023")
    )
    quarter_positive = all(event_stats[name]["absolute_return_pct"] > 0.0 for name in EVENT_WINDOWS if name.startswith("q"))
    payload = {
        "schema_version": 1,
        "mode": "cross_collateral_event_direction_pre2024",
        "post_2023_rows_opened": False,
        "grid_cells": len(cells),
        "clock_spec": CLOCK_SPEC,
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "selection_protocol": {
            "generic_annual_oos_folds": GENERIC_FOLDS,
            "clock_combination_test": EVENT_WINDOWS,
            "future_rows_excluded": "all inputs end before 2024-01-01",
        },
        "source": context["source"],
        "selected": selected,
        "event_stats": event_stats,
        "half_pass": half_pass,
        "quarter_positive": quarter_positive,
        "future_ready": bool(half_pass and quarter_positive),
        "verdict": (
            "freeze before future replay"
            if half_pass and quarter_positive
            else "reject before future replay"
        ),
        "cells": cells,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--clock-manifest", default=Config.clock_manifest)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "selected": payload["selected"],
                "event_stats": payload["event_stats"],
                "future_ready": payload["future_ready"],
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
