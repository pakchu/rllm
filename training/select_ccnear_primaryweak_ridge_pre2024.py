#!/usr/bin/env python3
"""Test one fixed primary-weak ridge direction model on the frozen CC-near clock.

There is no candidate grid.  The event timestamp and 288-bar hold remain
unchanged.  A three-feature ridge model fits 2023H1 event outcomes and H2 is a
strict pass/fail confirmation; 2024+ is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_causal_weak_signal_sequence_interactions import (
    _last_signal_state,
    load_primary_schedules,
)
from training.evaluate_metaorder_fragmentation_impact_curvature import weekly_cluster_sign_flip
from training.evaluate_weak_signal_feature_ensemble import load_execution_market_and_funding
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.select_cross_collateral_event_direction_pre2024 import exact_factor
from training.select_cross_collateral_near_pressure_pre2024 import (
    Config as ClockConfig,
    EXPECTED_SELECTED as CLOCK_SPEC,
    event_mask,
    load_sources as load_clock_sources,
    raw_pressure,
    resolve_existing,
    schedule as pressure_sign_schedule,
    sha256,
)
from training.preregister_cross_collateral_liquidity_void_refill import lagged_robust_zscore


DEFAULT_OUTPUT = "results/ccnear_primaryweak_ridge_pre2024_2026-07-16.json"
DEFAULT_DOCS = "docs/ccnear-primaryweak-ridge-pre2024-2026-07-16.md"
DEFAULT_MANIFEST = "results/ccnear_primaryweak_ridge_manifest_2026-07-16.json"
FEATURE_NAMES = ("cash_vote_36", "derivative_vote_36", "refill_vote_36")
CASH_FAMILIES = ("cspr", "catch", "clasp")
DERIVATIVE_FAMILIES = ("umfr", "luri")
LOOKBACK_BARS = 36
RIDGE_ALPHA = 1.0
HOLD_BARS = int(CLOCK_SPEC["hold_bars"])
PERMUTATIONS = 100_000
PERMUTATION_SEED = 20_260_716

WINDOWS = {
    "fit_2023h1": ("2023-01-01", "2023-07-01"),
    "confirm_2023h2": ("2023-07-01", "2024-01-01"),
    "q3_2023": ("2023-07-01", "2023-10-01"),
    "q4_2023": ("2023-10-01", "2024-01-01"),
    "full_2023": ("2023-01-01", "2024-01-01"),
}


@dataclass(frozen=True)
class Config:
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS
    manifest_output: str = DEFAULT_MANIFEST


def recent_side(schedule: pd.DataFrame, frame_length: int, lookback_bars: int) -> np.ndarray:
    last_position, last_side = _last_signal_state(schedule, frame_length)
    position = np.arange(frame_length, dtype=np.int64)
    age = position - last_position
    available = (last_position >= 0) & (age >= 0) & (age <= int(lookback_bars))
    output = np.zeros(frame_length, dtype=np.int8)
    output[available] = last_side[available]
    return output


def build_vote_matrix(schedules: dict[str, pd.DataFrame], frame_length: int) -> np.ndarray:
    recent = {
        family: recent_side(schedule, frame_length, LOOKBACK_BARS)
        for family, schedule in schedules.items()
    }
    cash = sum((recent[family].astype(np.int16) for family in CASH_FAMILIES), start=np.zeros(frame_length, dtype=np.int16))
    derivative = sum(
        (recent[family].astype(np.int16) for family in DERIVATIVE_FAMILIES),
        start=np.zeros(frame_length, dtype=np.int16),
    )
    refill = recent["rift"].astype(np.int16)
    matrix = np.column_stack([cash, derivative, refill]).astype(float)
    if matrix.shape != (frame_length, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("primary weak vote matrix is invalid")
    return matrix


def build_clock_score(clock_cfg: ClockConfig) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    shells, credibility, market, funding, source = load_clock_sources(clock_cfg)
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
    score = (venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)
    score = score.where(shells["source_complete"].astype(bool) & score.notna())
    return market, score, {"funding": funding, **source}


def load_context(cfg: Config) -> dict[str, Any]:
    del cfg
    frame, primary_funding = load_execution_market_and_funding()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if len(frame) and frame["date"].max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("primary weak frame opened 2024+")
    schedules = load_primary_schedules()
    votes = build_vote_matrix(schedules, len(frame))

    clock_cfg = ClockConfig(output="/tmp/no_write.json", manifest_output="/tmp/no_write.json", docs_output="")
    clock_market, score, clock_source = build_clock_score(clock_cfg)
    frame_2023 = frame[(frame["date"] >= "2023-01-01") & (frame["date"] < "2024-01-01")]
    if not np.array_equal(frame_2023["date"].to_numpy(), clock_market["date"].to_numpy()):
        raise RuntimeError("primary weak and cross-collateral clocks differ")
    for column in ("open", "high", "low"):
        if not np.array_equal(
            frame_2023[column].to_numpy(float), clock_market[column].to_numpy(float)
        ):
            raise RuntimeError(f"primary weak and clock execution {column} differ")

    funding = pd.DataFrame(
        {
            "date": pd.to_datetime(primary_funding["funding_time_ms"], unit="ms", utc=True).dt.tz_convert(None),
            "funding_rate": pd.to_numeric(primary_funding["funding_rate"], errors="raise"),
        }
    )
    execution_cfg = ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="/tmp/no_write_primaryweak_ridge.json",
        manifest_output="/tmp/no_write_primaryweak_ridge_manifest.json",
        exclude_from="2024-01-01",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    engine = ExecutionEngine(frame, funding, execution_cfg)
    onset, _ = event_mask(score, float(CLOCK_SPEC["threshold"]))
    offset = int(frame_2023.index[0])
    event_positions = np.flatnonzero(onset).astype(np.int64) + offset
    labels = build_event_targets(event_positions, engine, execution_cfg)
    return {
        "frame": frame,
        "funding": funding,
        "votes": votes,
        "event_positions": event_positions,
        "labels": labels,
        "engine": engine,
        "execution_cfg": execution_cfg,
        "score_2023": score,
        "clock_market": clock_market,
        "clock_engine": ExecutionEngine(clock_market, clock_source["funding"], execution_cfg),
        "source": {
            **{key: value for key, value in clock_source.items() if key != "funding"},
            "primary_schedule_hashes": {
                family: hashlib.sha256(
                    schedule.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode()
                ).hexdigest()
                for family, schedule in schedules.items()
            },
        },
    }


def build_event_targets(
    positions: np.ndarray, engine: ExecutionEngine, cfg: ExecutionConfig
) -> dict[str, np.ndarray]:
    kept: list[int] = []
    target: list[float] = []
    for signal in positions:
        long_trade = engine.trade_at(int(signal), 1, HOLD_BARS, 1_000_000, 1_000_000)
        short_trade = engine.trade_at(int(signal), -1, HOLD_BARS, 1_000_000, 1_000_000)
        if long_trade is None or short_trade is None:
            continue
        kept.append(int(signal))
        target.append(exact_factor(long_trade, cfg) - exact_factor(short_trade, cfg))
    return {
        "positions": np.asarray(kept, dtype=np.int64),
        "target": np.asarray(target, dtype=float),
    }


def fit_ridge(context: dict[str, Any], *, end: str) -> Pipeline:
    positions = context["labels"]["positions"]
    dates = context["frame"]["date"].iloc[positions]
    cutoff = pd.Timestamp(end) - pd.Timedelta(minutes=5 * (HOLD_BARS + 1))
    fit = dates.lt(cutoff).to_numpy(bool)
    x = context["votes"][positions[fit]]
    y = context["labels"]["target"][fit]
    if len(y) < 80 or not np.isfinite(y).all():
        raise RuntimeError("ridge fit support is insufficient")
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
    model.fit(x, y)
    return model


def ridge_schedule(
    context: dict[str, Any], model: Pipeline, *, start: str, end: str, flip: bool = False
) -> list[Trade]:
    dates = context["frame"]["date"]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    positions = context["event_positions"]
    positions = positions[period[positions]]
    scores = model.predict(context["votes"][positions])
    trades: list[Trade] = []
    next_allowed = 0
    for signal, score in zip(positions, scores, strict=True):
        signal = int(signal)
        if signal < next_allowed:
            continue
        side = 1 if float(score) >= 0.0 else -1
        if flip:
            side *= -1
        trade = context["engine"].trade_at(signal, side, HOLD_BARS, 1_000_000, 1_000_000)
        if trade is None or not period[trade.exit_position]:
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def compact_stats(
    trades: list[Trade], context: dict[str, Any], *, start: str, end: str, permutations: int = 0
) -> dict[str, Any]:
    stats = {
        **equity_stats(trades, start=start, end=end, cfg=context["execution_cfg"]),
        "schedule_hash": _schedule_hash(trades),
    }
    returns = [exact_factor(trade, context["execution_cfg"]) - 1.0 for trade in trades]
    entries = [trade.entry_date for trade in trades]
    if permutations:
        stats["weekly_cluster_sign_flip"] = weekly_cluster_sign_flip(
            returns,
            entries,
            permutations=permutations,
            seed=PERMUTATION_SEED,
        )
    return stats


def model_payload(model: Pipeline) -> dict[str, Any]:
    scaler: StandardScaler = model.named_steps["scale"]
    ridge: Ridge = model.named_steps["ridge"]
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "lookback_bars": LOOKBACK_BARS,
        "ridge_alpha": RIDGE_ALPHA,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": ridge.coef_.tolist(),
    }
    payload["model_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def implementation_hash() -> str:
    functions = (
        recent_side,
        build_vote_matrix,
        build_event_targets,
        fit_ridge,
        ridge_schedule,
        compact_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# CC-near primary-weak ridge — pre-2024",
        "",
        "Exactly one direction model was tested. The frozen book event clock and 288-bar hold are "
        "unchanged; pressure sign is not a feature.",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in payload["stats"].items():
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(
        [
            "",
            f"H2 weekly-cluster p-value: {payload['stats']['confirm_2023h2']['weekly_cluster_sign_flip']['p_value_one_sided']:.6f}",
            "",
            f"Verdict: **{payload['verdict']}**",
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    context = load_context(cfg)
    h1_model = fit_ridge(context, end="2023-07-01")
    stats: dict[str, dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        primary = ridge_schedule(context, h1_model, start=start, end=end)
        stats[name] = compact_stats(
            primary,
            context,
            start=start,
            end=end,
            permutations=PERMUTATIONS if name == "confirm_2023h2" else 0,
        )
        flipped = ridge_schedule(context, h1_model, start=start, end=end, flip=True)
        controls[name] = compact_stats(flipped, context, start=start, end=end)

    pressure: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        trades = pressure_sign_schedule(
            context["clock_market"],
            context["clock_engine"],
            context["score_2023"],
            threshold=float(CLOCK_SPEC["threshold"]),
            hold_bars=HOLD_BARS,
            start=start,
            end=end,
        )
        pressure[name] = compact_stats(trades, context, start=start, end=end)

    h2 = stats["confirm_2023h2"]
    gate_checks = {
        "h2_positive": h2["absolute_return_pct"] > 0.0,
        "h2_ratio_at_least_3": h2["cagr_to_strict_mdd"] >= 3.0,
        "h2_mdd_at_most_15": h2["strict_mdd_pct"] <= 15.0,
        "h2_long_support": h2["longs"] >= 15,
        "h2_short_support": h2["shorts"] >= 15,
        "q3_positive": stats["q3_2023"]["absolute_return_pct"] > 0.0,
        "q4_positive": stats["q4_2023"]["absolute_return_pct"] > 0.0,
        "weekly_cluster_p_below_0_10": h2["weekly_cluster_sign_flip"]["p_value_one_sided"] < 0.10,
        "beats_flipped_h2": h2["cagr_to_strict_mdd"] > controls["confirm_2023h2"]["cagr_to_strict_mdd"],
    }
    future_ready = all(gate_checks.values())
    final_model = fit_ridge(context, end="2024-01-01") if future_ready else None
    payload = {
        "schema_version": 1,
        "mode": "ccnear_primaryweak_ridge_pre2024",
        "candidate_count": 1,
        "post_2023_rows_opened": False,
        "clock_spec": CLOCK_SPEC,
        "candidate": {
            "name": "ccnear_primaryweak_ridge_v1",
            "feature_names": list(FEATURE_NAMES),
            "feature_definition": {
                "cash_vote_36": list(CASH_FAMILIES),
                "derivative_vote_36": list(DERIVATIVE_FAMILIES),
                "refill_vote_36": ["rift"],
            },
            "lookback_bars": LOOKBACK_BARS,
            "ridge_alpha": RIDGE_ALPHA,
            "fit": "2023H1 only; H2 is confirmation only",
            "abstention": False,
        },
        "h1_model": model_payload(h1_model),
        "final_model": model_payload(final_model) if final_model is not None else None,
        "stats": stats,
        "flipped_control": controls,
        "pressure_sign_control": pressure,
        "gate_checks": gate_checks,
        "future_ready": future_ready,
        "verdict": "freeze before future replay" if future_ready else "reject before future replay",
        "source": context["source"],
        "implementation_hash": implementation_hash(),
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(render_docs(payload), encoding="utf-8")
    if future_ready:
        manifest = {
            "schema_version": 1,
            "future_outcomes_opened": False,
            "selection_cutoff_exclusive": "2024-01-01",
            "candidate_count": 1,
            "clock_spec": CLOCK_SPEC,
            "candidate": payload["candidate"],
            "final_model": payload["final_model"],
            "selection_result": cfg.output,
            "selection_result_sha256": sha256(output),
            "implementation_hash": payload["implementation_hash"],
            "selected_schedule_hashes": {
                key: value["schedule_hash"] for key, value in stats.items()
            },
            "source": payload["source"],
        }
        manifest["manifest_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest_path = Path(cfg.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "candidate": payload["candidate"],
                "stats": payload["stats"],
                "gate_checks": payload["gate_checks"],
                "future_ready": payload["future_ready"],
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
