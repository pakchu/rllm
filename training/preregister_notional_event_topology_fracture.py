"""Support-only preregistration scaffold for the NETF alpha.

NETF detects a disagreement between aggressive-event breadth and aggressive
notional, then waits for a causal capital-revelation transition.  This module
contains no future-return or backtest calculation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
    nonoverlapping_schedule,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    confirmation_bars: int
    hold_bars: int


CANDIDATES = (
    Candidate("netf_fast", 6, 48),
    Candidate("netf_slow", 12, 96),
)
SUPPORT_CALIBRATION_GRID = (0.85, 0.875, 0.90, 0.95, 0.975)


@dataclass(frozen=True)
class Config:
    output: str = "results/notional_event_topology_fracture_support_2026-07-14.json"
    tension_quantile: float = 0.875
    structure_quantile: float = 0.80
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    minimum_agg_trade_count: int = 64
    minimum_nonoverlap_total: int = 250
    minimum_nonoverlap_per_year: int = 40
    minimum_nonoverlap_per_2023_half: int = 20
    minimum_side_share: float = 0.25


def _lagged_clean_quantile(
    values: pd.Series,
    clean: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if not 1 <= minimum <= window:
        raise ValueError("baseline periods are invalid")
    return (
        values.where(clean)
        .shift(1)
        .rolling(window, min_periods=minimum)
        .quantile(quantile)
    )


def compute_netf(
    frame: pd.DataFrame,
    candidate: Candidate,
    cfg: Config,
) -> pd.DataFrame:
    if candidate.confirmation_bars < 1 or candidate.hold_bars < 1:
        raise ValueError("NETF confirmation and hold bars must be positive")
    clean = ~frame["quarantined"].astype(bool)
    capital_direction = pd.Series(
        np.sign(frame["signed_quote_notional"].astype(float)),
        index=frame.index,
        dtype=float,
    )
    crowd_direction = pd.Series(
        np.sign(frame["signed_event_imbalance"].astype(float)),
        index=frame.index,
        dtype=float,
    )
    immediate_price_direction = pd.Series(
        np.sign(frame["micro_log_return"].astype(float)),
        index=frame.index,
        dtype=float,
    )

    # Breadth-versus-capital disagreement is strongest when both event-count
    # imbalance and quote-notional imbalance are material and the buy/sell
    # average-size asymmetry is large.
    topology_tension = (
        np.sqrt(
            frame["flow_coherence"].clip(lower=0.0).astype(float)
            * frame["signed_event_imbalance"].abs().astype(float)
        )
        * frame["buy_sell_event_size_log_ratio"].abs().astype(float)
    )
    tension_baseline = _lagged_clean_quantile(
        topology_tension,
        clean,
        quantile=cfg.tension_quantile,
        window=cfg.baseline_bars,
        minimum=cfg.baseline_min_periods,
    )

    structure_features = {
        "arrival_burst": frame["interarrival_burstiness"].astype(float),
        "notional_concentration": frame["event_notional_hhi"].astype(float),
        "trade_id_span_per_aggregate_event": frame[
            "underlying_trades_per_agg_event"
        ].astype(float),
    }
    structure_marks: dict[str, pd.Series] = {}
    structure_baselines: dict[str, pd.Series] = {}
    for name, values in structure_features.items():
        baseline = _lagged_clean_quantile(
            values,
            clean,
            quantile=cfg.structure_quantile,
            window=cfg.baseline_bars,
            minimum=cfg.baseline_min_periods,
        )
        structure_baselines[name] = baseline
        structure_marks[name] = values.ge(baseline)
    structure_count = sum(mark.astype(np.int8) for mark in structure_marks.values())

    setup = (
        clean
        & frame["agg_trade_count"].fillna(0).ge(cfg.minimum_agg_trade_count)
        & capital_direction.ne(0.0)
        & crowd_direction.ne(0.0)
        & capital_direction.eq(-crowd_direction)
        & immediate_price_direction.eq(crowd_direction)
        & topology_tension.ge(tension_baseline)
        & structure_count.ge(1)
    )

    confirmation = candidate.confirmation_bars
    origin_setup = setup.shift(confirmation, fill_value=False)
    origin_capital_direction = capital_direction.shift(confirmation)
    origin_position = pd.Series(
        np.arange(len(frame), dtype=np.int64) - confirmation,
        index=frame.index,
        dtype=np.int64,
    )
    clean_transition = (
        clean.astype(np.int8)
        .rolling(confirmation + 1, min_periods=confirmation + 1)
        .sum()
        .eq(confirmation + 1)
    )
    post_setup_capital_flow = origin_capital_direction * (
        frame["signed_quote_notional"]
        .astype(float)
        .rolling(confirmation, min_periods=confirmation)
        .sum()
    )
    post_setup_capital_breadth = origin_capital_direction * (
        frame["signed_event_imbalance"]
        .astype(float)
        .rolling(confirmation, min_periods=confirmation)
        .mean()
    )
    capital_price_revelation = origin_capital_direction * np.log(
        frame["close"].astype(float)
        / frame["close"].astype(float).shift(confirmation)
    )
    revealed = (
        origin_setup
        & clean_transition
        & post_setup_capital_flow.gt(0.0)
        & post_setup_capital_breadth.gt(0.0)
        & capital_price_revelation.gt(0.0)
    )
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.loc[revealed] = origin_capital_direction.loc[revealed].astype(np.int8)
    hold = pd.Series(0, index=frame.index, dtype=np.int16)
    hold.loc[revealed] = candidate.hold_bars
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[revealed] = "capital_revelation"

    output = pd.DataFrame(
        {
            "date": frame["date"],
            "capital_direction": capital_direction,
            "crowd_direction": crowd_direction,
            "immediate_price_direction": immediate_price_direction,
            "topology_tension": topology_tension,
            "tension_baseline": tension_baseline,
            "structure_count": structure_count,
            "setup": setup,
            "origin_position": origin_position,
            "post_setup_capital_flow": post_setup_capital_flow,
            "post_setup_capital_breadth": post_setup_capital_breadth,
            "capital_price_revelation": capital_price_revelation,
            "revealed": revealed,
            "side": side,
            "hold_bars": hold,
            "branch": branch,
            "quarantined": frame["quarantined"],
        }
    )
    for name, baseline in structure_baselines.items():
        output[f"{name}_baseline"] = baseline
        output[f"{name}_mark"] = structure_marks[name]
    return output


def nonoverlapping_netf_schedule(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = "2024-01-01",
) -> pd.DataFrame:
    start_timestamp = frame["date"].min() if start is None else pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    period = (
        frame["date"].ge(start_timestamp) & frame["date"].lt(end_timestamp)
    ).to_numpy(bool)
    origin = signal["origin_position"].to_numpy(np.int64)
    valid_origin = np.zeros(len(signal), dtype=bool)
    in_range = (origin >= 0) & (origin < len(frame))
    valid_origin[in_range] = period[origin[in_range]]
    eligible = signal.copy()
    eligible.loc[~valid_origin, "side"] = 0
    return nonoverlapping_schedule(
        eligible,
        frame,
        start=start_timestamp,
        end=end_timestamp,
    )


def _period_count(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> int:
    return len(
        nonoverlapping_netf_schedule(signal, frame, start=start, end=end)
    )


def support_summary(
    signal: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    schedule = nonoverlapping_netf_schedule(signal, frame)
    years = {
        str(year): _period_count(
            signal, frame, f"{year}-01-01", f"{year + 1}-01-01"
        )
        for year in (2020, 2021, 2022, 2023)
    }
    h1 = _period_count(signal, frame, "2023-01-01", "2023-07-01")
    h2 = _period_count(signal, frame, "2023-07-01", "2024-01-01")
    total = len(schedule)
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and all(value >= cfg.minimum_nonoverlap_per_year for value in years.values())
        and h1 >= cfg.minimum_nonoverlap_per_2023_half
        and h2 >= cfg.minimum_nonoverlap_per_2023_half
        and min(long_share, short_share) >= cfg.minimum_side_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_year": years,
        "2023_h1": int(h1),
        "2023_h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "passes_support": bool(passes),
    }


def _selected_support_quantile(trials: list[dict[str, Any]]) -> float | None:
    passing = [
        float(trial["tension_quantile"])
        for trial in trials
        if trial["all_candidates_pass_support"]
    ]
    return max(passing) if passing else None


def run_support(cfg: Config) -> dict[str, Any]:
    frame, source = load_causal_frame(SourceConfig())
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        signal = compute_netf(frame, candidate, cfg)
        setup_values = signal.loc[
            signal["setup"],
            ["topology_tension", "structure_count"],
        ]
        candidates.append(
            {
                "candidate": asdict(candidate),
                "setup_count": int(signal["setup"].sum()),
                "raw_signal_count": int(signal["side"].ne(0).sum()),
                "setup_feature_quantiles": {
                    column: {
                        str(quantile): float(setup_values[column].quantile(quantile))
                        for quantile in (0.01, 0.10, 0.50, 0.90, 0.99)
                    }
                    for column in setup_values.columns
                },
                "support": support_summary(signal, frame, cfg),
            }
        )
    calibration_trials: list[dict[str, Any]] = []
    for quantile in SUPPORT_CALIBRATION_GRID:
        if quantile == cfg.tension_quantile:
            trial_candidates = candidates
        else:
            trial_cfg = replace(cfg, tension_quantile=quantile)
            trial_candidates = []
            for candidate in CANDIDATES:
                signal = compute_netf(frame, candidate, trial_cfg)
                trial_candidates.append(
                    {
                        "candidate": asdict(candidate),
                        "setup_count": int(signal["setup"].sum()),
                        "raw_signal_count": int(signal["side"].ne(0).sum()),
                        "support": support_summary(signal, frame, trial_cfg),
                    }
                )
        calibration_trials.append(
            {
                "tension_quantile": quantile,
                "all_candidates_pass_support": all(
                    item["support"]["passes_support"]
                    for item in trial_candidates
                ),
                "candidates": [
                    {
                        "candidate": item["candidate"],
                        "setup_count": item["setup_count"],
                        "raw_signal_count": item["raw_signal_count"],
                        "support": item["support"],
                    }
                    for item in trial_candidates
                ],
            }
        )
    selected_quantile = _selected_support_quantile(calibration_trials)
    if selected_quantile != cfg.tension_quantile:
        raise ValueError("configured NETF tension quantile violates support stopping rule")

    return {
        "protocol": {
            "name": "NETF — Notional-Event Topology Fracture",
            "support_only": True,
            "outcomes_opened_for_netf": False,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "signal_availability": "completed confirmation bar; enter next 5m open",
            "source_gap_policy": "same verified full-day/missing-slot plus 24-bar quarantine",
        },
        "config": asdict(cfg),
        "source": source,
        "support_calibration": {
            "outcomes_opened_for_netf": False,
            "tested_tension_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_signal_parameters_fixed": True,
            "stopping_rule": "select the highest tested tension quantile where every frozen candidate passes every support floor",
            "selected_tension_quantile": selected_quantile,
            "further_support_repairs_allowed": False,
            "trials": calibration_trials,
        },
        "candidates": candidates,
        "all_candidates_pass_support": all(
            item["support"]["passes_support"] for item in candidates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    cfg = Config(output=args.output)
    result = run_support(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_netf": result["protocol"][
                    "outcomes_opened_for_netf"
                ],
                "all_candidates_pass_support": result[
                    "all_candidates_pass_support"
                ],
                "candidates": [
                    {
                        "name": item["candidate"]["name"],
                        "setup_count": item["setup_count"],
                        "raw_signal_count": item["raw_signal_count"],
                        **item["support"],
                    }
                    for item in result["candidates"]
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
