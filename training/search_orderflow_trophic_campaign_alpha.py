from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_orderflow_trophic_succession_alpha import (
    PROFILES,
    SEGMENTS,
    WINDOWS,
    build_profile_features,
    fit_policy_thresholds,
    load_pre2024,
    sequence_signals,
)
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop


def campaign_signals(
    event_long: np.ndarray,
    event_short: np.ndarray,
    *,
    lookback_bars: int,
    min_same_events: int,
    max_opposite_events: int = 1,
    cooldown_bars: int | None = None,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Confirm repeated same-direction events using current and prior bars only."""
    if lookback_bars <= 0 or min_same_events <= 0:
        raise ValueError("lookback and minimum event count must be positive")
    event_long = np.asarray(event_long, dtype=bool)
    event_short = np.asarray(event_short, dtype=bool)
    if event_long.shape != event_short.shape:
        raise ValueError("event arrays must have the same shape")
    if np.any(event_long & event_short):
        raise ValueError("an event cannot have both directions")
    long_count = (
        pd.Series(event_long.astype(np.int16))
        .rolling(lookback_bars, min_periods=1)
        .sum()
        .to_numpy(dtype=np.int16)
    )
    short_count = (
        pd.Series(event_short.astype(np.int16))
        .rolling(lookback_bars, min_periods=1)
        .sum()
        .to_numpy(dtype=np.int16)
    )
    eligible_long = event_long & (long_count >= min_same_events) & (short_count <= max_opposite_events)
    eligible_short = event_short & (short_count >= min_same_events) & (long_count <= max_opposite_events)
    cooldown = lookback_bars if cooldown_bars is None else cooldown_bars
    confirmed_long = np.zeros_like(event_long)
    confirmed_short = np.zeros_like(event_short)
    next_allowed = 0
    for index in np.flatnonzero(eligible_long | eligible_short):
        if index < next_allowed:
            continue
        confirmed_long[index] = bool(eligible_long[index])
        confirmed_short[index] = bool(eligible_short[index])
        next_allowed = int(index + cooldown)
    if flip:
        confirmed_long, confirmed_short = confirmed_short, confirmed_long
    return confirmed_long, confirmed_short, {
        "long_count": long_count,
        "short_count": short_count,
        "eligible_long": eligible_long,
        "eligible_short": eligible_short,
    }


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold_bars: int,
    extremes: tuple[np.ndarray, np.ndarray],
    side_cost: float = 0.0006,
) -> dict[str, dict[str, Any]]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold_bars,
            stride_bars=1,
            leverage=0.5,
            fee_rate=side_cost,
            slippage_rate=0.0,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    enough = (
        stats["fit"]["trades"] >= 60
        and stats["select_2023"]["trades"] >= 18
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 10
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] >= 0.0
        and stats["select_2023_h2"]["return_pct"] >= 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = stats["fit"]["trades"] >= 60 and stats["select_2023"]["trades"] >= 18
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    positive_segments = sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS)
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        positive_segments,
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        value = stats[window]
        print(window, f"ret={value['return_pct']:.2f}", f"cagr={value['cagr_pct']:.2f}", f"mdd={value['strict_mdd_pct']:.2f}", f"ratio={value['ratio']:.2f}", f"n={value['trades']}", f"L/S={value['longs']}/{value['shorts']}")


def base_events(
    market: pd.DataFrame,
    dates: pd.Series,
    profile: tuple[int, int, int],
    *,
    order_swap: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]:
    features = build_profile_features(market, profile)
    thresholds = fit_policy_thresholds(features, dates, 0.95)
    event_long, event_short, _ = sequence_signals(
        features,
        thresholds,
        "continuation",
        order_swap=order_swap,
    )
    return event_long, event_short, features, thresholds


def main() -> None:
    market, dates = load_pre2024()
    holds = (144, 288)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    banks: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, float]]] = {}
    for profile in PROFILES:
        banks[profile] = base_events(market, dates, profile)
        event_long, event_short, _, thresholds = banks[profile]
        for lookback, min_events, hold in itertools.product((72, 144), (2, 3), holds):
            long_active, short_active, diag = campaign_signals(
                event_long,
                event_short,
                lookback_bars=lookback,
                min_same_events=min_events,
                max_opposite_events=1,
            )
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "profile": list(profile),
                    "tail_quantile": 0.95,
                    "branch": "continuation_campaign",
                    "lookback": lookback,
                    "min_same_events": min_events,
                    "max_opposite_events": 1,
                    "cooldown": lookback,
                    "hold": hold,
                    "thresholds": thresholds,
                    "eligible_events": int((diag["eligible_long"] | diag["eligible_short"]).sum()),
                    "campaign_events": int((long_active | short_active).sum()),
                    "prelim_admitted": admission(stats),
                    "rank": rank_key(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "prelim_admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(f"RANK {index} profile={row['profile']} lb{row['lookback']} k{row['min_same_events']} h{row['hold']} campaigns={row['campaign_events']} rank={row['rank']}", row["stats"])

    top = rows[0]
    profile = tuple(top["profile"])
    event_long, event_short, _, thresholds = banks[profile]
    kwargs = {
        "lookback_bars": top["lookback"],
        "min_same_events": top["min_same_events"],
        "max_opposite_events": top["max_opposite_events"],
    }
    base_long, base_short, _ = campaign_signals(event_long, event_short, **kwargs)
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    controls["direction_flip"] = simulate(market, dates, base_short, base_long, top["hold"], extremes[top["hold"]])
    k1_long, k1_short, _ = campaign_signals(event_long, event_short, **{**kwargs, "min_same_events": 1})
    controls["single_event_cooldown"] = simulate(market, dates, k1_long, k1_short, top["hold"], extremes[top["hold"]])
    swapped_long_events, swapped_short_events, _, _ = base_events(market, dates, profile, order_swap=True)
    swapped_long, swapped_short, _ = campaign_signals(swapped_long_events, swapped_short_events, **kwargs)
    controls["phase_order_swap"] = simulate(market, dates, swapped_long, swapped_short, top["hold"], extremes[top["hold"]])
    lag = sum(profile)
    lag_long = np.r_[np.zeros(lag, dtype=bool), base_long[:-lag]]
    lag_short = np.r_[np.zeros(lag, dtype=bool), base_short[:-lag]]
    controls["campaign_lag"] = simulate(market, dates, lag_long, lag_short, top["hold"], extremes[top["hold"]])
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(market, dates, base_long, base_short, top["hold"], extremes[top["hold"]], side_cost=side_bp / 10_000.0)
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    control_pass = not any(admission(stats) for stats in controls.values())
    final_admitted = bool(top["prelim_admitted"] and control_pass)
    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "parent_usage": "frozen q95 continuation role sequence; no parent threshold retuning",
            "grid_size": 48,
            "campaign": "current event confirms k same-direction events within trailing lookback, <=1 opposite event, then one-lookback global cooldown",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "top_control_pass": control_pass,
        "final_admitted": final_admitted,
    }
    Path("results/orderflow_trophic_campaign_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
