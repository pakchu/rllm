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


def chirp_signals(
    event_long: np.ndarray,
    event_short: np.ndarray,
    *,
    max_gap_bars: int,
    branch: str,
    acceleration_ratio: float = 0.75,
    exhaustion_ratio: float = 1.50,
    require_clean_triplet: bool = True,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Trade causal compression/expansion of same-direction event intervals."""
    event_long = np.asarray(event_long, dtype=bool)
    event_short = np.asarray(event_short, dtype=bool)
    if event_long.shape != event_short.shape or np.any(event_long & event_short):
        raise ValueError("event arrays must be aligned and unambiguous")
    long_active = np.zeros_like(event_long)
    short_active = np.zeros_like(event_short)
    gap_ratio = np.full(len(event_long), np.nan)
    clean = np.zeros(len(event_long), dtype=bool)
    last_two: dict[int, list[int]] = {1: [], -1: []}
    last_opposite: dict[int, int] = {1: -1, -1: -1}
    event_positions = np.flatnonzero(event_long | event_short)
    for position in event_positions:
        direction = 1 if event_long[position] else -1
        opposite = -direction
        history = last_two[direction]
        if len(history) == 2:
            previous_two, previous_one = history
            old_gap = previous_one - previous_two
            new_gap = int(position - previous_one)
            gap_ratio[position] = new_gap / old_gap if old_gap > 0 else np.nan
            clean[position] = last_opposite[direction] < previous_two
            usable = (
                old_gap > 0
                and new_gap > 0
                and old_gap <= max_gap_bars
                and new_gap <= max_gap_bars
                and (clean[position] or not require_clean_triplet)
            )
            if branch == "acceleration_continuation":
                selected = usable and gap_ratio[position] <= acceleration_ratio
                side = direction
            elif branch == "deceleration_reversal":
                selected = usable and gap_ratio[position] >= exhaustion_ratio
                side = -direction
            elif branch == "triplet_continuation":
                selected = usable
                side = direction
            else:
                raise KeyError(branch)
            if selected:
                if flip:
                    side = -side
                long_active[position] = side > 0
                short_active[position] = side < 0
        history.append(int(position))
        if len(history) > 2:
            del history[0]
        last_opposite[opposite] = int(position)
    return long_active, short_active, {"gap_ratio": gap_ratio, "clean_triplet": clean}


def base_events(
    market: pd.DataFrame,
    dates: pd.Series,
    profile: tuple[int, int, int],
    *,
    order_swap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    features = build_profile_features(market, profile)
    thresholds = fit_policy_thresholds(features, dates, 0.95)
    long_active, short_active, _ = sequence_signals(
        features,
        thresholds,
        "continuation",
        order_swap=order_swap,
    )
    return long_active, short_active


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
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
            hold_bars=hold,
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
        stats["fit"]["trades"] >= 50
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0
        and stats["fit"]["ratio"] >= 3
        and stats["select_2023"]["return_pct"] > 0
        and stats["select_2023"]["ratio"] >= 3
        and stats["select_2023_h1"]["return_pct"] >= 0
        and stats["select_2023_h2"]["return_pct"] >= 0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 50
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    core = [stats[window]["ratio"] for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        enough,
        min(core) > 0,
        sum(stats[window]["return_pct"] > 0 for window in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        stats_window = stats[window]
        print(
            window,
            f"ret={stats_window['return_pct']:.2f}",
            f"cagr={stats_window['cagr_pct']:.2f}",
            f"mdd={stats_window['strict_mdd_pct']:.2f}",
            f"ratio={stats_window['ratio']:.2f}",
            f"n={stats_window['trades']}",
            f"L/S={stats_window['longs']}/{stats_window['shorts']}",
        )


def main() -> None:
    market, dates = load_pre2024()
    holds = (144, 288)
    extremes = {
        hold: (
            _future_extreme(market.low.to_numpy(float), hold, "min"),
            _future_extreme(market.high.to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    banks: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    grid = itertools.product(
        PROFILES,
        (144, 288),
        ("acceleration_continuation", "deceleration_reversal"),
        holds,
    )
    for profile, max_gap, branch, hold in grid:
        event_long, event_short = banks.setdefault(profile, base_events(market, dates, profile))
        long_active, short_active, diagnostics = chirp_signals(
            event_long,
            event_short,
            max_gap_bars=max_gap,
            branch=branch,
        )
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        active = long_active | short_active
        rows.append(
            {
                "profile": list(profile),
                "tail_quantile": 0.95,
                "max_gap": max_gap,
                "branch": branch,
                "hold": hold,
                "signals": int(active.sum()),
                "median_gap_ratio": float(np.nanmedian(diagnostics["gap_ratio"][active])) if active.any() else None,
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} p{row['profile']} g{row['max_gap']} {row['branch']} h{row['hold']} sig={row['signals']} ratio={row['median_gap_ratio']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    profile = tuple(top["profile"])
    event_long, event_short = banks[profile]
    kwargs = {"max_gap_bars": top["max_gap"], "branch": top["branch"]}
    long_active, short_active, _ = chirp_signals(event_long, event_short, **kwargs)
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}

    flip_long, flip_short, _ = chirp_signals(event_long, event_short, flip=True, **kwargs)
    controls["direction_flip"] = simulate(market, dates, flip_long, flip_short, hold, extremes[hold])
    triplet_long, triplet_short, _ = chirp_signals(
        event_long,
        event_short,
        max_gap_bars=top["max_gap"],
        branch="triplet_continuation",
    )
    controls["ignore_cadence"] = simulate(market, dates, triplet_long, triplet_short, hold, extremes[hold])
    no_quarantine_long, no_quarantine_short, _ = chirp_signals(
        event_long,
        event_short,
        require_clean_triplet=False,
        **kwargs,
    )
    controls["remove_opposite_quarantine"] = simulate(
        market,
        dates,
        no_quarantine_long,
        no_quarantine_short,
        hold,
        extremes[hold],
    )
    swapped_event_long, swapped_event_short = base_events(market, dates, profile, order_swap=True)
    swapped_long, swapped_short, _ = chirp_signals(swapped_event_long, swapped_event_short, **kwargs)
    controls["phase_order_swap"] = simulate(market, dates, swapped_long, swapped_short, hold, extremes[hold])
    lag = sum(profile)
    lag_long = np.r_[np.zeros(lag, dtype=bool), long_active[:-lag]]
    lag_short = np.r_[np.zeros(lag, dtype=bool), short_active[:-lag]]
    controls["chirp_lag"] = simulate(market, dates, lag_long, lag_short, hold, extremes[hold])
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            long_active,
            short_active,
            hold,
            extremes[hold],
            side_cost=side_bp / 10_000,
        )
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "parent": "frozen q95 continuation",
            "grid_size": 48,
            "acceleration_ratio": 0.75,
            "exhaustion_ratio": 1.5,
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            top["prelim_admitted"] and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path("results/orderflow_trophic_chirp_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
