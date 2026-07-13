from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_nested_barrier_witness_alpha import (
    HORIZONS,
    admission,
    build_barrier_bank,
    print_stats,
    rank_key,
    simulate,
)
from training.search_orderflow_trophic_succession_alpha import load_pre2024
from training.search_positioning_disagreement_alpha import _future_extreme


def candle_close_location(market: pd.DataFrame) -> np.ndarray:
    """Completed-bar close location in [-1, 1]; zero-range bars remain unknown."""
    close = market["close"].to_numpy(float)
    high = market["high"].to_numpy(float)
    low = market["low"].to_numpy(float)
    span = high - low
    output = np.full(len(market), np.nan)
    valid = np.isfinite(close) & np.isfinite(high) & np.isfinite(low) & (span > 0.0)
    output[valid] = 2.0 * (close[valid] - low[valid]) / span[valid] - 1.0
    return output


def barrier_constitution_signals(
    market: pd.DataFrame,
    bank: dict[Any, Any],
    *,
    min_coalescence: int,
    touch_width: float,
    clv_threshold: float = 0.5,
    work_low: float = 0.75,
    work_high: float = 1.25,
    max_origin_separation: int = 3,
    flip: bool = False,
    invert_origin: bool = False,
    ignore_origin: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Route a nested barrier by its creation candle and current/origin work ratio.

    A directional close at creation plus depleted work on revisit is interpreted as
    unfinished discovery (continuation). A rejected creation candle plus reinforced
    work on revisit is interpreted as defended inventory (fade).
    """
    close = market["close"].to_numpy(float)
    origin_clv = candle_close_location(market)
    buy_work = bank["buy_work"]
    sell_work = bank["sell_work"]
    long_state = np.zeros(len(market), dtype=bool)
    short_state = np.zeros(len(market), dtype=bool)
    selected_origin_clv = np.full(len(market), np.nan)
    selected_work_ratio = np.full(len(market), np.nan)
    selected_coalescence = np.zeros(len(market), dtype=np.int8)

    for position in range(len(market)):
        for barrier_side in ("high", "low"):
            touched: list[tuple[int, int, float]] = []
            for horizon in HORIZONS:
                witness_index = int(bank[horizon][f"{barrier_side}_index"][position])
                level = float(bank[horizon][f"{barrier_side}_price"][position])
                if (
                    witness_index >= 0
                    and np.isfinite(level)
                    and abs(close[position] / level - 1.0) <= touch_width
                ):
                    touched.append((horizon, witness_index, level))
            if len(touched) < min_coalescence:
                continue
            witness_indices = [witness_index for _, witness_index, _ in touched]
            if max(witness_indices) - min(witness_indices) > max_origin_separation:
                continue

            # The longest touched scale owns the barrier's creation constitution.
            _, witness_index, _ = max(touched, key=lambda item: item[0])
            witness_clv = origin_clv[witness_index]
            if invert_origin:
                witness_clv = -witness_clv
            levels = [level for _, _, level in touched]

            if barrier_side == "high":
                origin_work = buy_work[witness_index]
                current_work = buy_work[position]
                not_closed_through = close[position] <= max(levels)
            else:
                origin_work = sell_work[witness_index]
                current_work = sell_work[position]
                not_closed_through = close[position] >= min(levels)
            work_ratio = (
                current_work / origin_work
                if np.isfinite(origin_work)
                and origin_work > 1e-4
                and np.isfinite(current_work)
                else np.nan
            )
            if not not_closed_through or not np.isfinite(work_ratio):
                continue

            if barrier_side == "high":
                continuation = work_ratio <= work_low and (
                    ignore_origin or witness_clv >= clv_threshold
                )
                fade = work_ratio >= work_high and (
                    ignore_origin or witness_clv <= -clv_threshold
                )
                if continuation:
                    long_state[position] = True
                elif fade:
                    short_state[position] = True
            else:
                continuation = work_ratio <= work_low and (
                    ignore_origin or witness_clv <= -clv_threshold
                )
                fade = work_ratio >= work_high and (
                    ignore_origin or witness_clv >= clv_threshold
                )
                if continuation:
                    short_state[position] = True
                elif fade:
                    long_state[position] = True

            if continuation or fade:
                selected_origin_clv[position] = witness_clv
                selected_work_ratio[position] = work_ratio
                selected_coalescence[position] = len(touched)

    long_onset = long_state & ~np.r_[False, long_state[:-1]] & ~short_state
    short_onset = short_state & ~np.r_[False, short_state[:-1]] & ~long_state
    if flip:
        long_onset, short_onset = short_onset, long_onset
    return long_onset, short_onset, {
        "origin_clv": selected_origin_clv,
        "work_ratio": selected_work_ratio,
        "coalescence": selected_coalescence,
    }


def main() -> None:
    market, dates = load_pre2024()
    bank = build_barrier_bank(market)
    holds = (72, 144)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    signal_bank: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for min_coalescence, touch_width in itertools.product((2, 3), (0.001, 0.002)):
        long_active, short_active, _ = barrier_constitution_signals(
            market,
            bank,
            min_coalescence=min_coalescence,
            touch_width=touch_width,
        )
        signal_bank[(min_coalescence, touch_width)] = (long_active, short_active)
        for hold in holds:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "min_coalescence": min_coalescence,
                    "touch_width": touch_width,
                    "clv_threshold": 0.5,
                    "hold": hold,
                    "signals": int((long_active | short_active).sum()),
                    "rank": rank_key(stats),
                    "prelim_admitted": admission(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} c{row['min_coalescence']} w{row['touch_width']} "
            f"h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    signal_kwargs = {
        "min_coalescence": top["min_coalescence"],
        "touch_width": top["touch_width"],
    }
    long_active, short_active = signal_bank[
        (top["min_coalescence"], top["touch_width"])
    ]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, extra in (
        ("direction_flip", {"flip": True}),
        ("invert_origin", {"invert_origin": True}),
        ("ignore_origin", {"ignore_origin": True}),
    ):
        control_long, control_short, _ = barrier_constitution_signals(
            market,
            bank,
            **signal_kwargs,
            **extra,
        )
        controls[name] = simulate(
            market,
            dates,
            control_long,
            control_short,
            hold,
            extremes[hold],
        )
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
            "grid_size": 8,
            "horizons": HORIZONS,
            "origin_clv_threshold": 0.5,
            "work_ratio": {"depleted": 0.75, "reinforced": 1.25},
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
    Path("results/barrier_constitution_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
