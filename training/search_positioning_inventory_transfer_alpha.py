"""Test whether OI conservation routes aged positioning-disagreement resolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme
from training.search_positioning_lifecycle_hazard_alpha import (
    WINDOWS,
    admission,
    build_disagreement_states,
    load_pre2024,
    positioning_valid_mask,
    print_stats,
    rank_key,
    simulate,
)

DISAGREEMENT = "top_position_minus_global"
MIN_AGE = 432
HOLDS = (72, 216)


def inventory_transfer_signals(
    disagreement_z: np.ndarray,
    valid: np.ndarray,
    open_interest: np.ndarray,
    *,
    min_age: int = MIN_AGE,
    entry_z: float = 1.5,
    flip: bool = False,
    ignore_oi: bool = False,
    invert_oi: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Route the first aged zero-cross from causal episode-start OI.

    If OI survives the disagreement episode, trade in the resolution direction;
    if OI contracts, treat the resolution as completed liquidation and fade it.
    """
    disagreement_z = np.asarray(disagreement_z, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    open_interest = np.asarray(open_interest, dtype=float)
    if not (len(disagreement_z) == len(valid) == len(open_interest)):
        raise ValueError("signal inputs must have equal length")

    long_active = np.zeros(len(disagreement_z), dtype=bool)
    short_active = np.zeros(len(disagreement_z), dtype=bool)
    episode_start_index = np.full(len(disagreement_z), -1, dtype=np.int64)
    oi_log_change = np.full(len(disagreement_z), np.nan)
    inventory_conserved = np.zeros(len(disagreement_z), dtype=bool)

    active = False
    episode_side = 0
    age = 0
    start = -1
    for position, value in enumerate(disagreement_z):
        usable = (
            valid[position]
            and np.isfinite(value)
            and np.isfinite(open_interest[position])
            and open_interest[position] > 0.0
        )
        if not usable:
            active = False
            episode_side = 0
            age = 0
            start = -1
            continue

        current_side = int(np.sign(value))
        if not active:
            if abs(value) >= entry_z:
                active = True
                episode_side = current_side
                age = 1
                start = position
            continue

        age += 1
        crossed = current_side != 0 and current_side != episode_side
        if crossed and age >= min_age:
            oi_change = float(np.log(open_interest[position] / open_interest[start]))
            conserved = oi_change >= 0.0
            routed_conserved = not conserved if invert_oi else conserved
            side = -episode_side if (ignore_oi or routed_conserved) else episode_side
            if flip:
                side = -side
            long_active[position] = side > 0
            short_active[position] = side < 0
            episode_start_index[position] = start
            oi_log_change[position] = oi_change
            inventory_conserved[position] = conserved

        if crossed:
            if abs(value) >= entry_z:
                active = True
                episode_side = current_side
                age = 1
                start = position
            else:
                active = False
                episode_side = 0
                age = 0
                start = -1

    return long_active, short_active, {
        "episode_start_index": episode_start_index,
        "oi_log_change": oi_log_change,
        "inventory_conserved": inventory_conserved,
    }


def main() -> None:
    market, dates = load_pre2024()
    states = build_disagreement_states(market)
    valid = positioning_valid_mask(
        dates,
        market["positioning_available"].to_numpy(bool),
    )
    disagreement_z = states[DISAGREEMENT].to_numpy(float)
    open_interest = pd.to_numeric(
        market["sum_open_interest"], errors="coerce"
    ).to_numpy(float)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }

    long_active, short_active, diagnostics = inventory_transfer_signals(
        disagreement_z,
        valid,
        open_interest,
    )
    rows: list[dict[str, Any]] = []
    for hold in HOLDS:
        stats = simulate(
            market,
            dates,
            long_active,
            short_active,
            hold,
            extremes[hold],
        )
        rows.append(
            {
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
            f"RANK {index} inventory-transfer h{row['hold']} sig={row['signals']} "
            f"rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, extra in (
        ("direction_flip", {"flip": True}),
        ("ignore_oi_parent_lifecycle", {"ignore_oi": True}),
        ("invert_oi_route", {"invert_oi": True}),
    ):
        control_long, control_short, _ = inventory_transfer_signals(
            disagreement_z,
            valid,
            open_interest,
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
        print_stats("CONTROL " + name, controls[name])

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

    signal_positions = np.flatnonzero(long_active | short_active)
    oi_changes = diagnostics["oi_log_change"][signal_positions]
    event_summary = {
        "events": int(len(signal_positions)),
        "inventory_conserved": int(np.sum(oi_changes >= 0.0)),
        "inventory_contracted": int(np.sum(oi_changes < 0.0)),
        "median_oi_log_change": float(np.nanmedian(oi_changes)),
    }
    print("event_summary", event_summary)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "source_delay": "Binance UM metrics delayed by one complete 5m bar",
            "fit": WINDOWS["fit"],
            "quarantine": "2022 metrics state forcibly invalidated and reset",
            "grid_size": 2,
            "episode": "fixed parent: top-position-minus-global |z|>=1.5, age>=36h, first zero-cross",
            "direction": "OI nondecrease follows resolution; OI decline fades resolution",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "event_summary": event_summary,
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path(
        "results/positioning_inventory_transfer_alpha_scan_2026-07-13.json"
    ).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
