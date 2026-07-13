from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from training.search_liquidation_scar_field_alpha import (
    SEGMENTS,
    WINDOWS,
    admission,
    build_causal_inputs,
    fit_threshold,
    load_pre2024,
    print_stats,
    rank_key,
    simulate,
    _month_offset,
)
from training.search_positioning_disagreement_alpha import _future_extreme

DepositMode = Literal["actual", "month_offset", "lag12"]


@dataclass
class Scar:
    price: float
    side: int
    born: int
    mass: float
    armed: bool = False


def replay_first_passage(
    inputs: pd.DataFrame,
    dates: pd.Series,
    *,
    contraction_threshold: float,
    max_age: int,
    zone_width: float,
    leave_distance: float = 0.005,
    deposit_mode: DepositMode = "actual",
    require_leave: bool = True,
) -> pd.DataFrame:
    """Consume each scar at its first causal revisit after a required departure."""
    price = inputs["log_price"].to_numpy(float)
    contraction_z = inputs["contraction_z"].to_numpy(float)
    flow_z = inputs["flow_z"].to_numpy(float)
    up_touch = np.zeros(len(inputs), dtype=float)
    down_touch = np.zeros(len(inputs), dtype=float)
    touch_count = np.zeros(len(inputs), dtype=np.int32)
    active_count = np.zeros(len(inputs), dtype=np.int32)
    touch_age_sum = np.zeros(len(inputs), dtype=float)
    scars: list[Scar] = []

    for i in range(len(inputs)):
        current = price[i]
        previous = price[i - 1] if i > 0 else np.nan
        survivors: list[Scar] = []
        if np.isfinite(current):
            for scar in scars:
                age = i - scar.born
                if age > max_age:
                    continue
                if not scar.armed:
                    if not require_leave:
                        scar.armed = age >= 1
                    elif scar.side < 0 and current >= scar.price + leave_distance:
                        scar.armed = True
                    elif scar.side > 0 and current <= scar.price - leave_distance:
                        scar.armed = True
                touched = False
                if scar.armed:
                    if require_leave:
                        if scar.side < 0:
                            touched = (
                                np.isfinite(previous)
                                and previous > scar.price + zone_width
                                and scar.price - zone_width <= current <= scar.price + zone_width
                            )
                        else:
                            touched = (
                                np.isfinite(previous)
                                and previous < scar.price - zone_width
                                and scar.price - zone_width <= current <= scar.price + zone_width
                            )
                    else:
                        touched = scar.price - zone_width <= current <= scar.price + zone_width
                if touched:
                    if scar.side > 0:
                        up_touch[i] += scar.mass
                    else:
                        down_touch[i] += scar.mass
                    touch_count[i] += 1
                    touch_age_sum[i] += age
                else:
                    survivors.append(scar)
        else:
            survivors = scars
        scars = survivors
        active_count[i] = len(scars)

        # Deposit only after all t queries; the event cannot touch itself.
        cz = contraction_z[i]
        fz = flow_z[i]
        if not (np.isfinite(current) and np.isfinite(cz) and np.isfinite(fz)):
            continue
        if cz < contraction_threshold or abs(fz) < 1.0:
            continue
        mass = min(max(cz - contraction_threshold, 0.0), 5.0) * min(max(abs(fz) - 1.0, 0.0), 5.0)
        if mass <= 0.0:
            continue
        deposit_price = current
        if deposit_mode == "lag12":
            if i < 12 or not np.isfinite(price[i - 12]):
                continue
            deposit_price = price[i - 12]
        elif deposit_mode == "month_offset":
            deposit_price += _month_offset(pd.Timestamp(dates.iloc[i])) * 0.001
        scars.append(Scar(price=deposit_price, side=1 if fz > 0.0 else -1, born=i, mass=mass))

    mean_touch_age = np.divide(
        touch_age_sum,
        touch_count,
        out=np.full(len(inputs), np.nan),
        where=touch_count > 0,
    )
    return pd.DataFrame(
        {
            "up_touch_mass": up_touch,
            "down_touch_mass": down_touch,
            "touch_count": touch_count,
            "mean_touch_age": mean_touch_age,
            "active_scars": active_count,
        }
    )


def touch_signals(features: pd.DataFrame, mapping: str, *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    up = pd.to_numeric(features["up_touch_mass"], errors="coerce").fillna(0.0).to_numpy(float)
    down = pd.to_numeric(features["down_touch_mass"], errors="coerce").fillna(0.0).to_numpy(float)
    up_event = (up > 0.0) & (up > down)
    down_event = (down > 0.0) & (down > up)
    if mapping == "fade":
        long_active, short_active = down_event, up_event
    elif mapping == "permeability":
        long_active, short_active = up_event, down_event
    else:
        raise KeyError(mapping)
    if flip:
        long_active, short_active = short_active, long_active
    return long_active, short_active


def main() -> None:
    market, dates = load_pre2024()
    inputs = build_causal_inputs(market)
    holds = (24, 72)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    contraction_thresholds = {
        quantile: fit_threshold(inputs["contraction_z"], dates, quantile)
        for quantile in (0.90, 0.95)
    }
    rows: list[dict[str, Any]] = []
    banks: dict[tuple[float, int, float], pd.DataFrame] = {}
    for event_quantile, max_age, zone_width in itertools.product((0.90, 0.95), (288, 864, 2016), (0.001, 0.002)):
        key = (event_quantile, max_age, zone_width)
        touches = replay_first_passage(
            inputs,
            dates,
            contraction_threshold=contraction_thresholds[event_quantile],
            max_age=max_age,
            zone_width=zone_width,
        )
        banks[key] = touches
        for mapping, hold in itertools.product(("fade", "permeability"), holds):
            long_active, short_active = touch_signals(touches, mapping)
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "event_quantile": event_quantile,
                    "contraction_threshold": contraction_thresholds[event_quantile],
                    "max_age": max_age,
                    "zone_width": zone_width,
                    "leave_distance": 0.005,
                    "mapping": mapping,
                    "hold": hold,
                    "raw_events": int((long_active | short_active).sum()),
                    "median_touch_age": float(touches.loc[touches["touch_count"] > 0, "mean_touch_age"].median()),
                    "admitted": admission(stats),
                    "rank": rank_key(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} eq{row['event_quantile']} age{row['max_age']} zone{row['zone_width']} "
            f"{row['mapping']} h{row['hold']} events={row['raw_events']} medage={row['median_touch_age']:.1f} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    top_key = (top["event_quantile"], top["max_age"], top["zone_width"])
    top_touches = banks[top_key]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    base_long, base_short = touch_signals(top_touches, top["mapping"])
    long_active, short_active = touch_signals(top_touches, top["mapping"], flip=True)
    controls["direction_flip"] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])

    for name, mode, require_leave in (
        ("month_offset_price", "month_offset", True),
        ("lag12_deposit_price", "lag12", True),
        ("no_leave_quarantine", "actual", False),
    ):
        placebo = replay_first_passage(
            inputs,
            dates,
            contraction_threshold=top["contraction_threshold"],
            max_age=top["max_age"],
            zone_width=top["zone_width"],
            deposit_mode=mode,
            require_leave=require_leave,
        )
        long_active, short_active = touch_signals(placebo, top["mapping"])
        controls[name] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(side_bp): simulate(
            market,
            dates,
            base_long,
            base_short,
            top["hold"],
            extremes[top["hold"]],
            side_cost=side_bp / 10_000.0,
        )
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "oi_delay_bars": 1,
            "query_timing": "query and consume prior scars before depositing completed bar t; enter t+1 open",
            "grid_size": 48,
            "leave_distance": 0.005,
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
    }
    Path("results/liquidation_scar_first_passage_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
