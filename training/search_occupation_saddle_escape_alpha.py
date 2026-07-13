"""Search a causal occupation-density saddle-escape alpha.

At each UTC day boundary, the prior 30 days of log-price occupation are frozen
into a dual time/quote-volume density landscape.  Deep local minima between
adjacent high-occupation modes are price saddles.  A completed hourly close
crossing a frozen saddle is interpreted as escape from one liquidity basin
toward the next; the policy continues in the crossing direction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import (
    SEGMENTS,
    admission,
    load_pre2024,
    rank_key,
    simulate,
)
from training.search_positioning_disagreement_alpha import _future_extreme

LOOKBACK_BARS = 30 * 288
PROFILE_BINS = 64
MIN_PROFILE_ROWS = LOOKBACK_BARS // 2
DECISION_MINUTE = 55
HOUR_BARS = 12
MAX_SADDLE_RATIO = 0.50
MIN_MODE_SEPARATION = 3
HOLDS = (72, 144)
SMOOTH_KERNEL = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0


@dataclass(frozen=True)
class Saddle:
    index: int
    ratio: float
    left_mode: int
    right_mode: int


@dataclass(frozen=True)
class FrozenProfile:
    edges: np.ndarray
    density: np.ndarray
    saddles: tuple[Saddle, ...]


def occupation_profile(
    log_price: np.ndarray,
    quote_volume: np.ndarray,
    *,
    mode: str = "joint",
) -> FrozenProfile | None:
    price = np.asarray(log_price, dtype=float)
    quote = np.asarray(quote_volume, dtype=float)
    finite = np.isfinite(price) & np.isfinite(quote) & (quote >= 0.0)
    price = price[finite]
    quote = quote[finite]
    if len(price) < MIN_PROFILE_ROWS or not (price.max() > price.min()):
        return None
    edges = np.linspace(price.min(), price.max(), PROFILE_BINS + 1)
    time_count = np.histogram(price, edges)[0].astype(float)
    weights = quote[::-1] if mode == "reversed_volume" else quote
    volume_count = np.histogram(price, edges, weights=weights)[0].astype(float)
    time_share = time_count / max(time_count.sum(), 1.0)
    volume_share = volume_count / max(volume_count.sum(), 1.0)
    if mode == "joint" or mode == "reversed_volume":
        raw_density = np.sqrt(time_share * volume_share)
    elif mode == "time_only":
        raw_density = time_share
    elif mode == "volume_only":
        raw_density = volume_share
    else:
        raise ValueError(f"unknown occupation mode: {mode}")
    density = np.convolve(raw_density, SMOOTH_KERNEL, mode="same")
    peaks = np.flatnonzero(
        (density[1:-1] >= density[:-2]) & (density[1:-1] > density[2:])
    ) + 1
    saddles: list[Saddle] = []
    for left_mode, right_mode in zip(peaks[:-1], peaks[1:], strict=False):
        if right_mode - left_mode < MIN_MODE_SEPARATION:
            continue
        saddle_index = int(left_mode + np.argmin(density[left_mode : right_mode + 1]))
        mode_floor = min(density[left_mode], density[right_mode])
        if mode_floor <= 0.0:
            continue
        ratio = float(density[saddle_index] / mode_floor)
        if ratio <= MAX_SADDLE_RATIO:
            saddles.append(
                Saddle(
                    index=saddle_index,
                    ratio=ratio,
                    left_mode=int(left_mode),
                    right_mode=int(right_mode),
                )
            )
    return FrozenProfile(edges=edges, density=density, saddles=tuple(saddles))


def build_saddle_state(
    market: pd.DataFrame,
    dates: pd.Series,
    *,
    mode: str = "joint",
) -> pd.DataFrame:
    log_price = np.log(
        pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0)
    ).to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0).to_numpy(float)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    decision_positions = np.flatnonzero(dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool))
    side = np.zeros(len(market), dtype=np.int8)
    depth = np.full(len(market), np.nan, dtype=float)
    saddle_count = np.zeros(len(market), dtype=np.int16)
    profile_day = None
    profile: FrozenProfile | None = None
    for position in decision_positions:
        day = dates.iloc[position].normalize()
        if profile_day is None or day != profile_day:
            profile_day = day
            day_start = int(np.searchsorted(date_values, np.datetime64(day), side="left"))
            lookback_start = day_start - LOOKBACK_BARS
            profile = (
                occupation_profile(
                    log_price[lookback_start:day_start],
                    quote[lookback_start:day_start],
                    mode=mode,
                )
                if lookback_start >= 0
                else None
            )
        if profile is None or position < HOUR_BARS:
            continue
        saddle_count[position] = len(profile.saddles)
        previous_price = log_price[position - HOUR_BARS]
        current_price = log_price[position]
        if (
            not np.isfinite(previous_price)
            or not np.isfinite(current_price)
            or previous_price < profile.edges[0]
            or previous_price > profile.edges[-1]
            or current_price < profile.edges[0]
            or current_price > profile.edges[-1]
        ):
            continue
        previous_bin = int(
            np.clip(np.searchsorted(profile.edges, previous_price, side="right") - 1, 0, PROFILE_BINS - 1)
        )
        current_bin = int(
            np.clip(np.searchsorted(profile.edges, current_price, side="right") - 1, 0, PROFILE_BINS - 1)
        )
        crossed: list[tuple[int, float]] = []
        for saddle in profile.saddles:
            if previous_bin < saddle.index <= current_bin:
                crossed.append((1, 1.0 - saddle.ratio))
            elif current_bin <= saddle.index < previous_bin:
                crossed.append((-1, 1.0 - saddle.ratio))
        if crossed and all(direction == crossed[0][0] for direction, _ in crossed):
            side[position] = crossed[0][0]
            depth[position] = max(value for _, value in crossed)
    return pd.DataFrame(
        {
            "side": side,
            "barrier_depth": depth,
            "saddle_count": saddle_count,
            "decision": dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool),
        }
    )


def policy_masks(state: pd.DataFrame, *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    side = state["side"].to_numpy(np.int8)
    if flip:
        side = -side
    active = state["decision"].to_numpy(bool) & np.isfinite(
        state["barrier_depth"].to_numpy(float)
    )
    return active & (side > 0), active & (side < 0)


def lag_boolean(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    return np.r_[np.zeros(bars, dtype=bool), values[:-bars]]


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for name in ("fit", "select_2023", *SEGMENTS):
        value = stats[name]
        print(
            name,
            f"ret={value['return_pct']:.2f}",
            f"cagr={value['cagr_pct']:.2f}",
            f"mdd={value['strict_mdd_pct']:.2f}",
            f"ratio={value['ratio']:.2f}",
            f"n={value['trades']}",
            f"L/S={value['longs']}/{value['shorts']}",
        )


def main() -> None:
    market, dates = load_pre2024()
    state = build_saddle_state(market, dates)
    long_active, short_active = policy_masks(state)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    for hold in HOLDS:
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "hold": hold,
                "raw_signals": int((long_active | short_active).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    for index, row in enumerate(rows, 1):
        print_stats(f"RANK {index} hold{row['hold']} rank={row['rank']}", row["stats"])

    top = rows[0]
    flip_long, flip_short = policy_masks(state, flip=True)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]
        ),
        "signal_delay_1h": simulate(
            market,
            dates,
            lag_boolean(long_active, HOUR_BARS),
            lag_boolean(short_active, HOUR_BARS),
            top["hold"],
            extremes[top["hold"]],
        ),
        "signal_delay_24h": simulate(
            market,
            dates,
            lag_boolean(long_active, 288),
            lag_boolean(short_active, 288),
            top["hold"],
            extremes[top["hold"]],
        ),
        "signal_delay_7d": simulate(
            market,
            dates,
            lag_boolean(long_active, 2016),
            lag_boolean(short_active, 2016),
            top["hold"],
            extremes[top["hold"]],
        ),
    }
    control_signal_summary: dict[str, dict[str, int]] = {}
    for mode in ("time_only", "volume_only", "reversed_volume"):
        control_state = build_saddle_state(market, dates, mode=mode)
        control_long, control_short = policy_masks(control_state)
        controls[mode] = simulate(
            market, dates, control_long, control_short, top["hold"], extremes[top["hold"]]
        )
        control_signal_summary[mode] = {
            "raw": int((control_long | control_short).sum()),
            "long": int(control_long.sum()),
            "short": int(control_short.sum()),
        }
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            long_active,
            short_active,
            top["hold"],
            extremes[top["hold"]],
            side_cost=bp / 10_000,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    output = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "mechanism": "daily-frozen prior-30d joint time/quote-volume occupation density; continue first completed-hour crossing of a deep saddle between adjacent price modes",
            "lookback_bars": LOOKBACK_BARS,
            "profile_bins": PROFILE_BINS,
            "max_saddle_ratio": MAX_SADDLE_RATIO,
            "minimum_mode_separation_bins": MIN_MODE_SEPARATION,
            "grid_size": len(rows),
            "grid": "one frozen joint occupation-saddle signal x 6h/12h holds",
            "entry": "completed minute-55 crossing enters next minute-00 open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "all pre-2024 rows are exploratory; 2023 is inspected internal selection and 2024+ remained sealed",
            "support_only_design_probe": (
                "before outcome evaluation, fixed 30d/64-bin/depth<=0.5 construction "
                "produced 277 fit and 157 selection raw signals using an index-relative "
                "day-boundary probe; the boundary was corrected to UTC timestamps before "
                "return evaluation, yielding 467 final pre-2024 raw signals"
            ),
        },
        "state_summary": {
            "raw_signals": int((long_active | short_active).sum()),
            "raw_long_short": [int(long_active.sum()), int(short_active.sum())],
            "decision_profiles_with_saddles": int((state["saddle_count"] > 0).sum()),
            "median_saddles_when_available": float(
                state.loc[state["saddle_count"] > 0, "saddle_count"].median()
            ),
        },
        "rows": rows,
        "controls": controls,
        "control_signal_summary": control_signal_summary,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/occupation_saddle_escape_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
