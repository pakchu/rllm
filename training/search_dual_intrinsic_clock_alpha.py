"""Search a causal price/flow intrinsic-clock impact-elasticity alpha.

Adaptive directional-change state machines create separate event clocks for
log price and cumulative normalized aggressive taker flow.  A faster flow
clock with a flat price path is interpreted as absorption and faded; a faster
price clock with non-opposing flow is interpreted as a liquidity vacuum and
continued.  Only the first completed hourly entry into either state may trade.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
CUTOFF = "2024-01-01"
WINDOWS = {
    "fit": ("2020-10-15", "2023-01-01"),
    "fit_2020_h2": ("2020-10-15", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "fit_2022_h1": ("2022-01-01", "2022-07-01"),
    "fit_2022_h2": ("2022-07-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = (
    "fit_2020_h2",
    "fit_2021_h1",
    "fit_2021_h2",
    "fit_2022_h1",
    "fit_2022_h2",
    "select_2023_h1",
    "select_2023_h2",
)
DC_WIDTHS = (0.75, 1.0, 1.5)
CLOCK_WINDOWS = (72, 144, 288)
DOMINANCE_RATIOS = (1.5, 2.0)
DECISION_MINUTE = 55
DECISION_STRIDE = 12
SCALE_WINDOW = 2016
SCALE_MIN_PERIODS = 1008
MIN_FAST_EVENTS = 3


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("dual-clock search requires a complete 5-minute grid")
    return market, dates


def _prior_hourly_scale(values: pd.Series) -> np.ndarray:
    hourly_change = pd.to_numeric(values, errors="coerce").diff(DECISION_STRIDE)
    return (
        hourly_change.shift(1)
        .rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
        .to_numpy(float)
    )


def build_paths(market: pd.DataFrame) -> dict[str, np.ndarray]:
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0)
    buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    prior_hour_quote = quote.shift(1).rolling(DECISION_STRIDE, min_periods=DECISION_STRIDE).sum()
    flow_increment = (2.0 * buy - quote) / prior_hour_quote.replace(0.0, np.nan)
    flow_path = pd.Series(np.cumsum(np.nan_to_num(flow_increment.to_numpy(float))), index=market.index)
    return {
        "log_price": log_price.to_numpy(float),
        "flow_increment": flow_increment.to_numpy(float),
        "flow_path": flow_path.to_numpy(float),
        "price_scale": _prior_hourly_scale(log_price),
        "flow_scale": _prior_hourly_scale(flow_path),
    }


def directional_change_events(
    path: np.ndarray,
    prior_scale: np.ndarray,
    *,
    width: float,
) -> np.ndarray:
    """Online DC state machine with a threshold frozen at each regime switch."""
    values = np.asarray(path, dtype=float)
    scale = np.asarray(prior_scale, dtype=float)
    events = np.zeros(len(values), dtype=np.int8)
    active = False
    mode = 0
    anchor = high = low = threshold = np.nan
    for position in range(len(values)):
        if not np.isfinite(values[position]) or not np.isfinite(scale[position]) or scale[position] <= 0.0:
            active = False
            mode = 0
            continue
        if not active:
            active = True
            anchor = high = low = values[position]
            threshold = width * scale[position]
            continue
        if mode == 0:
            high = max(high, values[position])
            low = min(low, values[position])
            if values[position] - anchor >= threshold:
                events[position] = 1
                mode = 1
                high = values[position]
                threshold = width * scale[position]
            elif anchor - values[position] >= threshold:
                events[position] = -1
                mode = -1
                low = values[position]
                threshold = width * scale[position]
        elif mode == 1:
            high = max(high, values[position])
            if high - values[position] >= threshold:
                events[position] = -1
                mode = -1
                low = values[position]
                threshold = width * scale[position]
        else:
            low = min(low, values[position])
            if values[position] - low >= threshold:
                events[position] = 1
                mode = 1
                high = values[position]
                threshold = width * scale[position]
    return events


def build_clock_features(
    paths: dict[str, np.ndarray],
    price_events: np.ndarray,
    flow_events: np.ndarray,
    dates: pd.Series,
    *,
    clock_window: int,
) -> pd.DataFrame:
    price_count = pd.Series(price_events != 0).rolling(
        clock_window, min_periods=clock_window // 2
    ).sum()
    flow_count = pd.Series(flow_events != 0).rolling(
        clock_window, min_periods=clock_window // 2
    ).sum()
    root_hours = np.sqrt(clock_window / DECISION_STRIDE)
    price_move = pd.Series(paths["log_price"]).diff(clock_window).to_numpy(float)
    flow_move = pd.Series(paths["flow_path"]).diff(clock_window).to_numpy(float)
    price_displacement_z = price_move / (paths["price_scale"] * root_hours)
    flow_displacement_z = flow_move / (paths["flow_scale"] * root_hours)
    return pd.DataFrame(
        {
            "price_event_count": price_count,
            "flow_event_count": flow_count,
            "clock_log_ratio": np.log1p(flow_count) - np.log1p(price_count),
            "price_displacement_z": price_displacement_z,
            "flow_displacement_z": flow_displacement_z,
            "decision": (dates.dt.minute == DECISION_MINUTE).to_numpy(bool),
        }
    ).replace([np.inf, -np.inf], np.nan)


def impact_state(features: pd.DataFrame, dominance_ratio: float) -> np.ndarray:
    price_count = features["price_event_count"].to_numpy(float)
    flow_count = features["flow_event_count"].to_numpy(float)
    price_move = features["price_displacement_z"].to_numpy(float)
    flow_move = features["flow_displacement_z"].to_numpy(float)
    finite = np.isfinite(price_count) & np.isfinite(flow_count) & np.isfinite(price_move) & np.isfinite(flow_move)
    flow_fast = (
        finite
        & (flow_count >= MIN_FAST_EVENTS)
        & (flow_count >= dominance_ratio * np.maximum(1.0, price_count))
        & (np.abs(price_move) <= 1.0)
        & (np.abs(flow_move) >= 1.0)
    )
    price_fast = (
        finite
        & (price_count >= MIN_FAST_EVENTS)
        & (price_count >= dominance_ratio * np.maximum(1.0, flow_count))
        & (np.abs(price_move) >= 1.0)
        & ((price_move * flow_move >= 0.0) | (np.abs(flow_move) <= 0.5))
    )
    return np.where(flow_fast, 1, np.where(price_fast, 2, 0)).astype(np.int8)


def build_signals(
    features: pd.DataFrame,
    dominance_ratio: float,
    *,
    flip: bool = False,
    first_entry_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    state = impact_state(features, dominance_ratio)
    decision = features["decision"].to_numpy(bool)
    previous_state = np.r_[np.zeros(DECISION_STRIDE, dtype=np.int8), state[:-DECISION_STRIDE]]
    active = decision & (state != 0)
    if first_entry_only:
        active &= previous_state != state
    price_move = features["price_displacement_z"].to_numpy(float)
    flow_move = features["flow_displacement_z"].to_numpy(float)
    side = np.where(state == 1, -np.sign(flow_move), np.sign(price_move))
    side = np.nan_to_num(side).astype(np.int8)
    if flip:
        side = -side
    active &= side != 0
    return active & (side > 0), active & (side < 0), {
        "active": active,
        "state": state,
        "side": side,
        "first_entry": previous_state != state,
    }


def magnitude_only_signals(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Control with the same displacement semantics but no event clocks."""
    price_move = features["price_displacement_z"].to_numpy(float)
    flow_move = features["flow_displacement_z"].to_numpy(float)
    finite = np.isfinite(price_move) & np.isfinite(flow_move)
    flow_absorption = finite & (np.abs(price_move) <= 1.0) & (np.abs(flow_move) >= 1.0)
    price_vacuum = finite & (np.abs(price_move) >= 1.0) & (
        (price_move * flow_move >= 0.0) | (np.abs(flow_move) <= 0.5)
    )
    state = np.where(flow_absorption, 1, np.where(price_vacuum, 2, 0)).astype(np.int8)
    previous = np.r_[np.zeros(DECISION_STRIDE, dtype=np.int8), state[:-DECISION_STRIDE]]
    active = features["decision"].to_numpy(bool) & (state != 0) & (previous != state)
    side = np.nan_to_num(np.where(state == 1, -np.sign(flow_move), np.sign(price_move))).astype(np.int8)
    return active & (side > 0), active & (side < 0)


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
    extremes: tuple[np.ndarray, np.ndarray],
    *,
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
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
    )
    core = [stats[window]["ratio"] for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        sum(stats[window]["return_pct"] > 0.0 for window in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for window in ("fit", "select_2023", *SEGMENTS):
        value = stats[window]
        print(
            window,
            f"ret={value['return_pct']:.2f}",
            f"cagr={value['cagr_pct']:.2f}",
            f"mdd={value['strict_mdd_pct']:.2f}",
            f"ratio={value['ratio']:.2f}",
            f"n={value['trades']}",
            f"L/S={value['longs']}/{value['shorts']}",
        )


def main() -> None:
    market, dates = load_pre2024()
    paths = build_paths(market)
    holds = tuple(sorted({min(window, 144) for window in CLOCK_WINDOWS}))
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    clock_bank: dict[tuple[float, int], pd.DataFrame] = {}
    event_summary: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for width in DC_WIDTHS:
        price_events = directional_change_events(paths["log_price"], paths["price_scale"], width=width)
        flow_events = directional_change_events(paths["flow_path"], paths["flow_scale"], width=width)
        event_summary[str(width)] = {
            "price_events": int((price_events != 0).sum()),
            "flow_events": int((flow_events != 0).sum()),
            "price_up_down": [int((price_events > 0).sum()), int((price_events < 0).sum())],
            "flow_up_down": [int((flow_events > 0).sum()), int((flow_events < 0).sum())],
        }
        for clock_window in CLOCK_WINDOWS:
            features = build_clock_features(
                paths, price_events, flow_events, dates, clock_window=clock_window
            )
            clock_bank[(width, clock_window)] = features
            hold = min(clock_window, 144)
            for dominance_ratio in DOMINANCE_RATIOS:
                long_active, short_active, diagnostics = build_signals(features, dominance_ratio)
                stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
                rows.append(
                    {
                        "dc_width": width,
                        "clock_window": clock_window,
                        "dominance_ratio": dominance_ratio,
                        "hold": hold,
                        "signals": int((long_active | short_active).sum()),
                        "flow_fast_entries": int((diagnostics["active"] & (diagnostics["state"] == 1)).sum()),
                        "price_fast_entries": int((diagnostics["active"] & (diagnostics["state"] == 2)).sum()),
                        "rank": rank_key(stats),
                        "prelim_admitted": admission(stats),
                        "stats": stats,
                    }
                )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} dc{row['dc_width']} clock{row['clock_window']} ratio{row['dominance_ratio']} "
            f"hold{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = clock_bank[(top["dc_width"], top["clock_window"])]
    long_active, short_active, _ = build_signals(features, top["dominance_ratio"])
    hold = top["hold"]
    flip_long, flip_short, _ = build_signals(features, top["dominance_ratio"], flip=True)
    persistent_long, persistent_short, _ = build_signals(
        features, top["dominance_ratio"], first_entry_only=False
    )
    magnitude_long, magnitude_short = magnitude_only_signals(features)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, hold, extremes[hold]),
        "persistent_reentry": simulate(
            market, dates, persistent_long, persistent_short, hold, extremes[hold]
        ),
        "magnitude_only_no_clocks": simulate(
            market, dates, magnitude_long, magnitude_short, hold, extremes[hold]
        ),
    }
    lag = DECISION_STRIDE
    controls["signal_lag_1h"] = simulate(
        market,
        dates,
        np.r_[np.zeros(lag, dtype=bool), long_active[:-lag]],
        np.r_[np.zeros(lag, dtype=bool), short_active[:-lag]],
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
        for side_bp in (0, 1, 3, 6, 10)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "mechanism": "relative directional-change event clocks as price/flow impact elasticity",
            "flow_path": "cumulative signed taker quote flow divided by prior completed 1h quote volume",
            "dc_threshold": "frozen at each online state transition from prior-only one-hour path scale",
            "hourly_decision": "bar open timestamp minute 55 is completed at next hour; signal enters next row open",
            "grid_size": len(rows),
            "design_history": "more flexible pre-2024 quantile-tail, onset and recoupling probes were weak; final architect-reviewed 18-policy impact-elasticity grid is recorded, and all such semantics are contaminated/frozen",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "2023 is inspected internal selection; 2024+ remained sealed",
        },
        "event_summary": event_summary,
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/dual_intrinsic_clock_alpha_scan_2026-07-14.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
