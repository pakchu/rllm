"""Audit a causal 1D-persistence barrier-annihilation alpha before 2024+.

At each hourly decision, resistance/support barriers are computed from a price
window ending before the previous hourly decision and then frozen.  The next
hour's endpoint may cross a collection of those barriers.  Local extrema
prominence is the 1D persistence proxy; crossed persistence mass, largest
crossed barrier and mass density are tested as topological path-work scores.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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
HORIZONS = (576, 2016)
SCORE_VARIANTS = ("persistence_mass", "largest_persistence", "mass_density")
TAIL_QUANTILES = (0.80, 0.90, 0.95, 0.975)
HOLDS = (72, 144, 288)
DIRECTION_MODES = ("continue", "fade")
DECISION_STRIDE = 12
MIN_PROMINENCE_Z = 0.5
SCALE_WINDOW = 2016
SCALE_MIN_PERIODS = 1008


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("persistence search requires a complete 5-minute grid")
    return market, dates


def prior_hourly_scale(log_price: pd.Series) -> np.ndarray:
    hourly_change = pd.to_numeric(log_price, errors="coerce").diff(DECISION_STRIDE)
    return (
        hourly_change.shift(1)
        .rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
        .to_numpy(float)
    )


def persistent_barrier_features(
    log_price: np.ndarray,
    dates: pd.Series,
    *,
    horizon: int,
    minimum_prominence_z: float = MIN_PROMINENCE_Z,
) -> pd.DataFrame:
    """Measure persistence barriers crossed since the previous hourly decision.

    For decision ``t``, ``freeze=t-12``.  The extrema set uses only
    ``[freeze-horizon, freeze)`` and is therefore fixed before the one-hour
    traversal from ``price[freeze]`` to ``price[t]``.
    """
    price = np.asarray(log_price, dtype=float)
    size = len(price)
    side = np.zeros(size, dtype=np.int8)
    persistence_mass = np.full(size, np.nan)
    largest_persistence = np.full(size, np.nan)
    barrier_count = np.zeros(size, dtype=np.int16)
    mass_density = np.full(size, np.nan)
    traversal_z = np.full(size, np.nan)
    scale = prior_hourly_scale(pd.Series(price))
    decisions = np.flatnonzero((dates.dt.minute == 0).to_numpy(bool))

    for position in decisions:
        freeze = position - DECISION_STRIDE
        if freeze < horizon or not all(np.isfinite(value) for value in (scale[freeze], price[freeze], price[position])):
            continue
        if scale[freeze] <= 0.0:
            continue
        window = price[freeze - horizon : freeze]
        if len(window) != horizon or not np.isfinite(window).all():
            continue
        start = price[freeze]
        end = price[position]
        minimum_prominence = minimum_prominence_z * scale[freeze]
        if end > start:
            extrema, properties = find_peaks(window, prominence=minimum_prominence)
            levels = window[extrema]
            crossed = (levels > start) & (levels <= end)
            direction = 1
        elif end < start:
            extrema, properties = find_peaks(-window, prominence=minimum_prominence)
            levels = window[extrema]
            crossed = (levels < start) & (levels >= end)
            direction = -1
        else:
            continue
        if not crossed.any():
            continue
        normalized = np.asarray(properties["prominences"], dtype=float)[crossed] / scale[freeze]
        move = abs(end - start) / scale[freeze]
        mass = float(normalized.sum())
        side[position] = direction
        persistence_mass[position] = mass
        largest_persistence[position] = float(normalized.max())
        barrier_count[position] = int(len(normalized))
        mass_density[position] = mass / max(move, 1e-12)
        traversal_z[position] = direction * (end - start) / scale[freeze]

    return pd.DataFrame(
        {
            "side": side,
            "persistence_mass": persistence_mass,
            "largest_persistence": largest_persistence,
            "barrier_count": barrier_count,
            "mass_density": mass_density,
            "traversal_z": traversal_z,
        }
    )


def fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> tuple[float, int]:
    array = np.asarray(values, dtype=float)
    reference = array[np.asarray(fit_mask, dtype=bool) & np.isfinite(array) & (array > 0.0)]
    if len(reference) < 100:
        raise ValueError(f"insufficient positive fit events: {len(reference)}")
    return float(np.quantile(reference, quantile)), int(len(reference))


def build_signals(
    features: pd.DataFrame,
    variant: str,
    threshold: float,
    direction_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if direction_mode not in DIRECTION_MODES:
        raise KeyError(direction_mode)
    score = pd.to_numeric(features[variant], errors="coerce").to_numpy(float)
    side = features["side"].to_numpy(np.int8).copy()
    active = np.isfinite(score) & (score >= threshold) & (side != 0)
    if direction_mode == "fade":
        side = -side
    return active & (side > 0), active & (side < 0)


def frozen_rolling_extrema_signals(
    log_price: np.ndarray,
    dates: pd.Series,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Control: cross only the frozen window's single global max/min."""
    price = np.asarray(log_price, dtype=float)
    long_active = np.zeros(len(price), dtype=bool)
    short_active = np.zeros(len(price), dtype=bool)
    for position in np.flatnonzero((dates.dt.minute == 0).to_numpy(bool)):
        freeze = position - DECISION_STRIDE
        if freeze < horizon or not np.isfinite(price[[freeze, position]]).all():
            continue
        window = price[freeze - horizon : freeze]
        if len(window) != horizon or not np.isfinite(window).all():
            continue
        long_active[position] = price[freeze] < np.max(window) <= price[position]
        short_active[position] = price[freeze] > np.min(window) >= price[position]
    return long_active, short_active


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
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0)).to_numpy(float)
    fit_mask = (
        (dates >= pd.Timestamp(WINDOWS["fit"][0]))
        & (dates < pd.Timestamp(WINDOWS["fit"][1]))
    ).to_numpy(bool)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    feature_bank: dict[int, pd.DataFrame] = {}
    feature_summary: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        features = persistent_barrier_features(log_price, dates, horizon=horizon)
        feature_bank[horizon] = features
        feature_summary[str(horizon)] = {
            "crossing_events": int((features["side"] != 0).sum()),
            "up_events": int((features["side"] > 0).sum()),
            "down_events": int((features["side"] < 0).sum()),
            "median_barrier_count": float(features.loc[features["side"] != 0, "barrier_count"].median()),
            "median_persistence_mass_z": float(features["persistence_mass"].median()),
        }
        thresholds = {
            (variant, quantile): fit_threshold(features[variant].to_numpy(float), fit_mask, quantile)
            for variant, quantile in itertools.product(SCORE_VARIANTS, TAIL_QUANTILES)
        }
        for variant, quantile, hold, direction_mode in itertools.product(
            SCORE_VARIANTS, TAIL_QUANTILES, HOLDS, DIRECTION_MODES
        ):
            threshold, fit_events = thresholds[(variant, quantile)]
            long_active, short_active = build_signals(features, variant, threshold, direction_mode)
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "horizon": horizon,
                    "variant": variant,
                    "tail_quantile": quantile,
                    "hold": hold,
                    "direction_mode": direction_mode,
                    "threshold": threshold,
                    "positive_fit_events": fit_events,
                    "signals": int((long_active | short_active).sum()),
                    "rank": rank_key(stats),
                    "prelim_admitted": admission(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows[:16], 1):
        print_stats(
            f"RANK {index} hzn{row['horizon']} {row['variant']} q{row['tail_quantile']} "
            f"hold{row['hold']} {row['direction_mode']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = feature_bank[top["horizon"]]
    long_active, short_active = build_signals(
        features, top["variant"], top["threshold"], top["direction_mode"]
    )
    hold = top["hold"]
    inverse_mode = "fade" if top["direction_mode"] == "continue" else "continue"
    inverse_long, inverse_short = build_signals(features, top["variant"], top["threshold"], inverse_mode)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, inverse_long, inverse_short, hold, extremes[hold])
    }
    all_long, all_short = build_signals(features, "persistence_mass", 0.0, top["direction_mode"])
    controls["all_crossings_no_tail"] = simulate(market, dates, all_long, all_short, hold, extremes[hold])
    depth_threshold, _ = fit_threshold(
        features["barrier_count"].to_numpy(float), fit_mask, top["tail_quantile"]
    )
    depth_long, depth_short = build_signals(
        features, "barrier_count", depth_threshold, top["direction_mode"]
    )
    controls["barrier_count_only"] = simulate(
        market, dates, depth_long, depth_short, hold, extremes[hold]
    )
    extrema_long, extrema_short = frozen_rolling_extrema_signals(
        log_price, dates, horizon=top["horizon"]
    )
    if top["direction_mode"] == "fade":
        extrema_long, extrema_short = extrema_short, extrema_long
    controls["single_global_extrema"] = simulate(
        market, dates, extrema_long, extrema_short, hold, extremes[hold]
    )
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
            "feature_freeze": "at prior hourly decision; extrema window ends before freeze bar",
            "persistence_proxy": "scipy local-extrema topographic prominence, equivalent to 1D component persistence away from boundary-essential classes",
            "minimum_prominence": "fixed 0.5 prior-only one-hour sigma",
            "grid_size": len(rows),
            "preflight_history": "initial 24 continuation policies were weak; before any OOS, audit expanded to record q80/q90/q95/q97.5, 6h/12h/24h holds and both fixed direction interpretations",
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "2023 is inspected internal selection; 2024+ remained sealed",
        },
        "feature_summary": feature_summary,
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/persistent_barrier_annihilation_alpha_scan_2026-07-14.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
