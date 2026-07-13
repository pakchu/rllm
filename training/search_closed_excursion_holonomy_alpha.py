"""Search causal inventory holonomy left by a first-return price excursion."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import (
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
CUTOFF = "2024-01-01"
WINDOWS = {
    "fit": ("2020-06-01", "2023-01-01"),
    "fit_2020_h2": ("2020-06-01", "2021-01-01"),
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
DEPARTURE_Z = (2.0, 3.0)
MAX_AGES = (144, 288)
SCORE_VARIANTS = ("cumulative_flow", "dissipative_work", "flow_elasticity")
TAIL_QUANTILES = (0.80, 0.90)
HOLDS = (72, 144)
MIN_AGE = 12


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future market rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("closed-excursion search requires a complete 5-minute grid")
    return market, dates


def build_bar_state(
    market: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = pd.to_numeric(market["close"], errors="coerce")
    log_return = np.log(close.where(close > 0.0)).diff()
    prior_vol = log_return.shift(1).rolling(288, min_periods=144).std(ddof=0).replace(0.0, np.nan)
    return_z = (log_return / prior_vol).to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    prior_activity = quote.shift(1).rolling(8640, min_periods=4320).median()
    signed_flow = (
        (2.0 * taker_buy - quote) / prior_activity.replace(0.0, np.nan)
    ).clip(-20.0, 20.0)
    return (
        close.to_numpy(float),
        return_z,
        signed_flow.to_numpy(float),
        prior_vol.to_numpy(float),
    )


def closed_excursion_features(
    close: np.ndarray,
    return_z: np.ndarray,
    signed_flow: np.ndarray,
    prior_vol: np.ndarray,
    *,
    departure_z: float,
    max_age: int,
    min_age: int = MIN_AGE,
) -> pd.DataFrame:
    """Close a one-at-a-time excursion on its first causal anchor recross.

    Only loops whose net aggressive flow has the departure sign receive
    inventory-holonomy scores.  ``all_loop_side`` records the price-only
    first-return control before that flow condition is applied.
    """
    close = np.asarray(close, dtype=float)
    return_z = np.asarray(return_z, dtype=float)
    signed_flow = np.asarray(signed_flow, dtype=float)
    prior_vol = np.asarray(prior_vol, dtype=float)
    if not (len(close) == len(return_z) == len(signed_flow) == len(prior_vol)):
        raise ValueError("bar-state inputs must have equal length")
    size = len(close)
    cumulative_flow_score = np.full(size, np.nan)
    dissipative_work_score = np.full(size, np.nan)
    flow_elasticity_score = np.full(size, np.nan)
    excursion_side = np.zeros(size, dtype=np.int8)
    all_loop_side = np.zeros(size, dtype=np.int8)
    episode_age = np.zeros(size, dtype=np.int32)
    net_flow = np.full(size, np.nan)
    max_excursion_z = np.full(size, np.nan)

    active = False
    anchor = np.nan
    unit_vol = np.nan
    side = 0
    age = 0
    cumulative_flow = 0.0
    dissipative_work = 0.0
    maximum_displacement = 0.0
    for position in range(1, size):
        usable = (
            np.isfinite(close[position])
            and close[position] > 0.0
            and np.isfinite(return_z[position])
            and np.isfinite(signed_flow[position])
            and np.isfinite(prior_vol[position])
            and prior_vol[position] > 0.0
        )
        if not usable:
            active = False
            continue
        if active:
            age += 1
            cumulative_flow += signed_flow[position]
            displacement = (close[position] / anchor - 1.0) / unit_vol
            dissipative_work += signed_flow[position] * displacement
            maximum_displacement = max(maximum_displacement, side * displacement)
            returned = (
                (side > 0 and close[position] <= anchor)
                or (side < 0 and close[position] >= anchor)
            )
            if returned:
                if age >= min_age:
                    all_loop_side[position] = side
                    episode_age[position] = age
                    net_flow[position] = cumulative_flow
                    max_excursion_z[position] = maximum_displacement
                    if np.sign(cumulative_flow) == side:
                        root_age = np.sqrt(float(age))
                        cumulative_flow_score[position] = abs(cumulative_flow) / root_age
                        dissipative_work_score[position] = dissipative_work / root_age
                        flow_elasticity_score[position] = (
                            abs(cumulative_flow)
                            / root_age
                            / max(maximum_displacement, 1e-6)
                        )
                        excursion_side[position] = side
                active = False
            elif age >= max_age:
                active = False
        elif abs(return_z[position]) >= departure_z:
            start_vol = prior_vol[position - 1]
            if not np.isfinite(start_vol) or start_vol <= 0.0:
                start_vol = prior_vol[position]
            active = True
            anchor = close[position - 1]
            unit_vol = start_vol
            side = 1 if return_z[position] > 0.0 else -1
            age = 1
            cumulative_flow = signed_flow[position]
            displacement = (close[position] / anchor - 1.0) / unit_vol
            dissipative_work = signed_flow[position] * displacement
            maximum_displacement = max(side * displacement, 0.0)

    return pd.DataFrame(
        {
            "cumulative_flow": cumulative_flow_score,
            "dissipative_work": dissipative_work_score,
            "flow_elasticity": flow_elasticity_score,
            "excursion_side": excursion_side,
            "all_loop_side": all_loop_side,
            "episode_age": episode_age,
            "net_flow": net_flow,
            "max_excursion_z": max_excursion_z,
        }
    )


def fit_threshold(
    values: np.ndarray,
    fit_mask: np.ndarray,
    quantile: float,
) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    reference = values[np.asarray(fit_mask, dtype=bool) & np.isfinite(values) & (values > 0.0)]
    if len(reference) < 20:
        raise ValueError(f"insufficient positive loop scores: {len(reference)}")
    return float(np.quantile(reference, quantile)), int(len(reference))


def build_signals(
    features: pd.DataFrame,
    variant: str,
    threshold: float,
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(features[variant], errors="coerce").to_numpy(float)
    side = features["excursion_side"].to_numpy(np.int8)
    active = np.isfinite(values) & (values >= float(threshold)) & (side != 0)
    trade_side = -side
    if flip:
        trade_side = -trade_side
    return active & (trade_side > 0), active & (trade_side < 0)


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
        stats["fit"]["trades"] >= 40
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"])
        >= 5
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
    enough = (
        stats["fit"]["trades"] >= 40
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"])
        >= 5
    )
    core = [
        stats["fit"]["ratio"],
        stats["select_2023"]["ratio"],
        stats["select_2023_h1"]["ratio"],
        stats["select_2023_h2"]["ratio"],
    ]
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
    close, return_z, signed_flow, prior_vol = build_bar_state(market)
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
    rows: list[dict[str, Any]] = []
    feature_bank: dict[tuple[float, int], pd.DataFrame] = {}
    event_summary: dict[str, dict[str, Any]] = {}
    for departure_z, max_age in itertools.product(DEPARTURE_Z, MAX_AGES):
        features = closed_excursion_features(
            close,
            return_z,
            signed_flow,
            prior_vol,
            departure_z=departure_z,
            max_age=max_age,
        )
        feature_bank[(departure_z, max_age)] = features
        key = f"departure_{departure_z:g}_maxage_{max_age}"
        event_summary[key] = {
            "all_completed_loops": int((features["all_loop_side"] != 0).sum()),
            "same_side_inventory_loops": int((features["excursion_side"] != 0).sum()),
            "median_completed_age_bars": float(
                features.loc[features["episode_age"] > 0, "episode_age"].median()
            ),
        }
        for variant, quantile, hold in itertools.product(
            SCORE_VARIANTS,
            TAIL_QUANTILES,
            HOLDS,
        ):
            threshold, fit_events = fit_threshold(
                features[variant].to_numpy(float),
                fit_mask,
                quantile,
            )
            long_active, short_active = build_signals(features, variant, threshold)
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
                    "departure_z": departure_z,
                    "max_age": max_age,
                    "variant": variant,
                    "tail_quantile": quantile,
                    "hold": hold,
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
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} d{row['departure_z']} max{row['max_age']} "
            f"{row['variant']} q{row['tail_quantile']} h{row['hold']} "
            f"sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = feature_bank[(top["departure_z"], top["max_age"])]
    long_active, short_active = build_signals(
        features,
        top["variant"],
        top["threshold"],
    )
    hold = top["hold"]
    flip_long, flip_short = build_signals(
        features,
        top["variant"],
        top["threshold"],
        flip=True,
    )
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, flip_long, flip_short, hold, extremes[hold]
        )
    }
    loop_side = features["all_loop_side"].to_numpy(np.int8)
    controls["price_loop_only"] = simulate(
        market,
        dates,
        loop_side < 0,
        loop_side > 0,
        hold,
        extremes[hold],
    )
    lag = 12
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
        for side_bp in (0, 1, 3, 6)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "fit_thresholds": WINDOWS["fit"],
            "selection": WINDOWS["select_2023"],
            "grid_size": len(rows),
            "mechanism": "first return to pre-shock anchor with same-side net taker inventory",
            "direction": "fade the trapped aggressive inventory after the price loop closes",
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
    Path("results/closed_excursion_holonomy_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
