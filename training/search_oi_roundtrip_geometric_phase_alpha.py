"""Search a causal OI-round-trip geometric-phase alpha before opening 2024+.

A positive open-interest shock starts one leveraged-inventory episode.  The
anchor is frozen before the shock and the episode ends at the first completed
bar where OI returns to that anchor.  If leveraged inventory has round-tripped
but price remains displaced, the residual price phase is interpreted as a
cash/passive repricing that can continue.  The signed price/OI line integral is
also tested as a forced-inventory relaxation branch.
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
DEPARTURE_Z = (2.0, 3.0)
MAX_AGES = (144, 288)
SCORE_VARIANTS = ("residual_phase", "terminal_persistence", "inventory_work")
TAIL_QUANTILES = (0.80, 0.90)
HOLDS = (72, 144)
MIN_AGE = 12
OI_SCALE_WINDOW = 2016


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future market rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("OI round-trip search requires a complete 5-minute grid")
    return market, dates


def build_bar_state(market: pd.DataFrame) -> dict[str, np.ndarray]:
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0.0))
    price_return = log_price.diff()
    prior_price_vol = price_return.shift(1).rolling(288, min_periods=144).std(ddof=0).replace(0.0, np.nan)

    raw_oi = pd.to_numeric(market["open_interest"], errors="coerce")
    log_oi = np.log(raw_oi.where(raw_oi > 0.0))
    available = pd.to_numeric(market["open_interest_available"], errors="coerce").eq(1.0)
    valid_pair = available & available.shift(1, fill_value=False) & log_oi.notna() & log_oi.shift(1).notna()
    oi_delta = log_oi.diff().where(valid_pair)
    prior_oi_scale = oi_delta.shift(1).rolling(
        OI_SCALE_WINDOW,
        min_periods=OI_SCALE_WINDOW // 2,
    ).std(ddof=0).replace(0.0, np.nan)
    oi_delta_z = oi_delta / prior_oi_scale
    return {
        "log_price": log_price.to_numpy(float),
        "price_return": price_return.to_numpy(float),
        "prior_price_vol": prior_price_vol.to_numpy(float),
        "log_oi": log_oi.to_numpy(float),
        "oi_delta": oi_delta.to_numpy(float),
        "prior_oi_scale": prior_oi_scale.to_numpy(float),
        "oi_delta_z": oi_delta_z.to_numpy(float),
        "available": available.to_numpy(bool),
    }


def oi_roundtrip_features(
    state: dict[str, np.ndarray],
    *,
    departure_z: float,
    max_age: int,
    min_age: int = MIN_AGE,
    ignore_first_return: bool = False,
) -> pd.DataFrame:
    """Emit a feature only when a causally tracked OI episode closes.

    ``ignore_first_return`` is a fixed-age control.  It starts from the same
    positive OI shock but deliberately ignores OI recrosses and emits at the
    predeclared maximum age.
    """
    size = len(state["log_price"])
    residual_phase = np.full(size, np.nan)
    terminal_persistence = np.full(size, np.nan)
    inventory_work = np.full(size, np.nan)
    residual_side = np.zeros(size, dtype=np.int8)
    work_side = np.zeros(size, dtype=np.int8)
    episode_age = np.zeros(size, dtype=np.int32)
    residual_price_z = np.full(size, np.nan)
    geometric_work = np.full(size, np.nan)
    maximum_price_displacement = np.full(size, np.nan)
    departure_event = np.zeros(size, dtype=bool)

    active = False
    anchor_oi = anchor_price = price_unit = oi_unit = np.nan
    age = 0
    work = 0.0
    path_maximum = 0.0
    for position in range(1, size):
        usable = (
            state["available"][position]
            and np.isfinite(state["log_price"][position])
            and np.isfinite(state["log_oi"][position])
            and np.isfinite(state["prior_price_vol"][position])
            and state["prior_price_vol"][position] > 0.0
            and np.isfinite(state["oi_delta"][position])
        )
        if not usable:
            active = False
            continue
        if active:
            age += 1
            displacement = (state["log_price"][position] - anchor_price) / price_unit
            work += displacement * state["oi_delta"][position] / oi_unit
            path_maximum = max(path_maximum, abs(displacement))
            recrossed = state["log_oi"][position] <= anchor_oi
            should_emit = age >= max_age if ignore_first_return else recrossed
            if should_emit:
                if age >= min_age:
                    root_age = np.sqrt(float(age))
                    persistence = abs(displacement) / max(path_maximum, 1e-9)
                    residual_phase[position] = abs(displacement) / root_age
                    terminal_persistence[position] = residual_phase[position] * persistence
                    inventory_work[position] = abs(work) / root_age
                    residual_side[position] = int(np.sign(displacement))
                    work_side[position] = int(np.sign(work))
                    episode_age[position] = age
                    residual_price_z[position] = displacement
                    geometric_work[position] = work
                    maximum_price_displacement[position] = path_maximum
                active = False
            elif age >= max_age:
                active = False
        elif np.isfinite(state["oi_delta_z"][position]) and state["oi_delta_z"][position] >= departure_z:
            start_price_vol = state["prior_price_vol"][position - 1]
            if not np.isfinite(start_price_vol) or start_price_vol <= 0.0:
                start_price_vol = state["prior_price_vol"][position]
            start_oi_scale = state["prior_oi_scale"][position]
            if not np.isfinite(start_oi_scale) or start_oi_scale <= 0.0:
                continue
            active = True
            departure_event[position] = True
            anchor_oi = state["log_oi"][position - 1]
            anchor_price = state["log_price"][position - 1]
            price_unit = start_price_vol
            oi_unit = start_oi_scale
            age = 1
            displacement = (state["log_price"][position] - anchor_price) / price_unit
            work = displacement * state["oi_delta"][position] / oi_unit
            path_maximum = abs(displacement)

    return pd.DataFrame(
        {
            "residual_phase": residual_phase,
            "terminal_persistence": terminal_persistence,
            "inventory_work": inventory_work,
            "residual_side": residual_side,
            "work_side": work_side,
            "episode_age": episode_age,
            "residual_price_z": residual_price_z,
            "geometric_work": geometric_work,
            "maximum_price_displacement": maximum_price_displacement,
            "departure_event": departure_event,
        }
    )


def fit_threshold(values: np.ndarray, fit_mask: np.ndarray, quantile: float) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    reference = values[np.asarray(fit_mask, dtype=bool) & np.isfinite(values) & (values > 0.0)]
    if len(reference) < 20:
        raise ValueError(f"insufficient positive fit events: {len(reference)}")
    return float(np.quantile(reference, quantile)), int(len(reference))


def build_signals(
    features: pd.DataFrame,
    variant: str,
    threshold: float,
    *,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    score = pd.to_numeric(features[variant], errors="coerce").to_numpy(float)
    side_column = "work_side" if variant == "inventory_work" else "residual_side"
    side = features[side_column].to_numpy(np.int8)
    active = np.isfinite(score) & (score >= float(threshold)) & (side != 0)
    if flip:
        side = -side
    return active & (side > 0), active & (side < 0)


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
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    both_sides = (
        min(stats["fit"]["longs"], stats["fit"]["shorts"]) > 0
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) > 0
    )
    return bool(
        enough
        and both_sides
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
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
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
    state = build_bar_state(market)
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
    feature_bank: dict[tuple[float, int], pd.DataFrame] = {}
    event_summary: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for departure_z, max_age in itertools.product(DEPARTURE_Z, MAX_AGES):
        features = oi_roundtrip_features(state, departure_z=departure_z, max_age=max_age)
        feature_bank[(departure_z, max_age)] = features
        completed = features["episode_age"] > 0
        key = f"departure_{departure_z:g}_maxage_{max_age}"
        event_summary[key] = {
            "departure_events": int(features["departure_event"].sum()),
            "completed_oi_roundtrips": int(completed.sum()),
            "median_completed_age_bars": float(features.loc[completed, "episode_age"].median()),
            "residual_up": int((features["residual_side"] > 0).sum()),
            "residual_down": int((features["residual_side"] < 0).sum()),
        }
        for variant, quantile, hold in itertools.product(SCORE_VARIANTS, TAIL_QUANTILES, HOLDS):
            threshold, fit_events = fit_threshold(features[variant].to_numpy(float), fit_mask, quantile)
            long_active, short_active = build_signals(features, variant, threshold)
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
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
            f"RANK {index} d{row['departure_z']} max{row['max_age']} {row['variant']} "
            f"q{row['tail_quantile']} h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    features = feature_bank[(top["departure_z"], top["max_age"])]
    long_active, short_active = build_signals(features, top["variant"], top["threshold"])
    flip_long, flip_short = build_signals(features, top["variant"], top["threshold"], flip=True)
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, hold, extremes[hold])
    }
    fixed = oi_roundtrip_features(
        state,
        departure_z=top["departure_z"],
        max_age=top["max_age"],
        ignore_first_return=True,
    )
    fixed_threshold, _ = fit_threshold(fixed["residual_phase"].to_numpy(float), fit_mask, top["tail_quantile"])
    fixed_long, fixed_short = build_signals(fixed, "residual_phase", fixed_threshold)
    controls["fixed_age_ignore_oi_reclosure"] = simulate(
        market, dates, fixed_long, fixed_short, hold, extremes[hold]
    )
    all_side = features["residual_side"].to_numpy(np.int8)
    controls["all_oi_roundtrips_no_score"] = simulate(
        market, dates, all_side > 0, all_side < 0, hold, extremes[hold]
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
            "mechanism": "positive OI shock, first return of OI to its frozen pre-shock anchor, then residual price/work geometric phase",
            "direction": "continue residual price phase; inventory-work branch relaxes in signed-work direction",
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
    Path("results/oi_roundtrip_geometric_phase_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
