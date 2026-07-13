"""Search a causal three-strand market-braid alpha before opening 2024+.

The experiment observes the order in which spot price, perpetual price and a
joint leverage witness first cross frozen post-impulse barriers.  The leverage
witness requires both delayed open-interest expansion and side-aligned premium.
Same-bar first-passage ties are discarded because 5-minute data cannot prove
their intrabar order.
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
SPOT_PREMIUM = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
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
SHOCK_Z = (2.0, 3.0)
PASSAGE_Z = (0.5, 1.0)
MAX_AGES = (72, 144)
TOPOLOGY_MODES = ("strict_chain", "relative_order")
HOLDS = (72, 144)
SCALE_WINDOW = 2016
SCALE_MIN_PERIODS = 1008
OI_PASSAGE_Z = 1.0
PREMIUM_PASSAGE_Z = 1.0


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    auxiliary = _read_before(SPOT_PREMIUM, "date", CUTOFF)
    for frame in (market, auxiliary):
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    auxiliary = auxiliary.sort_values("date").drop_duplicates("date", keep="last")
    columns = ["date", "spot_close", "spot_rows", "premium_index_1m_close", "premium_rows"]
    market = market.merge(auxiliary[columns], on="date", how="left", validate="one_to_one").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("market-braid search requires a complete 5-minute grid")
    return market, dates


def _prior_std(values: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .shift(1)
        .rolling(SCALE_WINDOW, min_periods=SCALE_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
    )


def build_bar_state(market: pd.DataFrame) -> dict[str, np.ndarray]:
    spot = np.log(pd.to_numeric(market["spot_close"], errors="coerce").where(lambda value: value > 0.0))
    perp = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    premium = pd.to_numeric(market["premium_index_1m_close"], errors="coerce")

    # The OI cache is timestamped by its source row.  Delay both value and
    # availability one complete 5m bar before exposing them to a decision.
    raw_oi = pd.to_numeric(market["open_interest"], errors="coerce")
    raw_oi_available = pd.to_numeric(market["open_interest_available"], errors="coerce").eq(1.0)
    delayed_oi = raw_oi.shift(1)
    delayed_oi_available = raw_oi_available.shift(1, fill_value=False)
    log_oi = np.log(delayed_oi.where(delayed_oi > 0.0))

    complete = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5.0)
        & pd.to_numeric(market["premium_rows"], errors="coerce").eq(5.0)
        & delayed_oi_available
        & spot.notna()
        & perp.notna()
        & premium.notna()
        & log_oi.notna()
    )
    pair_complete = complete & complete.shift(1, fill_value=False)
    spot_return = spot.diff().where(pair_complete)
    perp_return = perp.diff().where(pair_complete)
    common_return = 0.5 * (spot_return + perp_return)

    twelve_complete = complete & complete.shift(12, fill_value=False)
    return {
        "spot": spot.to_numpy(float),
        "perp": perp.to_numpy(float),
        "log_oi": log_oi.to_numpy(float),
        "premium": premium.to_numpy(float),
        "valid": complete.to_numpy(bool),
        "shock_z": (common_return / _prior_std(common_return)).to_numpy(float),
        "spot_unit": _prior_std(spot.diff(12).where(twelve_complete)).to_numpy(float),
        "perp_unit": _prior_std(perp.diff(12).where(twelve_complete)).to_numpy(float),
        "oi_unit": _prior_std(log_oi.diff(12).where(twelve_complete)).to_numpy(float),
        "premium_unit": _prior_std(premium.diff(12).where(twelve_complete)).to_numpy(float),
    }


def market_braid_events(
    state: dict[str, np.ndarray],
    *,
    shock_z: float,
    passage_z: float,
    max_age: int,
    topology_mode: str,
    leverage_mode: str = "joint",
) -> pd.DataFrame:
    """Track one causal impulse episode and emit only after required passages.

    ``joint`` requires delayed OI expansion and side-aligned premium on the
    same completed bar. ``oi_only``, ``premium_only`` and ``none`` are fixed
    structural controls, not candidate-search branches.
    """
    if topology_mode not in TOPOLOGY_MODES:
        raise KeyError(topology_mode)
    if leverage_mode not in {"joint", "oi_only", "premium_only", "none"}:
        raise KeyError(leverage_mode)

    size = len(state["spot"])
    signal_side = np.zeros(size, dtype=np.int8)
    impulse_side = np.zeros(size, dtype=np.int8)
    episode_age = np.zeros(size, dtype=np.int16)
    sequence = np.full(size, "", dtype=object)
    departure_event = np.zeros(size, dtype=bool)
    tie_discarded = np.zeros(size, dtype=bool)

    active = False
    for position in range(1, size):
        usable = state["valid"][position] and all(
            np.isfinite(state[name][position])
            for name in ("spot", "perp", "log_oi", "premium", "spot_unit", "perp_unit", "oi_unit", "premium_unit")
        )
        usable = bool(
            usable
            and state["spot_unit"][position] > 0.0
            and state["perp_unit"][position] > 0.0
            and state["oi_unit"][position] > 0.0
            and state["premium_unit"][position] > 0.0
        )
        if active:
            if not usable:
                active = False
                continue
            age += 1
            hits: list[str] = []
            if "spot" not in order and side * (state["spot"][position] - anchor_spot) >= passage_z * spot_unit:
                hits.append("spot")
            if "perp" not in order and side * (state["perp"][position] - anchor_perp) >= passage_z * perp_unit:
                hits.append("perp")
            if leverage_mode != "none" and "leverage" not in order:
                oi_cross = state["log_oi"][position] - anchor_oi >= OI_PASSAGE_Z * oi_unit
                premium_cross = side * (state["premium"][position] - anchor_premium) >= PREMIUM_PASSAGE_Z * premium_unit
                leverage_cross = {
                    "joint": oi_cross and premium_cross,
                    "oi_only": oi_cross,
                    "premium_only": premium_cross,
                }[leverage_mode]
                if leverage_cross:
                    hits.append("leverage")

            # A completed 5m bar cannot establish order among simultaneous
            # passages; discarding is the conservative alternative to guessing.
            if len(hits) > 1:
                tie_discarded[position] = True
                active = False
                continue
            if hits:
                order.append(hits[0])
                required = 2 if leverage_mode == "none" else 3
                if len(order) == required:
                    text = ">".join(order)
                    mapped = 0
                    if leverage_mode == "none":
                        mapped = side if order == ["spot", "perp"] else -side
                    elif topology_mode == "strict_chain":
                        if order == ["spot", "perp", "leverage"]:
                            mapped = side
                        elif order == ["leverage", "perp", "spot"]:
                            mapped = -side
                    else:
                        mapped = side if order.index("spot") < order.index("leverage") else -side
                    signal_side[position] = mapped
                    impulse_side[position] = side
                    episode_age[position] = age
                    sequence[position] = text
                    active = False
            if active and age >= max_age:
                active = False
        elif usable and np.isfinite(state["shock_z"][position]) and abs(state["shock_z"][position]) >= shock_z:
            active = True
            departure_event[position] = True
            side = int(np.sign(state["shock_z"][position]))
            age = 0
            order: list[str] = []
            anchor_spot = state["spot"][position]
            anchor_perp = state["perp"][position]
            anchor_oi = state["log_oi"][position]
            anchor_premium = state["premium"][position]
            spot_unit = state["spot_unit"][position]
            perp_unit = state["perp_unit"][position]
            oi_unit = state["oi_unit"][position]
            premium_unit = state["premium_unit"][position]

    return pd.DataFrame(
        {
            "signal_side": signal_side,
            "impulse_side": impulse_side,
            "episode_age": episode_age,
            "sequence": sequence,
            "departure_event": departure_event,
            "tie_discarded": tie_discarded,
        }
    )


def build_signals(
    events: pd.DataFrame,
    *,
    flip: bool = False,
    order_blind: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    side = events["impulse_side" if order_blind else "signal_side"].to_numpy(np.int8).copy()
    active = events["episode_age"].to_numpy(int) > 0
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
        stats["fit"]["trades"] >= 40
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
    )
    core = ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    return bool(
        enough
        and all(stats[window]["return_pct"] > 0.0 for window in core)
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["ratio"] >= 3.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 40
        and stats["select_2023"]["trades"] >= 16
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 5
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
    state = build_bar_state(market)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    event_bank: dict[tuple[float, float, int, str], pd.DataFrame] = {}
    event_summary: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for shock_z, passage_z, max_age, topology_mode in itertools.product(
        SHOCK_Z, PASSAGE_Z, MAX_AGES, TOPOLOGY_MODES
    ):
        events = market_braid_events(
            state,
            shock_z=shock_z,
            passage_z=passage_z,
            max_age=max_age,
            topology_mode=topology_mode,
        )
        key = (shock_z, passage_z, max_age, topology_mode)
        event_bank[key] = events
        completed = events["episode_age"] > 0
        event_summary[f"shock_{shock_z:g}_passage_{passage_z:g}_age_{max_age}_{topology_mode}"] = {
            "departures": int(events["departure_event"].sum()),
            "completed": int(completed.sum()),
            "mapped_signals": int((events["signal_side"] != 0).sum()),
            "tie_discards": int(events["tie_discarded"].sum()),
            "sequence_counts": {str(name): int(count) for name, count in events.loc[completed, "sequence"].value_counts().items()},
        }
        long_active, short_active = build_signals(events)
        for hold in HOLDS:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "shock_z": shock_z,
                    "passage_z": passage_z,
                    "max_age": max_age,
                    "topology_mode": topology_mode,
                    "hold": hold,
                    "signals": int((long_active | short_active).sum()),
                    "rank": rank_key(stats),
                    "prelim_admitted": admission(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} shock{row['shock_z']} pass{row['passage_z']} age{row['max_age']} "
            f"{row['topology_mode']} hold{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    top_key = (top["shock_z"], top["passage_z"], top["max_age"], top["topology_mode"])
    events = event_bank[top_key]
    long_active, short_active = build_signals(events)
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    flip_long, flip_short = build_signals(events, flip=True)
    controls["direction_flip"] = simulate(market, dates, flip_long, flip_short, hold, extremes[hold])
    blind_long, blind_short = build_signals(events, order_blind=True)
    controls["order_blind_impulse_continuation"] = simulate(
        market, dates, blind_long, blind_short, hold, extremes[hold]
    )
    for leverage_mode in ("none", "oi_only", "premium_only"):
        control_events = market_braid_events(
            state,
            shock_z=top["shock_z"],
            passage_z=top["passage_z"],
            max_age=top["max_age"],
            topology_mode=top["topology_mode"],
            leverage_mode=leverage_mode,
        )
        control_long, control_short = build_signals(control_events)
        controls[f"leverage_witness_{leverage_mode}"] = simulate(
            market, dates, control_long, control_short, hold, extremes[hold]
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
        for side_bp in (0, 1, 3, 6, 10)
    }
    for side_bp, stats in cost_stress.items():
        print_stats(f"COST {side_bp}BP_SIDE", stats)

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "grid_size": len(rows),
            "mechanism": "post-impulse first-passage order of spot, perp, and delayed-OI plus aligned-premium leverage witness",
            "direction": "spot before leverage continues impulse; leverage before spot fades impulse; strict chain admits only spot-perp-leverage and leverage-perp-spot",
            "same_bar_ties": "discarded because intrabar order is unknowable",
            "oi_source_delay_bars": 1,
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
    Path("results/market_braid_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
