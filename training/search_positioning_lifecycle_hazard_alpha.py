"""Search BTC alpha in the first resolution hazard of aged positioning disagreement."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_cme_offshore_debt_handoff_alpha import prior_z
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
METRICS = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
CUTOFF = "2024-01-01"
WINDOWS = {
    "fit": ("2020-10-15", "2022-01-01"),
    "fit_2020_h2": ("2020-10-15", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
SEGMENTS = (
    "fit_2020_h2",
    "fit_2021_h1",
    "fit_2021_h2",
    "select_2023_h1",
    "select_2023_h2",
)


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    metrics = _read_before(METRICS, "create_time", CUTOFF)
    market = _attach_delayed_metrics(
        market,
        metrics,
        tolerance="10min",
        delay_bars=1,
    )
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future rows opened")
    return market, dates


def positioning_valid_mask(dates: pd.Series, available: np.ndarray) -> np.ndarray:
    """Reset lifecycle state across unavailable rows and the quarantined 2022 archive."""
    quarantined = (
        (dates >= pd.Timestamp("2022-01-01"))
        & (dates < pd.Timestamp("2023-01-01"))
    ).to_numpy(bool)
    return np.asarray(available, dtype=bool) & ~quarantined


def build_disagreement_states(market: pd.DataFrame) -> pd.DataFrame:
    top_position = np.log(
        pd.to_numeric(market["sum_toptrader_long_short_ratio"], errors="coerce").where(
            lambda values: values > 0.0
        )
    )
    top_account = np.log(
        pd.to_numeric(market["count_toptrader_long_short_ratio"], errors="coerce").where(
            lambda values: values > 0.0
        )
    )
    global_account = np.log(
        pd.to_numeric(market["count_long_short_ratio"], errors="coerce").where(
            lambda values: values > 0.0
        )
    )
    return pd.DataFrame(
        {
            "top_position_minus_global": prior_z(
                top_position - global_account,
                8640,
                4320,
            ),
            "top_account_minus_global": prior_z(
                top_account - global_account,
                8640,
                4320,
            ),
        },
        index=market.index,
    ).replace([np.inf, -np.inf], np.nan)


def lifecycle_signals(
    disagreement_z: np.ndarray,
    valid: np.ndarray,
    *,
    min_age: int,
    trigger: str,
    entry_z: float = 1.5,
    contraction_fraction: float = 0.5,
    reset_z: float = 0.25,
    flip: bool = False,
    ignore_age: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Emit the first causal resolution of an aged, same-sign disagreement episode."""
    disagreement_z = np.asarray(disagreement_z, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    long_active = np.zeros(len(disagreement_z), dtype=bool)
    short_active = np.zeros(len(disagreement_z), dtype=bool)
    episode_age = np.zeros(len(disagreement_z), dtype=np.int32)
    episode_peak = np.full(len(disagreement_z), np.nan)
    resolution_fraction = np.full(len(disagreement_z), np.nan)

    active = False
    episode_side = 0
    age = 0
    peak = 0.0
    fired = False
    for position, value in enumerate(disagreement_z):
        if not valid[position] or not np.isfinite(value):
            active = False
            episode_side = 0
            age = 0
            peak = 0.0
            fired = False
            continue
        current_side = int(np.sign(value))
        if not active:
            if abs(value) >= entry_z:
                active = True
                episode_side = current_side
                age = 1
                peak = abs(value)
                fired = False
                episode_age[position] = age
                episode_peak[position] = peak
                resolution_fraction[position] = 1.0
            continue

        age += 1
        peak = max(peak, abs(value))
        crossed = current_side != 0 and current_side != episode_side
        contracted = (
            current_side == episode_side
            and peak > 0.0
            and abs(value) <= contraction_fraction * peak
        )
        if trigger == "zero_cross":
            resolved = crossed
        elif trigger == "contraction":
            resolved = contracted
        else:
            raise KeyError(trigger)
        old_side = episode_side
        should_fire = resolved and (ignore_age or age >= min_age) and not fired
        if should_fire:
            side = -old_side
            if flip:
                side = -side
            long_active[position] = side > 0
            short_active[position] = side < 0
            fired = True

        episode_age[position] = age
        episode_peak[position] = peak
        resolution_fraction[position] = abs(value) / peak if peak > 0.0 else np.nan
        if crossed:
            if abs(value) >= entry_z:
                active = True
                episode_side = current_side
                age = 1
                peak = abs(value)
                fired = False
            else:
                active = False
                episode_side = 0
                age = 0
                peak = 0.0
                fired = False
        elif fired and abs(value) < reset_z:
            active = False
            episode_side = 0
            age = 0
            peak = 0.0
            fired = False
    return long_active, short_active, {
        "episode_age": episode_age,
        "episode_peak_abs_z": episode_peak,
        "resolution_fraction": resolution_fraction,
    }


def static_tail_onsets(
    disagreement_z: np.ndarray,
    valid: np.ndarray,
    *,
    entry_z: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(disagreement_z, dtype=float)
    state = np.asarray(valid, dtype=bool) & np.isfinite(values) & (np.abs(values) >= entry_z)
    onset = state & ~np.r_[False, state[:-1]]
    side = -np.sign(values)
    return onset & (side > 0.0), onset & (side < 0.0)


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
        stats[window]["ratio"]
        for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
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
    states = build_disagreement_states(market)
    valid = positioning_valid_mask(
        dates,
        market["positioning_available"].to_numpy(bool),
    )
    holds = (72, 216)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    signal_bank: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for disagreement, min_age, trigger in itertools.product(
        states.columns,
        (144, 432),
        ("contraction", "zero_cross"),
    ):
        long_active, short_active, _ = lifecycle_signals(
            states[disagreement].to_numpy(float),
            valid,
            min_age=min_age,
            trigger=trigger,
        )
        signal_bank[(disagreement, min_age, trigger)] = (long_active, short_active)
        for hold in holds:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "disagreement": disagreement,
                    "min_age": min_age,
                    "trigger": trigger,
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
            f"RANK {index} {row['disagreement']} age{row['min_age']} {row['trigger']} "
            f"h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    values = states[top["disagreement"]].to_numpy(float)
    long_active, short_active = signal_bank[
        (top["disagreement"], top["min_age"], top["trigger"])
    ]
    hold = top["hold"]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, extra in (
        ("direction_flip", {"flip": True}),
        ("ignore_age", {"ignore_age": True}),
    ):
        control_long, control_short, _ = lifecycle_signals(
            values,
            valid,
            min_age=top["min_age"],
            trigger=top["trigger"],
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
    control_long, control_short = static_tail_onsets(values, valid)
    controls["static_tail_onset"] = simulate(
        market,
        dates,
        control_long,
        control_short,
        hold,
        extremes[hold],
    )
    lag = 144
    controls["signal_lag_12h"] = simulate(
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
            "source_delay": "Binance UM metrics delayed by one complete 5m bar",
            "fit": WINDOWS["fit"],
            "quarantine": "2022 metrics state forcibly invalidated and reset",
            "grid_size": 16,
            "episode": "prior-only 30d disagreement z reaches |1.5|; first 50% contraction or zero-cross after 12h/36h",
            "direction": "trade opposite the aged disagreement side as inventory resolves",
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
    Path("results/positioning_lifecycle_hazard_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
