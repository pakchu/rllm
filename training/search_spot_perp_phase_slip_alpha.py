from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_spot_perp_transfer_entropy_alpha import load_pre2024

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
DEADBAND = 0.25
INVALID = np.int8(9)


def symbolic_state(log_return: pd.Series, deadband: float = DEADBAND) -> np.ndarray:
    """Discretize the completed return with volatility from prior bars only."""
    values = pd.to_numeric(log_return, errors="coerce")
    prior_vol = values.shift(1).rolling(288, min_periods=144).std(ddof=0).replace(0, np.nan)
    z = (values / prior_vol).to_numpy(float)
    state = np.full(len(z), INVALID, dtype=np.int8)
    finite = np.isfinite(z)
    state[finite & (z <= -deadband)] = -1
    state[finite & (np.abs(z) < deadband)] = 0
    state[finite & (z >= deadband)] = 1
    return state


def build_states(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    perp_close = pd.to_numeric(market["close"], errors="coerce")
    complete = (
        pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
        & spot_close.gt(0)
        & perp_close.gt(0)
    )
    pair_complete = complete & complete.shift(1, fill_value=False)
    spot_return = np.log(spot_close).diff().where(pair_complete)
    perp_return = np.log(perp_close).diff().where(pair_complete)
    return symbolic_state(spot_return), symbolic_state(perp_return)


def phase_signals(
    leader_state: np.ndarray,
    follower_state: np.ndarray,
    *,
    lock_window: int,
    slip_bars: int,
    min_excess: int,
    relock_mode: str,
    flip: bool = False,
    require_relock: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Detect prior lock -> leader slip -> follower relock using t-or-earlier states."""
    leader = pd.Series(leader_state, dtype="float64").mask(leader_state == INVALID)
    follower = pd.Series(follower_state, dtype="float64").mask(follower_state == INVALID)
    valid = leader.notna() & follower.notna()
    locked = (valid & leader.eq(follower) & leader.ne(0)).astype(float)

    # At t, the slip path is exactly [t-slip_bars, t-1].
    leader_sum = leader.shift(1).rolling(slip_bars, min_periods=slip_bars).sum()
    follower_sum = follower.shift(1).rolling(slip_bars, min_periods=slip_bars).sum()
    slip_valid = valid.astype(int).shift(1).rolling(slip_bars, min_periods=slip_bars).sum().eq(slip_bars)
    direction = np.sign(leader_sum).fillna(0.0)
    lead_excess = direction * (leader_sum - follower_sum)

    # This window ends at t-slip_bars-1 and cannot overlap the slip path.
    lock_count = locked.shift(slip_bars + 1).rolling(lock_window, min_periods=lock_window).sum()
    prior_lock = lock_count.ge(int(np.ceil(0.67 * lock_window)))
    base = slip_valid & prior_lock & direction.ne(0) & lead_excess.ge(min_excess)

    leader_now = leader.to_numpy(float)
    follower_now = follower.to_numpy(float)
    d = direction.to_numpy(float)
    current_valid = np.isfinite(leader_now) & np.isfinite(follower_now)
    follower_newly_matches = (follower_now == d) & (np.roll(follower_now, 1) != d)
    follower_newly_matches[0] = False
    if relock_mode == "soft_relock":
        leader_compatible = (leader_now == d) | (leader_now == 0)
    elif relock_mode == "hard_relock":
        leader_compatible = leader_now == d
    else:
        raise KeyError(relock_mode)
    relock = current_valid & follower_newly_matches & leader_compatible

    if require_relock:
        active = base.to_numpy(bool) & relock
    else:
        # Placebo: act when a qualifying slip first appears, without observing relock.
        base_array = base.to_numpy(bool)
        active = base_array & ~np.roll(base_array, 1)
        active[0] = False

    side = d.copy()
    if flip:
        side = -side
    long_active = active & (side > 0)
    short_active = active & (side < 0)
    diagnostics = {
        "active": active,
        "direction": d,
        "lead_excess": lead_excess.to_numpy(float),
        "prior_lock": prior_lock.to_numpy(bool),
        "relock": relock,
    }
    return long_active, short_active, diagnostics


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold_bars: int,
    extremes: tuple[np.ndarray, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=hold_bars,
            stride_bars=1,
            leverage=0.5,
            fee_rate=0.0005,
            slippage_rate=0.0001,
            extremes=extremes,
            windows=WINDOWS,
        )
        for window in WINDOWS
    }


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and stats["select_2023_h1"]["trades"] >= 8
        and stats["select_2023_h2"]["trades"] >= 8
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0
        and stats["fit"]["ratio"] >= 1.5
        and stats["select_2023"]["return_pct"] > 0
        and stats["select_2023"]["ratio"] >= 2.0
        and stats["select_2023_h1"]["return_pct"] > 0
        and stats["select_2023_h2"]["return_pct"] > 0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
    )
    core = [
        stats["fit"]["ratio"],
        stats["select_2023"]["ratio"],
        stats["select_2023_h1"]["ratio"],
        stats["select_2023_h2"]["ratio"],
    ]
    positive_segments = sum(stats[name]["return_pct"] > 0 for name in SEGMENTS)
    return (
        admission(stats),
        enough,
        min(core) > 0,
        positive_segments,
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
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future rows opened")
    spot_state, perp_state = build_states(market)
    holds = (6, 12, 24)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    for lock_window, slip_bars, min_excess, relock_mode, hold in itertools.product(
        (12, 24), (2, 3, 4), (2, 3), ("soft_relock", "hard_relock"), holds
    ):
        long_active, short_active, diagnostics = phase_signals(
            spot_state,
            perp_state,
            lock_window=lock_window,
            slip_bars=slip_bars,
            min_excess=min_excess,
            relock_mode=relock_mode,
        )
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "lock_window": lock_window,
                "slip_bars": slip_bars,
                "min_excess": min_excess,
                "relock_mode": relock_mode,
                "hold": hold,
                "raw_events": int(diagnostics["active"].sum()),
                "admitted": admission(stats),
                "rank": rank_key(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} lock{row['lock_window']} slip{row['slip_bars']} "
            f"excess{row['min_excess']} {row['relock_mode']} h{row['hold']} "
            f"events={row['raw_events']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    kwargs = {
        "lock_window": top["lock_window"],
        "slip_bars": top["slip_bars"],
        "min_excess": top["min_excess"],
        "relock_mode": top["relock_mode"],
    }
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    control_signals = {
        "direction_flip": phase_signals(spot_state, perp_state, **kwargs, flip=True)[:2],
        "no_relock": phase_signals(spot_state, perp_state, **kwargs, require_relock=False)[:2],
        "perp_led": phase_signals(perp_state, spot_state, **kwargs)[:2],
        "spot_lag_1": phase_signals(np.r_[INVALID, spot_state[:-1]], perp_state, **kwargs)[:2],
        "spot_lag_12": phase_signals(np.r_[np.full(12, INVALID, dtype=np.int8), spot_state[:-12]], perp_state, **kwargs)[:2],
    }
    for name, (long_active, short_active) in control_signals.items():
        controls[name] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])
        print_stats("CONTROL " + name, controls[name])

    result = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "deadband": DEADBAND,
            "grid_size": 72,
            "entry": "next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "controls": controls,
    }
    Path("results/spot_perp_phase_slip_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
