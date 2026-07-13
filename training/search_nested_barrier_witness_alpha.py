from __future__ import annotations

import itertools
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_orderflow_trophic_succession_alpha import SEGMENTS, WINDOWS, load_pre2024
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop

HORIZONS = (144, 576, 2016)


def rolling_prior_extreme_index(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    """Most recent max/min index in [t-window, t), never including decision bar t."""
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), -1, dtype=np.int64)
    candidates: deque[int] = deque()
    for position in range(len(values)):
        previous = position - 1
        if previous >= 0 and np.isfinite(values[previous]):
            if kind == "max":
                while candidates and values[candidates[-1]] <= values[previous]:
                    candidates.pop()
            elif kind == "min":
                while candidates and values[candidates[-1]] >= values[previous]:
                    candidates.pop()
            else:
                raise KeyError(kind)
            candidates.append(previous)
        cutoff = position - window
        while candidates and candidates[0] < cutoff:
            candidates.popleft()
        if position >= window and candidates:
            output[position] = candidates[0]
    return output


def directional_work(market: pd.DataFrame, bars: int = 3) -> tuple[np.ndarray, np.ndarray]:
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    imbalance = (2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)
    buy_work = imbalance.clip(lower=0.0).rolling(bars, min_periods=bars).sum().to_numpy(float)
    sell_work = (-imbalance.clip(upper=0.0)).rolling(bars, min_periods=bars).sum().to_numpy(float)
    return buy_work, sell_work


def build_barrier_bank(market: pd.DataFrame) -> dict[Any, Any]:
    high = market["high"].to_numpy(float)
    low = market["low"].to_numpy(float)
    bank: dict[Any, Any] = {}
    for horizon in HORIZONS:
        high_index = rolling_prior_extreme_index(high, horizon, "max")
        low_index = rolling_prior_extreme_index(low, horizon, "min")
        high_price = np.full(len(high), np.nan)
        low_price = np.full(len(low), np.nan)
        high_valid = high_index >= 0
        low_valid = low_index >= 0
        high_price[high_valid] = high[high_index[high_valid]]
        low_price[low_valid] = low[low_index[low_valid]]
        bank[horizon] = {
            "high_index": high_index,
            "low_index": low_index,
            "high_price": high_price,
            "low_price": low_price,
        }
    bank["buy_work"], bank["sell_work"] = directional_work(market)
    return bank


def coalesced_barrier_signals(
    market: pd.DataFrame,
    bank: dict[Any, Any],
    *,
    min_coalescence: int,
    touch_width: float,
    branch: str,
    work_low: float = 0.75,
    work_high: float = 1.25,
    max_origin_separation: int = 3,
    flip: bool = False,
    ignore_witness: bool = False,
    ignore_coalescence: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    close = market["close"].to_numpy(float)
    buy_work = bank["buy_work"]
    sell_work = bank["sell_work"]
    high_active = np.zeros(len(market), dtype=bool)
    low_active = np.zeros(len(market), dtype=bool)
    high_ratio = np.full(len(market), np.nan)
    low_ratio = np.full(len(market), np.nan)
    high_count = np.zeros(len(market), dtype=np.int8)
    low_count = np.zeros(len(market), dtype=np.int8)

    for position in range(len(market)):
        for barrier_side in ("high", "low"):
            touched: list[tuple[int, int, float]] = []
            for horizon in HORIZONS:
                witness_index = int(bank[horizon][f"{barrier_side}_index"][position])
                level = float(bank[horizon][f"{barrier_side}_price"][position])
                if witness_index >= 0 and np.isfinite(level) and abs(close[position] / level - 1.0) <= touch_width:
                    touched.append((horizon, witness_index, level))
            required = 1 if ignore_coalescence else min_coalescence
            if len(touched) < required:
                continue
            indices = [witness_index for _, witness_index, _ in touched]
            if not ignore_coalescence and max(indices) - min(indices) > max_origin_separation:
                continue

            # The longest touched horizon owns the ancestral witness state.
            _, witness_index, _ = max(touched, key=lambda item: item[0])
            levels = [level for _, _, level in touched]
            if barrier_side == "high":
                high_count[position] = len(touched)
                origin_work = buy_work[witness_index]
                current_work = buy_work[position]
                price_not_closed_through = close[position] <= max(levels)
                ratio = (
                    current_work / origin_work
                    if np.isfinite(origin_work) and origin_work > 1e-4 and np.isfinite(current_work)
                    else np.nan
                )
                high_ratio[position] = ratio
                if branch == "depleted_continuation":
                    selected = price_not_closed_through and (
                        ignore_witness or (np.isfinite(ratio) and ratio <= work_low)
                    )
                elif branch == "reinforced_fade":
                    selected = price_not_closed_through and (
                        ignore_witness or (np.isfinite(ratio) and ratio >= work_high)
                    )
                else:
                    raise KeyError(branch)
                high_active[position] = selected
            else:
                low_count[position] = len(touched)
                origin_work = sell_work[witness_index]
                current_work = sell_work[position]
                price_not_closed_through = close[position] >= min(levels)
                ratio = (
                    current_work / origin_work
                    if np.isfinite(origin_work) and origin_work > 1e-4 and np.isfinite(current_work)
                    else np.nan
                )
                low_ratio[position] = ratio
                if branch == "depleted_continuation":
                    selected = price_not_closed_through and (
                        ignore_witness or (np.isfinite(ratio) and ratio <= work_low)
                    )
                elif branch == "reinforced_fade":
                    selected = price_not_closed_through and (
                        ignore_witness or (np.isfinite(ratio) and ratio >= work_high)
                    )
                else:
                    raise KeyError(branch)
                low_active[position] = selected

    high_onset = high_active & ~np.r_[False, high_active[:-1]]
    low_onset = low_active & ~np.r_[False, low_active[:-1]]
    if branch == "depleted_continuation":
        long_active = high_onset & ~low_onset
        short_active = low_onset & ~high_onset
    else:
        long_active = low_onset & ~high_onset
        short_active = high_onset & ~low_onset
    if flip:
        long_active, short_active = short_active, long_active
    return long_active, short_active, {
        "high_work_ratio": high_ratio,
        "low_work_ratio": low_ratio,
        "high_coalescence": high_count,
        "low_coalescence": low_count,
    }


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
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0
        and stats["fit"]["ratio"] >= 3
        and stats["select_2023"]["return_pct"] > 0
        and stats["select_2023"]["ratio"] >= 3
        and stats["select_2023_h1"]["return_pct"] >= 0
        and stats["select_2023_h2"]["return_pct"] >= 0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 6
    )
    core = [stats[window]["ratio"] for window in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        enough,
        min(core) > 0,
        sum(stats[window]["return_pct"] > 0 for window in SEGMENTS),
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
    bank = build_barrier_bank(market)
    holds = (72, 144)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in holds
    }
    rows: list[dict[str, Any]] = []
    signal_bank: dict[tuple[int, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for min_coalescence, touch_width, branch in itertools.product(
        (2, 3),
        (0.001, 0.002),
        ("depleted_continuation", "reinforced_fade"),
    ):
        long_active, short_active, _ = coalesced_barrier_signals(
            market,
            bank,
            min_coalescence=min_coalescence,
            touch_width=touch_width,
            branch=branch,
        )
        signal_bank[(min_coalescence, touch_width, branch)] = (long_active, short_active)
        for hold in holds:
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "min_coalescence": min_coalescence,
                    "touch_width": touch_width,
                    "branch": branch,
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
            f"RANK {index} c{row['min_coalescence']} w{row['touch_width']} {row['branch']} h{row['hold']} sig={row['signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    key = (top["min_coalescence"], top["touch_width"], top["branch"])
    long_active, short_active = signal_bank[key]
    hold = top["hold"]
    signal_kwargs = {
        "min_coalescence": top["min_coalescence"],
        "touch_width": top["touch_width"],
        "branch": top["branch"],
    }
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    for name, extra in (
        ("direction_flip", {"flip": True}),
        ("ignore_witness", {"ignore_witness": True}),
        ("ignore_coalescence", {"ignore_coalescence": True}),
    ):
        control_long, control_short, _ = coalesced_barrier_signals(
            market,
            bank,
            **signal_kwargs,
            **extra,
        )
        controls[name] = simulate(market, dates, control_long, control_short, hold, extremes[hold])
    lag = 12
    lag_long = np.r_[np.zeros(lag, dtype=bool), long_active[:-lag]]
    lag_short = np.r_[np.zeros(lag, dtype=bool), short_active[:-lag]]
    controls["signal_lag"] = simulate(market, dates, lag_long, lag_short, hold, extremes[hold])
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
            "grid_size": 16,
            "horizons": HORIZONS,
            "work_ratio": {"depleted": 0.75, "reinforced": 1.25},
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
    Path("results/nested_barrier_witness_alpha_scan_2026-07-13.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
