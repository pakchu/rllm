from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop
from training.search_positioning_hgb_path_alpha import _read_before
from training.search_spot_perp_transfer_entropy_alpha import prior_z

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
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
DepositMode = Literal["actual", "month_offset", "lag12"]


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", "2024-01-01")
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dates = pd.to_datetime(market["date"])
    if dates.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError("future rows opened")
    return market, dates


def build_causal_inputs(market: pd.DataFrame) -> pd.DataFrame:
    """Use OI observed at least one complete 5m bar earlier."""
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0.0))
    raw_oi = pd.to_numeric(market["open_interest"], errors="coerce")
    available = pd.to_numeric(market["open_interest_available"], errors="coerce").fillna(0.0).gt(0.5)
    delayed_oi = np.log(raw_oi.where(raw_oi > 0.0)).shift(1)
    delayed_available = available.shift(1, fill_value=False)
    oi_change = delayed_oi.diff().where(delayed_available & delayed_available.shift(1, fill_value=False))
    contraction = (-oi_change).clip(lower=0.0)
    contraction_z = prior_z(contraction, 2016)

    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    signed_flow = (2.0 * taker_buy - quote) / quote.replace(0.0, np.nan)
    flow_z = prior_z(signed_flow, 288)
    return pd.DataFrame(
        {
            "log_price": log_price,
            "ret_12": log_price.diff(12),
            "contraction_z": contraction_z,
            "flow_z": flow_z,
            "oi_source_delay_bars": np.where(delayed_available, 1.0, np.nan),
        }
    ).replace([np.inf, -np.inf], np.nan)


def fit_threshold(values: pd.Series, dates: pd.Series, quantile: float, *, positive_only: bool = False) -> float:
    start, end = WINDOWS["fit"]
    mask = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    sample = pd.to_numeric(values[mask], errors="coerce").dropna()
    if positive_only:
        sample = sample[sample > 0.0]
    if len(sample) < 100:
        raise ValueError(f"insufficient fit sample: {len(sample)}")
    return float(sample.quantile(quantile))


def _month_offset(timestamp: pd.Timestamp) -> int:
    month_key = timestamp.year * 12 + timestamp.month
    value = int((month_key * 1103515245 + 12345) % 401) - 200
    if -30 <= value <= 30:
        value += 61 if value >= 0 else -61
    return value


def replay_scar_field(
    inputs: pd.DataFrame,
    dates: pd.Series,
    *,
    bin_width: float,
    half_life: int,
    contraction_threshold: float,
    deposit_mode: DepositMode = "actual",
) -> pd.DataFrame:
    """Query prior scar state, then deposit the current completed event for t+1+."""
    price = inputs["log_price"].to_numpy(float)
    contraction_z = inputs["contraction_z"].to_numpy(float)
    flow_z = inputs["flow_z"].to_numpy(float)
    radius = max(3, int(np.ceil(0.003 / bin_width)))
    decay = float(np.exp(np.log(0.5) / half_life))
    kernel_offsets = np.arange(-3, 4, dtype=int)
    kernel_weights = (4.0 - np.abs(kernel_offsets)).astype(float)
    kernel_weights /= kernel_weights.sum()

    up_field: dict[int, float] = {}
    down_field: dict[int, float] = {}
    up_total = 0.0
    down_total = 0.0
    scale = 1.0
    up_ahead = np.zeros(len(inputs), dtype=float)
    down_below = np.zeros(len(inputs), dtype=float)
    scalar_up = np.zeros(len(inputs), dtype=float)
    scalar_down = np.zeros(len(inputs), dtype=float)
    deposits = np.zeros(len(inputs), dtype=float)

    for i in range(len(inputs)):
        scale *= decay
        if scale < 1e-100:
            up_field = {key: value * scale for key, value in up_field.items() if value * scale > 1e-14}
            down_field = {key: value * scale for key, value in down_field.items() if value * scale > 1e-14}
            up_total *= scale
            down_total *= scale
            scale = 1.0
        if not np.isfinite(price[i]):
            continue
        current_bin = int(np.floor(price[i] / bin_width))
        # The current event is intentionally absent from this query.
        up_ahead[i] = scale * sum(up_field.get(key, 0.0) for key in range(current_bin, current_bin + radius + 1))
        down_below[i] = scale * sum(down_field.get(key, 0.0) for key in range(current_bin - radius, current_bin + 1))
        scalar_up[i] = scale * up_total
        scalar_down[i] = scale * down_total

        cz = contraction_z[i]
        fz = flow_z[i]
        if not (np.isfinite(cz) and np.isfinite(fz) and cz >= contraction_threshold and abs(fz) >= 1.0):
            continue
        mass = min(max(cz - contraction_threshold, 0.0), 5.0) * min(max(abs(fz) - 1.0, 0.0), 5.0)
        if mass <= 0.0:
            continue
        deposit_price_index = i
        if deposit_mode == "lag12":
            deposit_price_index = i - 12
            if deposit_price_index < 0 or not np.isfinite(price[deposit_price_index]):
                continue
        deposit_bin = int(np.floor(price[deposit_price_index] / bin_width))
        if deposit_mode == "month_offset":
            deposit_bin += _month_offset(pd.Timestamp(dates.iloc[i]))
        field = up_field if fz > 0.0 else down_field
        stored_mass = mass / scale
        for offset, weight in zip(kernel_offsets, kernel_weights, strict=True):
            key = deposit_bin + int(offset)
            field[key] = field.get(key, 0.0) + stored_mass * float(weight)
        if fz > 0.0:
            up_total += stored_mass
        else:
            down_total += stored_mass
        deposits[i] = mass

    return pd.DataFrame(
        {
            "up_scar_ahead": up_ahead,
            "down_scar_below": down_below,
            "scalar_up": scalar_up,
            "scalar_down": scalar_down,
            "ret_12": inputs["ret_12"].to_numpy(float),
            "deposit_mass": deposits,
        }
    )


def scar_signals(
    features: pd.DataFrame,
    *,
    up_threshold: float,
    down_threshold: float,
    mapping: str,
    scalar: bool = False,
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    up_column = "scalar_up" if scalar else "up_scar_ahead"
    down_column = "scalar_down" if scalar else "down_scar_below"
    up = pd.to_numeric(features[up_column], errors="coerce").to_numpy(float)
    down = pd.to_numeric(features[down_column], errors="coerce").to_numpy(float)
    momentum = pd.to_numeric(features["ret_12"], errors="coerce").to_numpy(float)
    up_condition = np.isfinite(up) & (up >= up_threshold) & np.isfinite(momentum) & (momentum > 0.0)
    down_condition = np.isfinite(down) & (down >= down_threshold) & np.isfinite(momentum) & (momentum < 0.0)
    up_onset = up_condition & ~np.roll(up_condition, 1)
    down_onset = down_condition & ~np.roll(down_condition, 1)
    up_onset[0] = down_onset[0] = False
    if mapping == "permeability":
        long_active, short_active = up_onset, down_onset
    elif mapping == "fade":
        long_active, short_active = down_onset, up_onset
    else:
        raise KeyError(mapping)
    if flip:
        long_active, short_active = short_active, long_active
    return long_active, short_active


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold_bars: int,
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
            hold_bars=hold_bars,
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
        and stats["select_2023"]["trades"] >= 24
        and stats["select_2023_h1"]["trades"] >= 8
        and stats["select_2023_h2"]["trades"] >= 8
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
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
    )
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    positive_segments = sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS)
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
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
    for bin_width, half_life, event_quantile in itertools.product((0.0005, 0.0010), (288, 864, 2016), (0.90, 0.95)):
        key = (bin_width, half_life, event_quantile)
        field = replay_scar_field(
            inputs,
            dates,
            bin_width=bin_width,
            half_life=half_life,
            contraction_threshold=contraction_thresholds[event_quantile],
        )
        banks[key] = field
        up_threshold = fit_threshold(field["up_scar_ahead"], dates, 0.80, positive_only=True)
        down_threshold = fit_threshold(field["down_scar_below"], dates, 0.80, positive_only=True)
        for mapping, hold in itertools.product(("permeability", "fade"), holds):
            long_active, short_active = scar_signals(
                field,
                up_threshold=up_threshold,
                down_threshold=down_threshold,
                mapping=mapping,
            )
            stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
            rows.append(
                {
                    "bin_width": bin_width,
                    "half_life": half_life,
                    "event_quantile": event_quantile,
                    "contraction_threshold": contraction_thresholds[event_quantile],
                    "query_quantile": 0.80,
                    "up_threshold": up_threshold,
                    "down_threshold": down_threshold,
                    "mapping": mapping,
                    "hold": hold,
                    "raw_events": int((long_active | short_active).sum()),
                    "admitted": admission(stats),
                    "rank": rank_key(stats),
                    "stats": stats,
                }
            )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["admitted"] for row in rows))
    for index, row in enumerate(rows[:12], 1):
        print_stats(
            f"RANK {index} bw{row['bin_width']} hl{row['half_life']} eq{row['event_quantile']} "
            f"{row['mapping']} h{row['hold']} events={row['raw_events']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    key = (top["bin_width"], top["half_life"], top["event_quantile"])
    top_field = banks[key]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    base_long, base_short = scar_signals(
        top_field,
        up_threshold=top["up_threshold"],
        down_threshold=top["down_threshold"],
        mapping=top["mapping"],
    )
    long_active, short_active = scar_signals(
        top_field,
        up_threshold=top["up_threshold"],
        down_threshold=top["down_threshold"],
        mapping=top["mapping"],
        flip=True,
    )
    controls["direction_flip"] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])

    scalar_up_threshold = fit_threshold(top_field["scalar_up"], dates, 0.80, positive_only=True)
    scalar_down_threshold = fit_threshold(top_field["scalar_down"], dates, 0.80, positive_only=True)
    long_active, short_active = scar_signals(
        top_field,
        up_threshold=scalar_up_threshold,
        down_threshold=scalar_down_threshold,
        mapping=top["mapping"],
        scalar=True,
    )
    controls["scalar_collapse"] = simulate(market, dates, long_active, short_active, top["hold"], extremes[top["hold"]])

    for name, mode in (("month_offset_bins", "month_offset"), ("lag12_deposit_bin", "lag12")):
        placebo = replay_scar_field(
            inputs,
            dates,
            bin_width=top["bin_width"],
            half_life=top["half_life"],
            contraction_threshold=top["contraction_threshold"],
            deposit_mode=mode,
        )
        placebo_up_threshold = fit_threshold(placebo["up_scar_ahead"], dates, 0.80, positive_only=True)
        placebo_down_threshold = fit_threshold(placebo["down_scar_below"], dates, 0.80, positive_only=True)
        long_active, short_active = scar_signals(
            placebo,
            up_threshold=placebo_up_threshold,
            down_threshold=placebo_down_threshold,
            mapping=top["mapping"],
        )
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
            "query_timing": "query prior field before depositing completed bar t; enter t+1 open",
            "grid_size": 48,
            "query_quantile": 0.80,
            "query_radius_log_return": 0.003,
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
    Path("results/liquidation_scar_field_alpha_scan_2026-07-13.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
