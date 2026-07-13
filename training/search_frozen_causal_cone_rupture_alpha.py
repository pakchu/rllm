"""Search a causal frozen-volatility diffusion-cone rupture alpha.

Each prior hourly anchor projects a square-root-time price envelope using only
the volatility known at that anchor.  A current completed price that breaches
many independently frozen envelopes in one direction forms an information-front
mass.  The static policy follows the dominant rupture direction.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import (
    SEGMENTS,
    WINDOWS,
    admission,
    load_pre2024,
    rank_key,
    simulate,
)
from training.search_positioning_disagreement_alpha import _future_extreme

ANCHOR_HORIZONS = (576, 2016)
SCORE_QUANTILES = (0.80, 0.90)
HOLDS = (72, 144)
ANCHOR_STRIDE = 12
DECISION_MINUTE = 55
CONE_WIDTH = 2.0
VOL_WINDOW = 2016
VOL_MIN_PERIODS = 1008


def prior_volatility(log_price: pd.Series) -> np.ndarray:
    return (
        log_price.diff()
        .shift(1)
        .rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS)
        .std(ddof=0)
        .replace(0.0, np.nan)
        .to_numpy(float)
    )


def cone_components(normalized_displacement: np.ndarray) -> dict[str, float]:
    z = np.asarray(normalized_displacement, dtype=float)
    z = z[np.isfinite(z)]
    if not len(z):
        raise ValueError("no finite cone displacements")
    upper_excess = np.maximum(z - CONE_WIDTH, 0.0)
    lower_excess = np.maximum(-z - CONE_WIDTH, 0.0)
    upper_mass = float(upper_excess.mean())
    lower_mass = float(lower_excess.mean())
    return {
        "upper_mass": upper_mass,
        "lower_mass": lower_mass,
        "score": max(upper_mass, lower_mass),
        "side": 1.0 if upper_mass > lower_mass else (-1.0 if lower_mass > upper_mass else 0.0),
        "breach_fraction": float(max(np.mean(z > CONE_WIDTH), np.mean(z < -CONE_WIDTH))),
    }


def build_cone_state(
    market: pd.DataFrame,
    dates: pd.Series,
    *,
    horizon: int,
    frozen_anchor_volatility: bool = True,
) -> pd.DataFrame:
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    price = log_price.to_numpy(float)
    volatility = prior_volatility(log_price)
    ages = np.arange(ANCHOR_STRIDE, horizon + 1, ANCHOR_STRIDE, dtype=np.int64)
    fields = {
        name: np.full(len(market), np.nan, dtype=float)
        for name in ("upper_mass", "lower_mass", "score", "side", "breach_fraction", "simple_z")
    }
    anchor_count = np.zeros(len(market), dtype=np.int32)
    decision = dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool)
    for position in np.flatnonzero(decision):
        anchor_positions = position - ages
        in_bounds = anchor_positions >= 0
        if not in_bounds.any() or not np.isfinite(price[position]):
            continue
        anchors = anchor_positions[in_bounds]
        anchor_ages = ages[in_bounds]
        if frozen_anchor_volatility:
            scale = volatility[anchors]
        else:
            scale = np.full(len(anchors), volatility[position], dtype=float)
        finite = np.isfinite(price[anchors]) & np.isfinite(scale) & (scale > 0.0)
        if finite.sum() < max(8, len(ages) // 2):
            continue
        z = (price[position] - price[anchors[finite]]) / (
            scale[finite] * np.sqrt(anchor_ages[finite])
        )
        values = cone_components(z)
        for name in ("upper_mass", "lower_mass", "score", "side", "breach_fraction"):
            fields[name][position] = values[name]
        anchor_count[position] = int(finite.sum())
        if position >= horizon and np.isfinite(volatility[position]) and volatility[position] > 0.0:
            fields["simple_z"][position] = (
                price[position] - price[position - horizon]
            ) / (volatility[position] * np.sqrt(horizon))
    fields["anchor_count"] = anchor_count
    fields["decision"] = decision
    return pd.DataFrame(fields)


def fit_threshold(score: np.ndarray, dates: pd.Series, quantile: float) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = np.asarray(score, dtype=float)[fit]
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite fit cone scores")
    return float(np.quantile(values, quantile))


def policy_masks(
    score: np.ndarray,
    side: np.ndarray,
    decision: np.ndarray,
    threshold: float,
    *,
    flip: bool = False,
    onset_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(score, dtype=float)
    direction = np.nan_to_num(np.sign(np.asarray(side, dtype=float))).astype(np.int8)
    active = np.asarray(decision, dtype=bool) & np.isfinite(values) & (values >= threshold) & (direction != 0)
    if onset_only:
        previous_active = np.r_[np.zeros(ANCHOR_STRIDE, dtype=bool), active[:-ANCHOR_STRIDE]]
        previous_side = np.r_[np.zeros(ANCHOR_STRIDE, dtype=np.int8), direction[:-ANCHOR_STRIDE]]
        active &= (~previous_active) | (previous_side != direction)
    if flip:
        direction = -direction
    return active & (direction > 0), active & (direction < 0)


def lag_mask(values: np.ndarray, bars: int) -> np.ndarray:
    return np.r_[np.zeros(bars, dtype=bool), np.asarray(values, dtype=bool)[:-bars]]


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
    states = {
        horizon: build_cone_state(market, dates, horizon=horizon)
        for horizon in ANCHOR_HORIZONS
    }
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    for horizon, quantile, hold in itertools.product(ANCHOR_HORIZONS, SCORE_QUANTILES, HOLDS):
        state = states[horizon]
        threshold = fit_threshold(state["score"].to_numpy(float), dates, quantile)
        long_active, short_active = policy_masks(
            state["score"].to_numpy(float),
            state["side"].to_numpy(float),
            state["decision"].to_numpy(bool),
            threshold,
        )
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "anchor_horizon": horizon,
                "score_quantile": quantile,
                "hold": hold,
                "score_threshold": threshold,
                "raw_signals": int((long_active | short_active).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} horizon{row['anchor_horizon']} q{row['score_quantile']} "
            f"h{row['hold']} raw={row['raw_signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    state = states[top["anchor_horizon"]]
    score = state["score"].to_numpy(float)
    side = state["side"].to_numpy(float)
    decision = state["decision"].to_numpy(bool)
    long_active, short_active = policy_masks(score, side, decision, top["score_threshold"])
    flip_long, flip_short = policy_masks(score, side, decision, top["score_threshold"], flip=True)
    onset_long, onset_short = policy_masks(score, side, decision, top["score_threshold"], onset_only=True)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]),
        "onset_only": simulate(market, dates, onset_long, onset_short, top["hold"], extremes[top["hold"]]),
    }
    fraction = state["breach_fraction"].to_numpy(float)
    fraction_threshold = fit_threshold(fraction, dates, top["score_quantile"])
    fraction_long, fraction_short = policy_masks(fraction, side, decision, fraction_threshold)
    controls["breach_fraction_only"] = simulate(
        market, dates, fraction_long, fraction_short, top["hold"], extremes[top["hold"]]
    )
    current_state = build_cone_state(
        market,
        dates,
        horizon=top["anchor_horizon"],
        frozen_anchor_volatility=False,
    )
    current_threshold = fit_threshold(current_state["score"].to_numpy(float), dates, top["score_quantile"])
    current_long, current_short = policy_masks(
        current_state["score"].to_numpy(float),
        current_state["side"].to_numpy(float),
        current_state["decision"].to_numpy(bool),
        current_threshold,
    )
    controls["current_volatility_rewrites_anchors"] = simulate(
        market, dates, current_long, current_short, top["hold"], extremes[top["hold"]]
    )
    simple_z = state["simple_z"].to_numpy(float)
    simple_threshold = fit_threshold(np.abs(simple_z), dates, top["score_quantile"])
    simple_long, simple_short = policy_masks(
        np.abs(simple_z), simple_z, decision, simple_threshold
    )
    controls["single_horizon_return_z"] = simulate(
        market, dates, simple_long, simple_short, top["hold"], extremes[top["hold"]]
    )
    for hours in (1, 6, 24 * 7):
        bars = hours * ANCHOR_STRIDE
        controls[f"signal_lag_{hours}h"] = simulate(
            market,
            dates,
            lag_mask(long_active, bars),
            lag_mask(short_active, bars),
            top["hold"],
            extremes[top["hold"]],
        )
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
            "mechanism": "hourly prior anchors project sqrt-time diffusion cones using volatility frozen strictly at each anchor; current multi-cone excess mass defines rupture direction",
            "grid_size": len(rows),
            "grid": "2 anchor horizons x 2 fit-only mass tails x 2 holds",
            "cone_width": CONE_WIDTH,
            "anchor_stride": ANCHOR_STRIDE,
            "entry": "completed minute-55 signal enters next minute-00 open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "2023 is inspected internal selection; 2024+ remained sealed",
        },
        "state_summary": {
            str(horizon): {
                "valid_decisions": int(np.isfinite(state["score"]).sum()),
                "positive_mass_decisions": int((state["score"] > 0.0).sum()),
                "median_anchor_count": float(state.loc[np.isfinite(state["score"]), "anchor_count"].median()),
            }
            for horizon, state in states.items()
        },
        "rows": rows,
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/frozen_causal_cone_rupture_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
