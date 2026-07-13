"""Search a dynamic causal-cone breach age-front alpha.

The static causal-cone family only measures how much of the prior-anchor
ensemble is breached.  This experiment instead tracks *which generations* of
anchors are being invalidated.  A coherent repricing wave should push the
weighted breach frontier from young anchors toward older anchors while breach
mass grows.  That outward front is followed in the rupture direction.

The cone geometry is preregistered from the preceding experiment: 168 hourly
anchors (2016 five-minute bars), width 2, and volatility frozen strictly at
each anchor.  Only lag and fixed holding period form the four-policy grid.
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
from training.search_frozen_causal_cone_rupture_alpha import (
    ANCHOR_STRIDE,
    CONE_WIDTH,
    DECISION_MINUTE,
    prior_volatility,
)
from training.search_positioning_disagreement_alpha import _future_extreme

ANCHOR_HORIZON = 2016
LAGS = (12, 72)
HOLDS = (72, 144)
BREACH_FRONT_QUANTILE = 0.80
SIGNAL_QUANTILE = 0.80
OLD_AGE_FRACTION = 0.50


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return float("nan")
    ordered = np.argsort(values[finite], kind="stable")
    selected_values = values[finite][ordered]
    selected_weights = weights[finite][ordered]
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * selected_weights.sum()
    position = min(int(np.searchsorted(np.cumsum(selected_weights), cutoff, side="left")), len(ordered) - 1)
    return float(selected_values[position])


def breach_age_profile(
    normalized_displacement: np.ndarray,
    normalized_age: np.ndarray,
) -> dict[str, float]:
    z = np.asarray(normalized_displacement, dtype=float)
    age = np.asarray(normalized_age, dtype=float)
    finite = np.isfinite(z) & np.isfinite(age)
    z = z[finite]
    age = age[finite]
    if not len(z):
        raise ValueError("no finite cone displacements")
    result: dict[str, float] = {}
    for name, weights in (
        ("upper", np.maximum(z - CONE_WIDTH, 0.0)),
        ("lower", np.maximum(-z - CONE_WIDTH, 0.0)),
    ):
        total = float(weights.sum())
        result[f"{name}_mass"] = float(weights.mean())
        if total <= 0.0:
            result[f"{name}_front"] = float("nan")
            result[f"{name}_center"] = float("nan")
            result[f"{name}_old_share"] = float("nan")
            continue
        result[f"{name}_front"] = weighted_quantile(age, weights, BREACH_FRONT_QUANTILE)
        result[f"{name}_center"] = float(np.dot(age, weights) / total)
        result[f"{name}_old_share"] = float(weights[age >= OLD_AGE_FRACTION].sum() / total)
    result["side"] = float(
        1 if result["upper_mass"] > result["lower_mass"] else (-1 if result["lower_mass"] > result["upper_mass"] else 0)
    )
    return result


def build_age_front_state(
    market: pd.DataFrame,
    dates: pd.Series,
    *,
    frozen_anchor_volatility: bool = True,
    reverse_age_order: bool = False,
) -> pd.DataFrame:
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    price = log_price.to_numpy(float)
    volatility = prior_volatility(log_price)
    ages = np.arange(ANCHOR_STRIDE, ANCHOR_HORIZON + 1, ANCHOR_STRIDE, dtype=np.int64)
    normalized_ages = ages / float(ANCHOR_HORIZON)
    if reverse_age_order:
        normalized_ages = normalized_ages[::-1].copy()
    names = (
        "upper_mass",
        "lower_mass",
        "upper_front",
        "lower_front",
        "upper_center",
        "lower_center",
        "upper_old_share",
        "lower_old_share",
        "side",
    )
    fields = {name: np.full(len(market), np.nan, dtype=float) for name in names}
    anchor_count = np.zeros(len(market), dtype=np.int32)
    decision = dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool)
    for position in np.flatnonzero(decision):
        anchor_positions = position - ages
        in_bounds = anchor_positions >= 0
        if not in_bounds.any() or not np.isfinite(price[position]):
            continue
        anchors = anchor_positions[in_bounds]
        anchor_ages = ages[in_bounds]
        age_fraction = normalized_ages[in_bounds]
        scale = volatility[anchors] if frozen_anchor_volatility else np.full(len(anchors), volatility[position])
        finite = np.isfinite(price[anchors]) & np.isfinite(scale) & (scale > 0.0)
        if finite.sum() < max(8, len(ages) // 2):
            continue
        z = (price[position] - price[anchors[finite]]) / (
            scale[finite] * np.sqrt(anchor_ages[finite])
        )
        profile = breach_age_profile(z, age_fraction[finite])
        for name in names:
            fields[name][position] = profile[name]
        anchor_count[position] = int(finite.sum())
    fields["anchor_count"] = anchor_count
    fields["decision"] = decision
    return pd.DataFrame(fields)


def _lag_float(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.r_[np.full(bars, np.nan), values[:-bars]]


def _lag_int(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int8)
    return np.r_[np.zeros(bars, dtype=np.int8), values[:-bars]]


def _select_side(upper: np.ndarray, lower: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.where(side > 0, upper, np.where(side < 0, lower, np.nan))


def build_front_dynamics(state: pd.DataFrame, *, lag: int) -> pd.DataFrame:
    upper_mass = state["upper_mass"].to_numpy(float)
    lower_mass = state["lower_mass"].to_numpy(float)
    side = np.nan_to_num(state["side"].to_numpy(float)).astype(np.int8)
    previous_side = _lag_int(side, lag)
    same_side = (side != 0) & (side == previous_side)

    current_mass = _select_side(upper_mass, lower_mass, side)
    previous_mass = _select_side(_lag_float(upper_mass, lag), _lag_float(lower_mass, lag), side)
    current_front = _select_side(
        state["upper_front"].to_numpy(float), state["lower_front"].to_numpy(float), side
    )
    previous_front = _select_side(
        _lag_float(state["upper_front"].to_numpy(float), lag),
        _lag_float(state["lower_front"].to_numpy(float), lag),
        side,
    )
    current_center = _select_side(
        state["upper_center"].to_numpy(float), state["lower_center"].to_numpy(float), side
    )
    previous_center = _select_side(
        _lag_float(state["upper_center"].to_numpy(float), lag),
        _lag_float(state["lower_center"].to_numpy(float), lag),
        side,
    )
    current_old_share = _select_side(
        state["upper_old_share"].to_numpy(float), state["lower_old_share"].to_numpy(float), side
    )
    previous_old_share = _select_side(
        _lag_float(state["upper_old_share"].to_numpy(float), lag),
        _lag_float(state["lower_old_share"].to_numpy(float), lag),
        side,
    )

    front_velocity = current_front - previous_front
    center_velocity = current_center - previous_center
    old_share_velocity = current_old_share - previous_old_share
    mass_growth = np.log1p(current_mass) - np.log1p(previous_mass)
    finite = (
        same_side
        & np.isfinite(front_velocity)
        & np.isfinite(center_velocity)
        & np.isfinite(old_share_velocity)
        & np.isfinite(mass_growth)
    )
    propagation = finite & (front_velocity > 0.0) & (center_velocity > 0.0) & (mass_growth > 0.0)
    retreat = finite & (front_velocity < 0.0) & (center_velocity < 0.0) & (mass_growth < 0.0)
    propagation_score = np.where(
        propagation,
        np.sqrt(np.maximum(front_velocity, 0.0) * np.maximum(mass_growth, 0.0))
        * (1.0 + np.maximum(old_share_velocity, 0.0)),
        np.nan,
    )
    retreat_score = np.where(
        retreat,
        np.sqrt(np.maximum(-front_velocity, 0.0) * np.maximum(-mass_growth, 0.0))
        * (1.0 + np.maximum(-old_share_velocity, 0.0)),
        np.nan,
    )
    return pd.DataFrame(
        {
            "side": side,
            "same_side": same_side,
            "current_mass": current_mass,
            "front_velocity": front_velocity,
            "center_velocity": center_velocity,
            "old_share_velocity": old_share_velocity,
            "mass_growth": mass_growth,
            "propagation_score": propagation_score,
            "retreat_score": retreat_score,
        }
    )


def fit_positive_threshold(values: np.ndarray, dates: pd.Series, quantile: float = SIGNAL_QUANTILE) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    selected = np.asarray(values, dtype=float)[fit]
    selected = selected[np.isfinite(selected) & (selected > 0.0)]
    if not len(selected):
        raise ValueError("no positive fit front scores")
    return float(np.quantile(selected, quantile))


def policy_masks(
    dynamics: pd.DataFrame,
    decision: np.ndarray,
    threshold: float,
    *,
    mode: str = "propagation",
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"propagation", "retreat"}:
        raise ValueError(f"unknown front mode: {mode}")
    score = dynamics[f"{mode}_score"].to_numpy(float)
    side = dynamics["side"].to_numpy(np.int8)
    direction = side if mode == "propagation" else -side
    if flip:
        direction = -direction
    active = np.asarray(decision, dtype=bool) & np.isfinite(score) & (score >= threshold) & (direction != 0)
    return active & (direction > 0), active & (direction < 0)


def simple_control_masks(
    dynamics: pd.DataFrame,
    decision: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    side = dynamics["side"].to_numpy(np.int8)
    score = np.asarray(values, dtype=float)
    active = np.asarray(decision, dtype=bool) & dynamics["same_side"].to_numpy(bool) & np.isfinite(score) & (score >= threshold)
    return active & (side > 0), active & (side < 0)


def lag_boolean(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    return np.r_[np.zeros(bars, dtype=bool), values[:-bars]]


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


def evaluate_state(
    market: pd.DataFrame,
    dates: pd.Series,
    state: pd.DataFrame,
    *,
    lag: int,
    hold: int,
    extremes: tuple[np.ndarray, np.ndarray],
    side_cost: float = 0.0006,
) -> tuple[dict[str, dict[str, Any]], float, pd.DataFrame, np.ndarray, np.ndarray]:
    dynamics = build_front_dynamics(state, lag=lag)
    threshold = fit_positive_threshold(dynamics["propagation_score"].to_numpy(float), dates)
    long_active, short_active = policy_masks(
        dynamics, state["decision"].to_numpy(bool), threshold
    )
    stats = simulate(
        market,
        dates,
        long_active,
        short_active,
        hold,
        extremes,
        side_cost=side_cost,
    )
    return stats, threshold, dynamics, long_active, short_active


def main() -> None:
    market, dates = load_pre2024()
    state = build_age_front_state(market, dates)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    for lag, hold in itertools.product(LAGS, HOLDS):
        stats, threshold, dynamics, long_active, short_active = evaluate_state(
            market, dates, state, lag=lag, hold=hold, extremes=extremes[hold]
        )
        rows.append(
            {
                "lag": lag,
                "hold": hold,
                "signal_quantile": SIGNAL_QUANTILE,
                "score_threshold": threshold,
                "raw_signals": int((long_active | short_active).sum()),
                "coherent_front_observations": int(np.isfinite(dynamics["propagation_score"]).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    print("candidates", len(rows), "admitted", sum(row["prelim_admitted"] for row in rows))
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} lag{row['lag']} hold{row['hold']} raw={row['raw_signals']} rank={row['rank']}",
            row["stats"],
        )

    top = rows[0]
    top_stats, threshold, dynamics, long_active, short_active = evaluate_state(
        market,
        dates,
        state,
        lag=top["lag"],
        hold=top["hold"],
        extremes=extremes[top["hold"]],
    )
    flip_long, flip_short = policy_masks(
        dynamics,
        state["decision"].to_numpy(bool),
        threshold,
        flip=True,
    )
    retreat_threshold = fit_positive_threshold(dynamics["retreat_score"].to_numpy(float), dates)
    retreat_long, retreat_short = policy_masks(
        dynamics,
        state["decision"].to_numpy(bool),
        retreat_threshold,
        mode="retreat",
    )
    mass_growth = dynamics["mass_growth"].to_numpy(float)
    mass_threshold = fit_positive_threshold(mass_growth, dates)
    mass_long, mass_short = simple_control_masks(
        dynamics, state["decision"].to_numpy(bool), mass_growth, mass_threshold
    )
    front_velocity = dynamics["front_velocity"].to_numpy(float)
    front_threshold = fit_positive_threshold(front_velocity, dates)
    front_long, front_short = simple_control_masks(
        dynamics, state["decision"].to_numpy(bool), front_velocity, front_threshold
    )
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]
        ),
        "retreat_reversal": simulate(
            market, dates, retreat_long, retreat_short, top["hold"], extremes[top["hold"]]
        ),
        "mass_growth_only": simulate(
            market, dates, mass_long, mass_short, top["hold"], extremes[top["hold"]]
        ),
        "front_velocity_only": simulate(
            market, dates, front_long, front_short, top["hold"], extremes[top["hold"]]
        ),
        "signal_delay_7d": simulate(
            market,
            dates,
            lag_boolean(long_active, ANCHOR_HORIZON),
            lag_boolean(short_active, ANCHOR_HORIZON),
            top["hold"],
            extremes[top["hold"]],
        ),
    }
    current_state = build_age_front_state(market, dates, frozen_anchor_volatility=False)
    current_stats, current_threshold, _, _, _ = evaluate_state(
        market,
        dates,
        current_state,
        lag=top["lag"],
        hold=top["hold"],
        extremes=extremes[top["hold"]],
    )
    controls["current_volatility_rewrites_anchor_scales"] = current_stats
    reversed_age_state = build_age_front_state(market, dates, reverse_age_order=True)
    reversed_stats, reversed_threshold, _, _, _ = evaluate_state(
        market,
        dates,
        reversed_age_state,
        lag=top["lag"],
        hold=top["hold"],
        extremes=extremes[top["hold"]],
    )
    controls["reversed_anchor_age_order"] = reversed_stats
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
            "mechanism": "a coherent rupture follows the dominant side only when breach mass grows and the weighted q80/center of breached anchor ages propagate outward",
            "fixed_geometry": {
                "anchor_horizon": ANCHOR_HORIZON,
                "hourly_anchor_count": ANCHOR_HORIZON // ANCHOR_STRIDE,
                "cone_width": CONE_WIDTH,
                "breach_front_quantile": BREACH_FRONT_QUANTILE,
                "old_age_fraction": OLD_AGE_FRACTION,
            },
            "grid_size": len(rows),
            "grid": "2 preregistered lags (1h/6h) x 2 holds (6h/12h); q80 score tail fixed",
            "entry": "completed minute-55 signal enters next minute-00 open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "all pre-2024 rows are exploratory; 2023 is inspected internal selection and 2024+ remained sealed",
        },
        "state_summary": {
            "valid_decisions": int(np.isfinite(state["side"]).sum()),
            "nonzero_side_decisions": int((state["side"].fillna(0.0) != 0.0).sum()),
            "median_anchor_count": float(state.loc[np.isfinite(state["side"]), "anchor_count"].median()),
        },
        "rows": rows,
        "top_recomputed": top_stats,
        "controls": controls,
        "control_thresholds": {
            "retreat": retreat_threshold,
            "mass_growth": mass_threshold,
            "front_velocity": front_threshold,
            "current_volatility_propagation": current_threshold,
            "reversed_anchor_age_order": reversed_threshold,
        },
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/causal_cone_age_front_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
