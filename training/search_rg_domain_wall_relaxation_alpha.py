"""Search a causal renormalization-group domain-wall relaxation alpha."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import SEGMENTS, WINDOWS, load_pre2024, simulate
from training.search_positioning_disagreement_alpha import _future_extreme

BASE_SCALES = (24, 48)
SCORE_QUANTILES = (0.70, 0.80)
HOLDS = (72, 144)
DECISION_MINUTE = 55
FIELD_FLOOR = 0.25
FIXED_POINT_TOLERANCE = 0.10


def directional_field(log_price: pd.Series, returns: pd.Series, scale: int) -> tuple[np.ndarray, np.ndarray]:
    level = log_price.ewm(
        halflife=scale / 2,
        adjust=False,
        min_periods=4 * scale,
    ).mean()
    volatility = (
        returns.pow(2)
        .ewm(halflife=scale / 2, adjust=False, min_periods=4 * scale)
        .mean()
        .pow(0.5)
        * np.sqrt(scale)
    )
    normalized = (level - level.shift(scale)) / volatility.replace(0.0, np.nan)
    return np.tanh(normalized.to_numpy(float) / 2.0), volatility.to_numpy(float)


def build_rg_state(market: pd.DataFrame, dates: pd.Series, *, base_scale: int) -> pd.DataFrame:
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    returns = log_price.diff()
    fields: list[np.ndarray] = []
    volatilities: list[np.ndarray] = []
    for scale in (base_scale, 2 * base_scale, 4 * base_scale, 8 * base_scale):
        field, volatility = directional_field(log_price, returns, scale)
        fields.append(field)
        volatilities.append(volatility)
    values = np.column_stack(fields)
    d0, d1, d2, d3 = values.T
    beta0 = d1 - d0
    beta1 = d2 - d1
    curvature = d2 - 2.0 * d1 + d0
    coarse_side = np.nan_to_num(np.sign(d3)).astype(np.int8)
    finite = np.isfinite(values).all(axis=1)
    fixed_point = (
        finite
        & (np.sign(d2) == np.sign(d3))
        & (np.abs(d2) >= FIELD_FLOOR)
        & (np.abs(d3) >= FIELD_FLOOR)
        & (np.abs(d3 - d2) <= FIXED_POINT_TOLERANCE)
    )
    domain_wall = finite & (np.sign(d0) == -np.sign(d3)) & (np.abs(d0) >= FIELD_FLOOR)
    relaxation = (coarse_side * beta0 > 0.0) & (coarse_side * beta1 > 0.0)
    candidate = fixed_point & domain_wall & relaxation & dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool)
    score = np.minimum(np.abs(d0), np.abs(d2)) * (
        np.abs(beta0) + np.abs(beta1)
    ) * np.abs(curvature)
    score[~candidate] = np.nan
    return pd.DataFrame(
        {
            "d0": d0,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "beta0": beta0,
            "beta1": beta1,
            "curvature": curvature,
            "coarse_side": coarse_side,
            "fixed_point": fixed_point,
            "domain_wall": domain_wall,
            "relaxation": relaxation,
            "candidate": candidate,
            "score": score,
            "volterm": np.asarray(volatilities[0]) / np.where(
                np.asarray(volatilities[2]) > 0.0, np.asarray(volatilities[2]), np.nan
            ),
        }
    )


def fit_threshold(score: np.ndarray, dates: pd.Series, quantile: float) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = np.asarray(score, dtype=float)[fit]
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no fit RG candidates")
    return float(np.quantile(values, quantile))


def masks(state: pd.DataFrame, threshold: float, *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    score = state["score"].to_numpy(float)
    active = state["candidate"].to_numpy(bool) & np.isfinite(score) & (score >= threshold)
    side = state["coarse_side"].to_numpy(np.int8)
    if flip:
        side = -side
    return active & (side > 0), active & (side < 0)


def admission(stats: dict[str, dict[str, Any]]) -> bool:
    return bool(
        stats["fit"]["return_pct"] > 0.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] >= 0.0
        and stats["select_2023_h2"]["return_pct"] >= 0.0
        and stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    supported = (
        stats["select_2023"]["trades"] >= 20
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )
    core = [stats[name]["ratio"] for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")]
    return (
        admission(stats),
        supported,
        stats["select_2023"]["trades"],
        min(core),
        float(np.median(core)),
    )


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
    states = {base: build_rg_state(market, dates, base_scale=base) for base in BASE_SCALES}
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    for base, quantile, hold in itertools.product(BASE_SCALES, SCORE_QUANTILES, HOLDS):
        state = states[base]
        threshold = fit_threshold(state["score"].to_numpy(float), dates, quantile)
        long_active, short_active = masks(state, threshold)
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "base_scale": base,
                "score_quantile": quantile,
                "hold": hold,
                "threshold": threshold,
                "raw_signals": int((long_active | short_active).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    for index, row in enumerate(rows, 1):
        print_stats(
            f"RANK {index} base{row['base_scale']} q{row['score_quantile']} h{row['hold']} "
            f"raw={row['raw_signals']} rank={row['rank']}",
            row["stats"],
        )
    top = rows[0]
    state = states[top["base_scale"]]
    long_active, short_active = masks(state, top["threshold"])
    flip_long, flip_short = masks(state, top["threshold"], flip=True)
    controls = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]),
    }
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
        for bp in (0, 3, 6, 10)
    }
    output = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "mechanism": "causal EWM dyadic directional-field fixed point, domain wall and relaxation curvature",
            "grid_size": len(rows),
            "grid": "2 base scales x 2 fit score tails x 2 holds",
            "precursor_disclosure": "a separate 16-policy hard-sign domain-wall propagation/contraction probe was weak before this final architect-reviewed relaxation grid; every such setting is frozen",
            "entry": "completed minute-55 signal enters next minute-00 open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
        },
        "candidate_counts": {
            str(base): int(states[base]["candidate"].sum()) for base in BASE_SCALES
        },
        "rows": rows,
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/rg_domain_wall_relaxation_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
