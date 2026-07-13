"""Evaluate a causal Preisach flow-scar avalanche alpha.

A fixed lattice of persistent price relays stores path-dependent domain state.
Each relay accumulates normalized aggressive taker flow until it flips.  A
multi-relay Barkhausen avalanche may release flow that had opposed the new
price domain; the single preregistered policy follows that release direction.

Unlike a breakout stack, relay state persists inside each deadband.  Unlike a
dual intrinsic clock, the feature does not count recent events.  Unlike a path
signature, the memory is not truncated at a rolling-area boundary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import (
    SEGMENTS,
    WINDOWS,
    admission,
    load_pre2024,
    simulate,
)
from training.search_positioning_disagreement_alpha import _future_extreme

FIELD_WINDOW = 144
FIELD_MIN_PERIODS = 72
FLOW_DENOM_WINDOW = 12
HOLD = 72
SIGNAL_QUANTILE = 0.80
MIN_COHERENCE = 0.67
DELAY_7D = 2016

BETA_LEVELS = (-2.0, -1.5, -1.0, -0.5, 0.0)
ALPHA_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)
RELAY_GAPS = (1.0, 1.5, 2.0)


def relay_lattice() -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (alpha, beta)
        for beta in BETA_LEVELS
        for alpha in ALPHA_LEVELS
        if alpha > beta and round(alpha - beta, 10) in RELAY_GAPS
    ]
    return (
        np.asarray([alpha for alpha, _ in pairs], dtype=float),
        np.asarray([beta for _, beta in pairs], dtype=float),
    )


ALPHAS, BETAS = relay_lattice()
MIN_AVALANCHE = math.ceil(0.20 * len(ALPHAS))


def build_inputs(market: pd.DataFrame) -> pd.DataFrame:
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0))
    log_return = log_price.diff()
    prior_center = log_price.shift(1).rolling(
        FIELD_WINDOW, min_periods=FIELD_MIN_PERIODS
    ).median()
    prior_scale = (
        log_return.shift(1)
        .rolling(FIELD_WINDOW, min_periods=FIELD_MIN_PERIODS)
        .std(ddof=0)
        * np.sqrt(FIELD_WINDOW)
    ).replace(0.0, np.nan)
    price_field = ((log_price - prior_center) / prior_scale).clip(-8.0, 8.0)

    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    prior_hour_quote = quote.shift(1).rolling(
        FLOW_DENOM_WINDOW, min_periods=FLOW_DENOM_WINDOW
    ).sum()
    signed_flow = ((2.0 * taker_buy - quote) / prior_hour_quote.replace(0.0, np.nan)).clip(-2.0, 2.0)
    return pd.DataFrame(
        {
            "price_field": price_field,
            "prior_price_center": prior_center,
            "prior_price_scale": prior_scale,
            "signed_flow": signed_flow,
            "prior_hour_quote": prior_hour_quote,
        }
    ).replace([np.inf, -np.inf], np.nan)


def run_relay_ensemble(
    price_field: np.ndarray,
    signed_flow: np.ndarray,
    *,
    retain_inside_state: bool = True,
) -> pd.DataFrame:
    x = np.asarray(price_field, dtype=float)
    flow = np.asarray(signed_flow, dtype=float)
    if len(x) != len(flow):
        raise ValueError("price and flow paths must have equal length")
    relay_count = len(ALPHAS)
    state = np.zeros(relay_count, dtype=np.int8)
    scar_sum = np.zeros(relay_count, dtype=float)
    scar_age = np.zeros(relay_count, dtype=np.int64)
    started = False
    fields = {
        name: np.full(len(x), np.nan, dtype=float)
        for name in (
            "direction",
            "avalanche_count",
            "coherence",
            "opposing_pressure",
            "same_pressure",
            "current_chase",
            "release_score",
            "same_release_score",
            "scarless_score",
            "chase_score",
            "median_relay_age",
            "magnetization",
        )
    }
    for position in range(len(x)):
        if np.isfinite(x[position]):
            started = True
            old_state = state.copy()
            if retain_inside_state:
                new_state = np.where(
                    x[position] >= ALPHAS,
                    1,
                    np.where(x[position] <= BETAS, -1, state),
                ).astype(np.int8)
                initialization = (old_state == 0) & (new_state != 0)
                switched = (old_state != 0) & (new_state != old_state)
            else:
                new_state = np.where(
                    x[position] >= ALPHAS,
                    1,
                    np.where(x[position] <= BETAS, -1, 0),
                ).astype(np.int8)
                initialization = np.zeros(relay_count, dtype=bool)
                switched = (new_state != 0) & (new_state != old_state)

            if switched.any():
                switch_direction = new_state[switched]
                direction = int(np.sign(switch_direction.sum()))
                count = int(switched.sum())
                coherence = float(abs(switch_direction.sum()) / count)
                normalized_scar = scar_sum[switched] / np.sqrt(
                    np.maximum(scar_age[switched], 1)
                )
                opposing = float(np.maximum(-direction * normalized_scar, 0.0).mean())
                same = float(np.maximum(direction * normalized_scar, 0.0).mean())
                current_flow = flow[position] if np.isfinite(flow[position]) else 0.0
                chase = float(max(0.0, direction * current_flow))
                fields["direction"][position] = direction
                fields["avalanche_count"][position] = count
                fields["coherence"][position] = coherence
                fields["opposing_pressure"][position] = opposing
                fields["same_pressure"][position] = same
                fields["current_chase"][position] = chase
                fields["release_score"][position] = opposing / (1.0 + chase)
                fields["same_release_score"][position] = same / (1.0 + chase)
                fields["scarless_score"][position] = count * coherence
                fields["chase_score"][position] = chase
                fields["median_relay_age"][position] = float(np.median(scar_age[switched]))

            reset = switched | initialization
            scar_sum[reset] = 0.0
            scar_age[reset] = 0
            state = new_state
            fields["magnetization"][position] = float(state.mean())

        if started:
            if np.isfinite(flow[position]):
                scar_sum += flow[position]
            scar_age += 1
    return pd.DataFrame(fields)


def qualifying_events(state: pd.DataFrame) -> np.ndarray:
    count = state["avalanche_count"].to_numpy(float)
    coherence = state["coherence"].to_numpy(float)
    direction = state["direction"].to_numpy(float)
    return (
        np.isfinite(count)
        & np.isfinite(coherence)
        & (count >= MIN_AVALANCHE)
        & (coherence >= MIN_COHERENCE)
        & (direction != 0.0)
    )


def fit_score_threshold(
    state: pd.DataFrame,
    dates: pd.Series,
    column: str,
    *,
    quantile: float = SIGNAL_QUANTILE,
) -> float:
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = state[column].to_numpy(float)
    selected = values[fit & qualifying_events(state) & np.isfinite(values) & (values > 0.0)]
    if not len(selected):
        raise ValueError(f"no positive fit scores for {column}")
    return float(np.quantile(selected, quantile))


def policy_masks(
    state: pd.DataFrame,
    threshold: float,
    *,
    column: str = "release_score",
    flip: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    score = state[column].to_numpy(float)
    direction = np.nan_to_num(state["direction"].to_numpy(float)).astype(np.int8)
    if flip:
        direction = -direction
    active = qualifying_events(state) & np.isfinite(score) & (score >= threshold)
    return active & (direction > 0), active & (direction < 0)


def lag_values(values: np.ndarray, bars: int, *, fill: float = 0.0) -> np.ndarray:
    values = np.asarray(values)
    return np.r_[np.full(bars, fill, dtype=values.dtype), values[:-bars]]


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
    inputs = build_inputs(market)
    state = run_relay_ensemble(
        inputs["price_field"].to_numpy(float),
        inputs["signed_flow"].to_numpy(float),
    )
    threshold = fit_score_threshold(state, dates, "release_score")
    long_active, short_active = policy_masks(state, threshold)
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD, "max"),
    )
    stats = simulate(market, dates, long_active, short_active, HOLD, extremes)
    print_stats("PRIMARY Preisach opposing-flow scar release", stats)

    flip_long, flip_short = policy_masks(state, threshold, flip=True)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(market, dates, flip_long, flip_short, HOLD, extremes),
        "signal_delay_1bar": simulate(
            market,
            dates,
            lag_values(long_active, 1, fill=False),
            lag_values(short_active, 1, fill=False),
            HOLD,
            extremes,
        ),
        "signal_delay_1h": simulate(
            market,
            dates,
            lag_values(long_active, 12, fill=False),
            lag_values(short_active, 12, fill=False),
            HOLD,
            extremes,
        ),
        "signal_delay_7d": simulate(
            market,
            dates,
            lag_values(long_active, DELAY_7D, fill=False),
            lag_values(short_active, DELAY_7D, fill=False),
            HOLD,
            extremes,
        ),
    }
    control_thresholds: dict[str, float] = {}
    for name, column in (
        ("same_direction_scar", "same_release_score"),
        ("scarless_avalanche", "scarless_score"),
        ("current_flow_chase", "chase_score"),
    ):
        control_threshold = fit_score_threshold(state, dates, column)
        control_long, control_short = policy_masks(state, control_threshold, column=column)
        controls[name] = simulate(market, dates, control_long, control_short, HOLD, extremes)
        control_thresholds[name] = control_threshold

    memoryless_state = run_relay_ensemble(
        inputs["price_field"].to_numpy(float),
        inputs["signed_flow"].to_numpy(float),
        retain_inside_state=False,
    )
    memoryless_threshold = fit_score_threshold(memoryless_state, dates, "release_score")
    memoryless_long, memoryless_short = policy_masks(memoryless_state, memoryless_threshold)
    controls["memory_erased_threshold_entries"] = simulate(
        market, dates, memoryless_long, memoryless_short, HOLD, extremes
    )
    control_thresholds["memory_erased_threshold_entries"] = memoryless_threshold

    delayed_flow = lag_values(inputs["signed_flow"].to_numpy(float), DELAY_7D, fill=np.nan)
    delayed_flow_state = run_relay_ensemble(
        inputs["price_field"].to_numpy(float), delayed_flow
    )
    delayed_flow_threshold = fit_score_threshold(delayed_flow_state, dates, "release_score")
    delayed_flow_long, delayed_flow_short = policy_masks(delayed_flow_state, delayed_flow_threshold)
    controls["flow_scar_delayed_7d"] = simulate(
        market, dates, delayed_flow_long, delayed_flow_short, HOLD, extremes
    )
    control_thresholds["flow_scar_delayed_7d"] = delayed_flow_threshold
    for name, control_stats in controls.items():
        print_stats("CONTROL " + name, control_stats)

    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            long_active,
            short_active,
            HOLD,
            extremes,
            side_cost=bp / 10_000,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    conservative_admission = bool(
        admission(stats)
        and stats["fit"]["strict_mdd_pct"] <= 15.0
        and stats["select_2023"]["strict_mdd_pct"] <= 15.0
        and cost_stress["10"]["fit"]["return_pct"] > 0.0
        and cost_stress["10"]["select_2023"]["return_pct"] > 0.0
    )
    output = {
        "protocol": {
            "source_cutoff": "strictly before 2024-01-01",
            "mechanism": "persistent Preisach price relays accumulate per-domain taker-flow scars; a coherent multi-relay avalanche follows release of previously opposing flow",
            "policy_count": 1,
            "field_window": FIELD_WINDOW,
            "relay_count": len(ALPHAS),
            "relay_pairs": [
                {"alpha": float(alpha), "beta": float(beta)}
                for alpha, beta in zip(ALPHAS, BETAS, strict=True)
            ],
            "min_avalanche": MIN_AVALANCHE,
            "min_coherence": MIN_COHERENCE,
            "signal_quantile": SIGNAL_QUANTILE,
            "hold": HOLD,
            "entry": "completed 5m close signal enters next 5m open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "all pre-2024 rows are exploratory; 2023 is inspected internal selection and 2024+ remained sealed",
            "support_only_design_probe": "before outcome evaluation, 7d/q90 produced only 4 fit signals; 12h/q80 was frozen to meet statistical-support requirements without reading returns",
        },
        "input_summary": {
            "rows": len(market),
            "start": str(dates.min()),
            "end": str(dates.max()),
            "finite_price_field": int(np.isfinite(inputs["price_field"]).sum()),
            "finite_signed_flow": int(np.isfinite(inputs["signed_flow"]).sum()),
        },
        "state_summary": {
            "qualifying_avalanches": int(qualifying_events(state).sum()),
            "positive_release_events": int(
                (qualifying_events(state) & (state["release_score"].to_numpy(float) > 0.0)).sum()
            ),
            "raw_primary_signals": int((long_active | short_active).sum()),
            "raw_primary_long_short": [int(long_active.sum()), int(short_active.sum())],
        },
        "score_threshold": threshold,
        "stats": stats,
        "controls": controls,
        "control_thresholds": control_thresholds,
        "cost_stress": cost_stress,
        "final_admitted": conservative_admission,
    }
    Path("results/preisach_flow_scar_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
