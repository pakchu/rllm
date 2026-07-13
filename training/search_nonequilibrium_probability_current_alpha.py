"""Search an unsupervised nonequilibrium probability-current alpha.

Hourly price, aggressive-flow and open-interest signs define eight observable
microstates.  A causal rolling transition graph is decomposed into its
time-symmetric traffic and antisymmetric probability current.  The policy uses
only the latter: from the current state, it follows the price sign of states
reached by positive irreversible current.  No realized trade outcome is used
to estimate the state graph or signal direction.
"""
from __future__ import annotations

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

DECISION_MINUTE = 55
HOUR_BARS = 12
TRANSITION_WINDOW = 720
MIN_TRANSITIONS = 360
DIRICHLET_PRIOR = 0.5
SIGNAL_QUANTILE = 0.80
MIN_DIRECTIONALITY = 0.50
HOLDS = (72, 144)
DELAY_7D = 2016


def build_hourly_microstates(
    market: pd.DataFrame,
    dates: pd.Series,
    *,
    use_flow: bool = True,
    use_oi: bool = True,
) -> pd.DataFrame:
    positions = np.flatnonzero(dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool))
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0.0))
    price_change = log_price.diff(HOUR_BARS)

    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    signed_quote = 2.0 * taker_buy - quote
    flow_change = signed_quote.rolling(HOUR_BARS, min_periods=HOUR_BARS).sum()

    oi = pd.to_numeric(market["open_interest"], errors="coerce").where(lambda value: value > 0.0)
    oi_change = np.log(oi).diff(HOUR_BARS)
    frame = pd.DataFrame(
        {
            "position": positions,
            "date": dates.iloc[positions].to_numpy(),
            "price_change": price_change.iloc[positions].to_numpy(float),
            "flow_change": flow_change.iloc[positions].to_numpy(float),
            "oi_change": oi_change.iloc[positions].to_numpy(float),
        }
    )
    required = np.isfinite(frame["price_change"].to_numpy(float))
    if use_flow:
        required &= np.isfinite(frame["flow_change"].to_numpy(float))
    if use_oi:
        required &= np.isfinite(frame["oi_change"].to_numpy(float))
    price_bit = (frame["price_change"].to_numpy(float) >= 0.0).astype(np.int8)
    flow_bit = (frame["flow_change"].to_numpy(float) >= 0.0).astype(np.int8) if use_flow else np.zeros(len(frame), dtype=np.int8)
    oi_bit = (frame["oi_change"].to_numpy(float) >= 0.0).astype(np.int8) if use_oi else np.zeros(len(frame), dtype=np.int8)
    if use_flow and use_oi:
        state = price_bit * 4 + flow_bit * 2 + oi_bit
        state_count = 8
    elif use_flow or use_oi:
        auxiliary = flow_bit if use_flow else oi_bit
        state = price_bit * 2 + auxiliary
        state_count = 4
    else:
        state = price_bit
        state_count = 2
    frame["state"] = np.where(required, state, -1).astype(np.int16)
    frame["price_side"] = np.where(required, 2 * price_bit - 1, 0).astype(np.int8)
    frame.attrs["state_count"] = state_count
    frame.attrs["price_signs"] = np.where(
        (np.arange(state_count) // (state_count // 2)) > 0, 1.0, -1.0
    )
    return frame


def transition_counts_before(
    states: np.ndarray,
    position: int,
    *,
    state_count: int,
    window: int,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.int16)
    counts = np.zeros((state_count, state_count), dtype=np.int64)
    first_end = max(1, int(position) - int(window))
    for end in range(first_end, int(position)):
        source = int(values[end - 1])
        target = int(values[end])
        if 0 <= source < state_count and 0 <= target < state_count:
            counts[source, target] += 1
    return counts


def _current_projection(
    joint_flux: np.ndarray,
    state: int,
    price_signs: np.ndarray,
    *,
    reverse: bool = False,
) -> tuple[float, float, float]:
    flux = joint_flux.T if reverse else joint_flux
    current = flux - flux.T
    outgoing = np.maximum(current[int(state)], 0.0)
    strength = float(outgoing.sum())
    if strength <= 0.0:
        return 0.0, 0.0, 0.0
    velocity = float(np.dot(outgoing, price_signs) / strength)
    reverse_flux = flux[:, int(state)]
    entropy = float(
        np.sum(
            outgoing
            * np.log(
                np.maximum(flux[int(state)], 1e-15)
                / np.maximum(reverse_flux, 1e-15)
            )
        )
    )
    return velocity, strength, max(entropy, 0.0)


def build_transition_features(
    hourly: pd.DataFrame,
    *,
    window: int = TRANSITION_WINDOW,
) -> pd.DataFrame:
    states = hourly["state"].to_numpy(np.int16)
    state_count = int(hourly.attrs["state_count"])
    price_signs = np.asarray(hourly.attrs["price_signs"], dtype=float)
    counts = np.zeros((state_count, state_count), dtype=np.int64)
    valid_transitions = 0
    fields = {
        name: np.full(len(hourly), np.nan, dtype=float)
        for name in (
            "current_direction",
            "current_score",
            "current_strength",
            "current_velocity",
            "entropy_production",
            "reverse_direction",
            "reverse_score",
            "markov_direction",
            "markov_score",
            "transition_count",
        )
    }
    for position in range(len(hourly)):
        add_end = position - 1
        if add_end >= 1:
            source = int(states[add_end - 1])
            target = int(states[add_end])
            if 0 <= source < state_count and 0 <= target < state_count:
                counts[source, target] += 1
                valid_transitions += 1
        remove_end = position - window - 1
        if remove_end >= 1:
            source = int(states[remove_end - 1])
            target = int(states[remove_end])
            if 0 <= source < state_count and 0 <= target < state_count:
                counts[source, target] -= 1
                valid_transitions -= 1

        current_state = int(states[position])
        fields["transition_count"][position] = valid_transitions
        if current_state < 0 or valid_transitions < MIN_TRANSITIONS:
            continue
        smoothed = counts.astype(float) + DIRICHLET_PRIOR
        joint = smoothed / smoothed.sum()
        velocity, strength, entropy = _current_projection(
            joint, current_state, price_signs
        )
        reverse_velocity, reverse_strength, reverse_entropy = _current_projection(
            joint, current_state, price_signs, reverse=True
        )
        row = smoothed[current_state] / smoothed[current_state].sum()
        markov_expectation = float(np.dot(row, price_signs))

        fields["current_direction"][position] = np.sign(velocity)
        fields["current_score"][position] = entropy * abs(velocity)
        fields["current_strength"][position] = strength
        fields["current_velocity"][position] = velocity
        fields["entropy_production"][position] = entropy
        fields["reverse_direction"][position] = np.sign(reverse_velocity)
        fields["reverse_score"][position] = reverse_entropy * abs(reverse_velocity)
        fields["markov_direction"][position] = np.sign(markov_expectation)
        fields["markov_score"][position] = abs(markov_expectation)
    return pd.concat(
        [hourly.reset_index(drop=True), pd.DataFrame(fields)], axis=1
    )


def fit_threshold(
    features: pd.DataFrame,
    column: str,
    *,
    quantile: float = SIGNAL_QUANTILE,
) -> float:
    dates = pd.to_datetime(features["date"])
    start, end = WINDOWS["fit"]
    fit = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    values = features[column].to_numpy(float)
    selected = values[fit & np.isfinite(values) & (values > 0.0)]
    if not len(selected):
        raise ValueError(f"no positive fit scores for {column}")
    return float(np.quantile(selected, quantile))


def policy_masks(
    features: pd.DataFrame,
    rows: int,
    threshold: float,
    *,
    score_column: str = "current_score",
    direction_column: str = "current_direction",
    flip: bool = False,
    use_price_side: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    score = features[score_column].to_numpy(float)
    velocity = features["current_velocity"].to_numpy(float)
    direction = (
        features["price_side"].to_numpy(np.int8)
        if use_price_side
        else np.nan_to_num(features[direction_column].to_numpy(float)).astype(np.int8)
    )
    if flip:
        direction = -direction
    active = (
        np.isfinite(score)
        & (score >= threshold)
        & (direction != 0)
        & (np.abs(velocity) >= MIN_DIRECTIONALITY if score_column == "current_score" else True)
    )
    long_active = np.zeros(rows, dtype=bool)
    short_active = np.zeros(rows, dtype=bool)
    positions = features["position"].to_numpy(np.int64)
    long_active[positions[active & (direction > 0)]] = True
    short_active[positions[active & (direction < 0)]] = True
    return long_active, short_active


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


def main() -> None:
    market, dates = load_pre2024()
    hourly = build_hourly_microstates(market, dates)
    features = build_transition_features(hourly)
    threshold = fit_threshold(features, "current_score")
    long_active, short_active = policy_masks(features, len(market), threshold)
    extremes = {
        hold: (
            _future_extreme(market["low"].to_numpy(float), hold, "min"),
            _future_extreme(market["high"].to_numpy(float), hold, "max"),
        )
        for hold in HOLDS
    }
    rows: list[dict[str, Any]] = []
    for hold in HOLDS:
        stats = simulate(market, dates, long_active, short_active, hold, extremes[hold])
        rows.append(
            {
                "hold": hold,
                "score_threshold": threshold,
                "raw_signals": int((long_active | short_active).sum()),
                "rank": rank_key(stats),
                "prelim_admitted": admission(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    for index, row in enumerate(rows, 1):
        print_stats(f"RANK {index} hold{row['hold']} rank={row['rank']}", row["stats"])

    top = rows[0]
    flip_long, flip_short = policy_masks(features, len(market), threshold, flip=True)
    price_long, price_short = policy_masks(
        features, len(market), threshold, use_price_side=True
    )
    reverse_threshold = fit_threshold(features, "reverse_score")
    reverse_long, reverse_short = policy_masks(
        features,
        len(market),
        reverse_threshold,
        score_column="reverse_score",
        direction_column="reverse_direction",
    )
    markov_threshold = fit_threshold(features, "markov_score")
    markov_long, markov_short = policy_masks(
        features,
        len(market),
        markov_threshold,
        score_column="markov_score",
        direction_column="markov_direction",
    )
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market, dates, flip_long, flip_short, top["hold"], extremes[top["hold"]]
        ),
        "same_events_current_price_side": simulate(
            market, dates, price_long, price_short, top["hold"], extremes[top["hold"]]
        ),
        "time_reversed_probability_current": simulate(
            market, dates, reverse_long, reverse_short, top["hold"], extremes[top["hold"]]
        ),
        "ordinary_markov_expectation": simulate(
            market, dates, markov_long, markov_short, top["hold"], extremes[top["hold"]]
        ),
        "signal_delay_1h": simulate(
            market,
            dates,
            lag_boolean(long_active, HOUR_BARS),
            lag_boolean(short_active, HOUR_BARS),
            top["hold"],
            extremes[top["hold"]],
        ),
        "signal_delay_7d": simulate(
            market,
            dates,
            lag_boolean(long_active, DELAY_7D),
            lag_boolean(short_active, DELAY_7D),
            top["hold"],
            extremes[top["hold"]],
        ),
    }
    ablated_hourly = build_hourly_microstates(market, dates, use_oi=False)
    ablated_features = build_transition_features(ablated_hourly)
    ablated_threshold = fit_threshold(ablated_features, "current_score")
    ablated_long, ablated_short = policy_masks(
        ablated_features, len(market), ablated_threshold
    )
    controls["remove_open_interest_state"] = simulate(
        market, dates, ablated_long, ablated_short, top["hold"], extremes[top["hold"]]
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
            "mechanism": "hourly price/flow/OI sign microstates; prior 30d transition joint flux is decomposed into antisymmetric probability current; current outgoing cycle predicts next-state price sign",
            "state_count": 8,
            "transition_window_hours": TRANSITION_WINDOW,
            "minimum_transitions": MIN_TRANSITIONS,
            "dirichlet_prior": DIRICHLET_PRIOR,
            "signal_quantile": SIGNAL_QUANTILE,
            "minimum_directionality": MIN_DIRECTIONALITY,
            "grid_size": len(rows),
            "grid": "one frozen probability-current signal x 6h/12h holds",
            "entry": "completed minute-55 state enters next minute-00 open",
            "leverage": 0.5,
            "cost": "6bp/side",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "all pre-2024 rows are exploratory; 2023 is inspected internal selection and 2024+ remained sealed",
        },
        "state_summary": {
            "hourly_rows": len(hourly),
            "valid_microstates": int((hourly["state"] >= 0).sum()),
            "finite_current_scores": int(np.isfinite(features["current_score"]).sum()),
            "positive_current_scores": int((features["current_score"] > 0.0).sum()),
            "raw_signals": int((long_active | short_active).sum()),
            "raw_long_short": [int(long_active.sum()), int(short_active.sum())],
        },
        "score_threshold": threshold,
        "rows": rows,
        "controls": controls,
        "control_thresholds": {
            "time_reversed_probability_current": reverse_threshold,
            "ordinary_markov_expectation": markov_threshold,
            "remove_open_interest_state": ablated_threshold,
        },
        "cost_stress": cost_stress,
        "final_admitted": bool(top["prelim_admitted"]),
    }
    Path("results/nonequilibrium_probability_current_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
