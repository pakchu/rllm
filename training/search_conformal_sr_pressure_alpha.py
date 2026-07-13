"""Search causal conformal surprise in leveraged order-flow pressure.

At each completed hour, aggressive taker flow and positive open-interest build
form a leverage-pressure estimate.  The difference between that estimate and
actual price response is ranked against only the previous 180 days.  Two
one-sided power-betting Shiryaev-Roberts statistics accumulate repeated
exchangeability departures.  A fixed evidence crossing emits either a fade or
stored-pressure-release policy at the next open.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right, insort
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_dual_intrinsic_clock_alpha import SEGMENTS, WINDOWS, load_pre2024
from training.search_positioning_disagreement_alpha import _future_extreme, _simulate_no_stop

HOUR_BARS = 12
DECISION_MINUTE = 55
NORMALIZATION_HOURS = 30 * 24
NORMALIZATION_MIN_HOURS = 15 * 24
REFERENCE_HOURS = 180 * 24
REFERENCE_MIN_HOURS = 90 * 24
POWER = 0.5
SR_BOUNDARY = 200.0
HOLD_BARS = 12 * 12
MAPPINGS = ("fade", "release")
SIDE_COST = 0.0006


def prior_zscore(
    values: np.ndarray | pd.Series,
    *,
    window: int = NORMALIZATION_HOURS,
    min_periods: int = NORMALIZATION_MIN_HOURS,
) -> np.ndarray:
    series = pd.Series(values, dtype=float).reset_index(drop=True)
    prior = series.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).to_numpy(float)


def rolling_conformal_pvalues(
    residual: np.ndarray,
    *,
    reference_window: int = REFERENCE_HOURS,
    min_history: int = REFERENCE_MIN_HOURS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative one-sided ranks against prior positions only."""
    values = np.asarray(residual, dtype=float)
    upper = np.full(len(values), np.nan, dtype=float)
    lower = np.full(len(values), np.nan, dtype=float)
    history_size = np.zeros(len(values), dtype=np.int32)
    ordered: list[float] = []
    queue: deque[tuple[int, float]] = deque()
    for position, value in enumerate(values):
        while queue and queue[0][0] < position - reference_window:
            _, expired = queue.popleft()
            ordered.pop(bisect_left(ordered, expired))
        history_size[position] = len(ordered)
        if np.isfinite(value):
            if len(ordered) >= min_history:
                denominator = len(ordered) + 1
                upper[position] = (
                    1 + len(ordered) - bisect_left(ordered, float(value))
                ) / denominator
                lower[position] = (
                    1 + bisect_right(ordered, float(value))
                ) / denominator
            insort(ordered, float(value))
            queue.append((position, float(value)))
    return upper, lower, history_size


def shiryaev_roberts_events(
    residual: np.ndarray,
    upper_p: np.ndarray,
    lower_p: np.ndarray,
    *,
    boundary: float = SR_BOUNDARY,
    power: float = POWER,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Emit signed evidence events and reset both restart statistics."""
    residual = np.asarray(residual, dtype=float)
    upper_p = np.asarray(upper_p, dtype=float)
    lower_p = np.asarray(lower_p, dtype=float)
    if not (residual.shape == upper_p.shape == lower_p.shape):
        raise ValueError("residual and p-value arrays must align")
    if boundary <= 1.0 or not 0.0 < power < 1.0:
        raise ValueError("boundary must exceed one and power must lie in (0,1)")
    side = np.zeros(len(residual), dtype=np.int8)
    log_up = np.full(len(residual), np.nan, dtype=float)
    log_down = np.full(len(residual), np.nan, dtype=float)
    running_up = running_down = -np.inf
    log_boundary = np.log(boundary)
    log_power = np.log(power)
    for position in range(len(residual)):
        if not (
            np.isfinite(residual[position])
            and np.isfinite(upper_p[position])
            and np.isfinite(lower_p[position])
        ):
            continue
        running_up = (
            np.logaddexp(0.0, running_up)
            + log_power
            + (power - 1.0) * np.log(upper_p[position])
        )
        running_down = (
            np.logaddexp(0.0, running_down)
            + log_power
            + (power - 1.0) * np.log(lower_p[position])
        )
        log_up[position] = running_up
        log_down[position] = running_down
        if (
            running_up >= log_boundary
            and running_up > running_down
            and residual[position] > 0.0
        ):
            side[position] = 1
        elif (
            running_down >= log_boundary
            and running_down > running_up
            and residual[position] < 0.0
        ):
            side[position] = -1
        if side[position]:
            running_up = running_down = -np.inf
    return side, {"log_sr_up": log_up, "log_sr_down": log_down}


def hourly_pressure_components(
    market: pd.DataFrame,
    dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return completed-hour flow, price return and one-bar-delayed OI change."""
    decision_positions = np.flatnonzero(dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").clip(lower=0.0)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    signed_flow = 2.0 * taker_buy - quote
    hourly_flow = (
        signed_flow.rolling(HOUR_BARS, min_periods=HOUR_BARS).sum()
        / quote.rolling(HOUR_BARS, min_periods=HOUR_BARS).sum().replace(0.0, np.nan)
    ).iloc[decision_positions]
    hourly_close = np.log(
        pd.to_numeric(market["close"], errors="coerce").where(lambda value: value > 0.0)
    ).iloc[decision_positions]
    # OI snapshots can be timestamped at the source boundary.  Delay one full
    # 5-minute row before hourly sampling so the signal never depends on the
    # snapshot sharing the completed decision row's source timestamp.
    delayed_oi = pd.to_numeric(market["open_interest"], errors="coerce").shift(1)
    hourly_oi = np.log(delayed_oi.where(lambda value: value > 0.0)).iloc[decision_positions]
    return (
        decision_positions,
        hourly_flow.to_numpy(float),
        hourly_close.reset_index(drop=True).diff().to_numpy(float),
        hourly_oi.reset_index(drop=True).diff().to_numpy(float),
    )


def build_pressure_state(
    market: pd.DataFrame,
    dates: pd.Series,
    *,
    include_oi: bool = True,
    reverse_flow: bool = False,
) -> pd.DataFrame:
    decision_positions, hourly_flow, hourly_return, hourly_oi_change = hourly_pressure_components(
        market, dates
    )
    flow_z = prior_zscore(hourly_flow)
    if reverse_flow:
        flow_z = -flow_z
    price_z = prior_zscore(hourly_return)
    oi_z = prior_zscore(hourly_oi_change)
    oi_multiplier = 1.0 + np.clip(oi_z, 0.0, 3.0) if include_oi else np.ones(len(flow_z))
    residual = flow_z * oi_multiplier - price_z
    if include_oi:
        residual[~np.isfinite(oi_z)] = np.nan
    upper_p, lower_p, history_size = rolling_conformal_pvalues(residual)
    event_side, evidence = shiryaev_roberts_events(residual, upper_p, lower_p)

    state = pd.DataFrame(
        {
            "decision": np.zeros(len(market), dtype=bool),
            "flow_z": np.full(len(market), np.nan),
            "price_z": np.full(len(market), np.nan),
            "oi_z": np.full(len(market), np.nan),
            "pressure_residual": np.full(len(market), np.nan),
            "upper_p": np.full(len(market), np.nan),
            "lower_p": np.full(len(market), np.nan),
            "history_size": np.zeros(len(market), dtype=np.int32),
            "log_sr_up": np.full(len(market), np.nan),
            "log_sr_down": np.full(len(market), np.nan),
            "pressure_side": np.zeros(len(market), dtype=np.int8),
        }
    )
    state.loc[decision_positions, "decision"] = True
    state.loc[decision_positions, "flow_z"] = flow_z
    state.loc[decision_positions, "price_z"] = price_z
    state.loc[decision_positions, "oi_z"] = oi_z
    state.loc[decision_positions, "pressure_residual"] = residual
    state.loc[decision_positions, "upper_p"] = upper_p
    state.loc[decision_positions, "lower_p"] = lower_p
    state.loc[decision_positions, "history_size"] = history_size
    state.loc[decision_positions, "log_sr_up"] = evidence["log_sr_up"]
    state.loc[decision_positions, "log_sr_down"] = evidence["log_sr_down"]
    state.loc[decision_positions, "pressure_side"] = event_side
    return state


def policy_masks(state: pd.DataFrame, mapping: str) -> tuple[np.ndarray, np.ndarray]:
    pressure_side = state["pressure_side"].to_numpy(np.int8)
    if mapping == "fade":
        trade_side = -pressure_side
    elif mapping == "release":
        trade_side = pressure_side
    else:
        raise KeyError(mapping)
    active = state["decision"].to_numpy(bool) & (pressure_side != 0)
    return active & (trade_side > 0), active & (trade_side < 0)


def single_tail_masks(state: pd.DataFrame, mapping: str) -> tuple[np.ndarray, np.ndarray]:
    residual = state["pressure_residual"].to_numpy(float)
    upper_p = state["upper_p"].to_numpy(float)
    lower_p = state["lower_p"].to_numpy(float)
    pressure_side = np.where((upper_p <= 0.01) & (residual > 0.0), 1, 0)
    pressure_side = np.where((lower_p <= 0.01) & (residual < 0.0), -1, pressure_side)
    trade_side = -pressure_side if mapping == "fade" else pressure_side
    return trade_side > 0, trade_side < 0


def lag_boolean(values: np.ndarray, bars: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if bars <= 0:
        raise ValueError("lag must be positive")
    return np.r_[np.zeros(bars, dtype=bool), values[:-bars]]


def support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
    hold_bars: int = HOLD_BARS,
) -> dict[str, int]:
    start, end = WINDOWS[window]
    period = ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(bool)
    active = np.asarray(long_active, dtype=bool) | np.asarray(short_active, dtype=bool)
    raw = int((period & active).sum())
    candidates = np.arange(0, len(dates) - hold_bars - 2, dtype=np.int64)
    candidates = candidates[period[candidates] & active[candidates]]
    executable = 0
    next_position = 0
    for position in candidates:
        if position < next_position:
            continue
        entry_position = position + 1
        exit_position = entry_position + hold_bars
        if exit_position >= len(dates) or not period[exit_position]:
            continue
        executable += 1
        next_position = exit_position + 1
    return {"raw": raw, "strict_executable": executable}


def event_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    return float((left & right).sum() / union.sum()) if union.any() else 0.0


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    return float(pd.Series(left[finite]).corr(pd.Series(right[finite]), method="spearman"))


def novelty_overlap_audit(
    market: pd.DataFrame,
    dates: pd.Series,
    state: pd.DataFrame,
    no_oi_state: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the new residual/events with fixed prior/simple baselines."""
    from training.search_nonequilibrium_probability_current_alpha import (
        build_hourly_microstates,
        build_transition_features,
        fit_threshold,
        policy_masks as current_policy_masks,
    )
    from training.search_online_rls_price_impact_alpha import build_features as build_rls_features

    decision = state["decision"].to_numpy(bool)
    primary_residual = state["pressure_residual"].to_numpy(float)
    primary_events = state["pressure_side"].to_numpy(np.int8) != 0
    no_oi_residual = no_oi_state["pressure_residual"].to_numpy(float)
    no_oi_events = no_oi_state["pressure_side"].to_numpy(np.int8) != 0

    rls = build_rls_features(market)
    rls_residual = rls["rls_residual_z_2016"].to_numpy(float)
    rls_tail_events = decision & np.isfinite(rls_residual) & (np.abs(rls_residual) >= 1.5)

    hourly = build_hourly_microstates(market, dates)
    current = build_transition_features(hourly)
    current_threshold = fit_threshold(current, "current_score")
    current_long, current_short = current_policy_masks(
        current, len(market), current_threshold
    )
    current_events = current_long | current_short
    current_score = np.full(len(market), np.nan, dtype=float)
    current_score[current["position"].to_numpy(np.int64)] = current["current_score"].to_numpy(float)

    simple_tail_long, simple_tail_short = single_tail_masks(state, "release")
    simple_tail_events = simple_tail_long | simple_tail_short
    oi_z = state["oi_z"].to_numpy(float)
    flow_z = state["flow_z"].to_numpy(float)
    inventory_proxy = flow_z * np.clip(oi_z, 0.0, 3.0)
    feature_spearman = {
        "online_rls_residual_z2016": finite_spearman(primary_residual, rls_residual),
        "nonequilibrium_current_score": finite_spearman(primary_residual, current_score),
        "flow_price_residual_without_oi": finite_spearman(primary_residual, no_oi_residual),
        "simple_oi_build_pressure": finite_spearman(primary_residual, inventory_proxy),
    }
    event_overlap = {
        "online_rls_abs_residual_ge_1p5": event_jaccard(primary_events, rls_tail_events),
        "nonequilibrium_current_q80": event_jaccard(primary_events, current_events),
        "single_conformal_tail_p01": event_jaccard(primary_events, simple_tail_events),
        "same_sr_without_oi": event_jaccard(primary_events, no_oi_events),
    }
    return {
        "feature_spearman": feature_spearman,
        "event_jaccard": event_overlap,
        "max_event_jaccard": max(event_overlap.values()),
        "novelty_pass": bool(max(event_overlap.values()) < 0.50),
        "gate": "all fixed baseline event Jaccards must be below 0.50",
        "baseline_note": "inventory-conservation exact feature requires a different funding/premium cache; simple OI-build pressure is reported here while exact family overlap is disclosed qualitatively",
    }


def simulate(
    market: pd.DataFrame,
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    extremes: tuple[np.ndarray, np.ndarray],
    *,
    side_cost: float = SIDE_COST,
) -> dict[str, dict[str, Any]]:
    return {
        window: _simulate_no_stop(
            market,
            dates,
            long_active,
            short_active,
            window=window,
            hold_bars=HOLD_BARS,
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
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 10
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )
    return bool(
        enough
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and stats["select_2023_h1"]["return_pct"] > 0.0
        and stats["select_2023_h2"]["return_pct"] > 0.0
    )


def rank_key(stats: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    enough = stats["fit"]["trades"] >= 80 and stats["select_2023"]["trades"] >= 24
    core = [
        stats[name]["ratio"]
        for name in ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    ]
    return (
        admission(stats),
        enough,
        min(core) > 0.0,
        sum(stats[name]["return_pct"] > 0.0 for name in SEGMENTS),
        min(core),
        float(np.median(core)),
        stats["select_2023"]["trades"],
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
    state = build_pressure_state(market, dates)
    no_oi_state = build_pressure_state(market, dates, include_oi=False)
    reversed_state = build_pressure_state(market, dates, reverse_flow=True)
    novelty_audit = novelty_overlap_audit(market, dates, state, no_oi_state)
    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    rows: list[dict[str, Any]] = []
    signal_bank: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    support_preflight: dict[str, dict[str, dict[str, int]]] = {}
    for mapping in MAPPINGS:
        long_active, short_active = policy_masks(state, mapping)
        signal_bank[mapping] = (long_active, short_active)
        support_preflight[mapping] = {
            window: support_counts(
                dates, long_active, short_active, window=window, hold_bars=HOLD_BARS
            )
            for window in ("fit", "select_2023")
        }
        stats = simulate(market, dates, long_active, short_active, extremes)
        rows.append(
            {
                "mapping": mapping,
                "hold": HOLD_BARS,
                "raw_events": int((long_active | short_active).sum()),
                "raw_long_short": [int(long_active.sum()), int(short_active.sum())],
                "prelim_admitted": admission(stats),
                "rank": rank_key(stats),
                "stats": stats,
            }
        )
    rows.sort(key=lambda row: row["rank"], reverse=True)
    for index, row in enumerate(rows, 1):
        print_stats(f"RANK {index} {row['mapping']} rank={row['rank']}", row["stats"])

    top = rows[0]
    base_long, base_short = signal_bank[top["mapping"]]
    controls: dict[str, dict[str, dict[str, Any]]] = {}
    no_oi_long, no_oi_short = policy_masks(no_oi_state, top["mapping"])
    controls["remove_oi_multiplier"] = simulate(
        market, dates, no_oi_long, no_oi_short, extremes
    )
    tail_long, tail_short = single_tail_masks(state, top["mapping"])
    controls["single_tail_no_sequential_evidence"] = simulate(
        market, dates, tail_long, tail_short, extremes
    )
    reversed_long, reversed_short = policy_masks(reversed_state, top["mapping"])
    controls["reverse_flow_sign"] = simulate(
        market, dates, reversed_long, reversed_short, extremes
    )
    for name, bars in (("signal_delay_1h", 12), ("signal_delay_24h", 288), ("signal_delay_7d", 2016)):
        controls[name] = simulate(
            market,
            dates,
            lag_boolean(base_long, bars),
            lag_boolean(base_short, bars),
            extremes,
        )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            base_long,
            base_short,
            extremes,
            side_cost=bp / 10_000.0,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    residual = state["pressure_residual"].to_numpy(float)
    pressure_side = state["pressure_side"].to_numpy(np.int8)
    output = {
        "protocol": {
            "source_cutoff": "returned analysis frame hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "shared chunk parser may read and immediately discard later rows in the cutoff-crossing chunk; none enters the returned frame or computation",
            "mechanism": "30d prior-standardized flow/OI-build pressure minus price response; 180d prior-only conformal ranks; two one-sided power-betting Shiryaev-Roberts restart statistics",
            "normalization_hours": NORMALIZATION_HOURS,
            "reference_hours": REFERENCE_HOURS,
            "reference_min_hours": REFERENCE_MIN_HOURS,
            "power": POWER,
            "sr_boundary": SR_BOUNDARY,
            "two_tail_multiplicity": "boundary raised from the nominal 100 change-evidence level to 200 before outcomes as a conservative two-tail union adjustment; no formal dependent-data false-alarm guarantee is claimed",
            "grid_size": len(rows),
            "grid": "two co-primary economic mappings (fade/release), one fixed 12h hold",
            "selection_rule": "both exact-opposite mappings are always reported; deterministic rank_key over admission, support, temporal positivity, minimum/median core ratios and 2023 trades is frozen before outcomes; no post-result expansion",
            "support_only_preflight": {"performed_before_returns": True, "counts": support_preflight},
            "entry": "completed minute-55 state enters next minute-00 open",
            "oi_availability": "open_interest delayed one complete 5m source row before completed-hour sampling",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "pre-2024 exploratory mechanism; 2023 is inspected internal selection and 2024+ remained excluded from computation",
            "statistical_note": "finite-window conformal ranks plus Shiryaev-Roberts restart evidence are used as a sequential change detector, not claimed as an anytime-valid e-process under dependent market data",
        },
        "state_summary": {
            "valid_residual_hours": int(np.isfinite(residual).sum()),
            "raw_evidence_events": int(np.count_nonzero(pressure_side)),
            "positive_negative_events": [
                int(np.count_nonzero(pressure_side > 0)),
                int(np.count_nonzero(pressure_side < 0)),
            ],
        },
        "rows": [{**row, "rank": list(row["rank"])} for row in rows],
        "novelty_overlap_audit": novelty_audit,
        "controls": controls,
        "cost_stress": cost_stress,
        "final_admitted": bool(
            top["prelim_admitted"]
            and novelty_audit["novelty_pass"]
            and not any(admission(stats) for stats in controls.values())
        ),
    }
    Path("results/conformal_sr_pressure_alpha_scan_2026-07-14.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
