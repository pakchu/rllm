"""Search a causal positioning-simplex curvature exhaustion alpha.

Three delayed Binance USD-M positioning ratios represent top-trader positions,
global accounts and taker flow.  Six-hour cohort-migration vectors are compared
with their preceding six-hour vectors.  Their normalized cross product measures
how strongly positioning bends away from one-dimensional consensus.  The fixed
primary event fades the net cohort migration when that bending occurs together
with expanding open interest in the fit-frozen upper energy quintile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_conformal_sr_pressure_alpha import (
    event_jaccard,
    finite_spearman,
    lag_boolean,
)
from training.search_positioning_disagreement_alpha import (
    _attach_delayed_metrics,
    _future_extreme,
    _simulate_no_stop,
)
from training.search_positioning_hgb_path_alpha import _read_before

MARKET = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
METRICS = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
CUTOFF = "2024-01-01"
RESULT_PATH = Path("results/positioning_simplex_curvature_alpha_scan_2026-07-14.json")
WINDOWS = {
    "fit": ("2020-10-15", "2022-01-01"),
    "fit_2020_q4": ("2020-10-15", "2021-01-01"),
    "fit_2021_h1": ("2021-01-01", "2021-07-01"),
    "fit_2021_h2": ("2021-07-01", "2022-01-01"),
    "quarantine_2022": ("2022-01-01", "2023-01-01"),
    "select_2023": ("2023-01-01", "2024-01-01"),
    "select_2023_h1": ("2023-01-01", "2023-07-01"),
    "select_2023_h2": ("2023-07-01", "2024-01-01"),
}
ROBUSTNESS_SEGMENTS = (
    "fit_2020_q4",
    "fit_2021_h1",
    "fit_2021_h2",
    "select_2023_h1",
    "select_2023_h2",
)
DECISION_MINUTE = 55
MIGRATION_HOURS = 6
HOLD_BARS = 12 * 12
TAIL_QUANTILE = 0.80
SIDE_COST = 0.0006
SCORE_VARIANTS = (
    "oi_curvature_speed",
    "oi_speed",
    "curvature_speed",
    "speed_only",
    "oi_pairwise_dispersion",
)
RATIO_COLUMNS = (
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def load_pre2024() -> tuple[pd.DataFrame, pd.Series]:
    market = _read_before(MARKET, "date", CUTOFF)
    metrics = _read_before(METRICS, "create_time", CUTOFF)
    market = _attach_delayed_metrics(
        market, metrics, tolerance="10min", delay_bars=1
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(CUTOFF):
        raise RuntimeError("future market rows opened")
    source_time = pd.to_datetime(
        market["positioning_source_time"], errors="coerce"
    )
    valid_source = source_time.notna()
    if (
        valid_source.any()
        and (source_time[valid_source] > dates[valid_source] - pd.Timedelta("5min")).any()
    ):
        raise RuntimeError("positioning source was not delayed one complete row")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("simplex search requires a complete 5-minute market grid")
    return market, dates


def simplex_curvature(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return signed normalized bending around the cohort-consensus axis."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 3:
        raise ValueError("curvature inputs must have matching (n, 3) shapes")
    cross = np.cross(previous, current)
    denominator = np.linalg.norm(previous, axis=1) * np.linalg.norm(current, axis=1)
    axis = np.ones(3, dtype=float) / np.sqrt(3.0)
    signed = np.einsum("ij,j->i", cross, axis)
    return np.divide(
        signed,
        denominator,
        out=np.full(len(previous), np.nan),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )


def build_simplex_state(market: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    decision_positions = np.flatnonzero(
        dates.dt.minute.eq(DECISION_MINUTE).to_numpy(bool)
    )
    hourly_ratios = np.column_stack(
        [
            np.log(
                pd.to_numeric(market[column], errors="coerce").where(
                    lambda value: value > 0.0
                )
            )
            .iloc[decision_positions]
            .to_numpy(float)
            for column in RATIO_COLUMNS
        ]
    )
    current_velocity = np.full_like(hourly_ratios, np.nan)
    previous_velocity = np.full_like(hourly_ratios, np.nan)
    h = MIGRATION_HOURS
    current_velocity[h:] = hourly_ratios[h:] - hourly_ratios[:-h]
    previous_velocity[2 * h :] = (
        hourly_ratios[h:-h] - hourly_ratios[: -2 * h]
    )
    curvature = simplex_curvature(previous_velocity, current_velocity)
    speed = np.linalg.norm(current_velocity, axis=1)
    complete_velocity = np.all(np.isfinite(current_velocity), axis=1)
    pairwise_dispersion = np.full(len(current_velocity), np.nan)
    migration = np.full(len(current_velocity), np.nan)
    pairwise_dispersion[complete_velocity] = np.std(
        current_velocity[complete_velocity], axis=1
    )
    migration[complete_velocity] = np.mean(
        current_velocity[complete_velocity], axis=1
    )

    hourly_oi = np.log(
        pd.to_numeric(market["sum_open_interest"], errors="coerce").where(
            lambda value: value > 0.0
        )
    ).iloc[decision_positions].to_numpy(float)
    oi_change = np.full(len(hourly_oi), np.nan)
    oi_change[h:] = hourly_oi[h:] - hourly_oi[:-h]
    oi_build = np.maximum(oi_change, 0.0)
    score_values = {
        "oi_curvature_speed": oi_build * speed * np.abs(curvature),
        "oi_speed": oi_build * speed,
        "curvature_speed": speed * np.abs(curvature),
        "speed_only": speed,
        "oi_pairwise_dispersion": oi_build * pairwise_dispersion,
    }

    state = pd.DataFrame(
        {
            "decision": np.zeros(len(market), dtype=bool),
            "migration": np.full(len(market), np.nan),
            "curvature": np.full(len(market), np.nan),
            "speed": np.full(len(market), np.nan),
            "pairwise_dispersion": np.full(len(market), np.nan),
            "oi_change": np.full(len(market), np.nan),
        }
    )
    state.loc[decision_positions, "decision"] = True
    for name, values in {
        "migration": migration,
        "curvature": curvature,
        "speed": speed,
        "pairwise_dispersion": pairwise_dispersion,
        "oi_change": oi_change,
        **score_values,
    }.items():
        if name not in state:
            state[name] = np.nan
        state.loc[decision_positions, name] = values
    return state.replace([np.inf, -np.inf], np.nan)


def window_mask(dates: pd.Series, window: str) -> np.ndarray:
    start, end = WINDOWS[window]
    return ((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))).to_numpy(
        bool
    )


def fit_threshold(state: pd.DataFrame, dates: pd.Series, score_name: str) -> float:
    values = state[score_name].to_numpy(float)
    reference = values[window_mask(dates, "fit") & np.isfinite(values)]
    if len(reference) < 1_000:
        raise ValueError(f"insufficient fit support for {score_name}: {len(reference)}")
    return float(np.quantile(reference, TAIL_QUANTILE))


def policy_masks(
    state: pd.DataFrame,
    score_name: str,
    threshold: float,
    *,
    mapping: str = "fade",
) -> tuple[np.ndarray, np.ndarray]:
    score = state[score_name].to_numpy(float)
    migration = state["migration"].to_numpy(float)
    active = (
        state["decision"].to_numpy(bool)
        & np.isfinite(score)
        & np.isfinite(migration)
        & (score >= threshold)
        & (migration != 0.0)
    )
    side = -np.sign(migration) if mapping == "fade" else np.sign(migration)
    if mapping not in {"fade", "continuation"}:
        raise KeyError(mapping)
    return active & (side > 0.0), active & (side < 0.0)


def support_counts(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    window: str,
) -> dict[str, int]:
    period = window_mask(dates, window)
    long_active = np.asarray(long_active, dtype=bool)
    short_active = np.asarray(short_active, dtype=bool)
    active = long_active | short_active
    candidates = np.arange(0, len(dates) - HOLD_BARS - 2, dtype=np.int64)
    candidates = candidates[period[candidates] & active[candidates]]
    executable_long = executable_short = 0
    next_position = 0
    for position in candidates:
        if position < next_position:
            continue
        entry = position + 1
        exit_position = entry + HOLD_BARS
        if exit_position >= len(dates) or not period[exit_position]:
            continue
        executable_long += int(long_active[position])
        executable_short += int(short_active[position])
        next_position = exit_position + 1
    return {
        "raw": int((period & active).sum()),
        "raw_long": int((period & long_active).sum()),
        "raw_short": int((period & short_active).sum()),
        "strict_executable": executable_long + executable_short,
        "strict_executable_long": executable_long,
        "strict_executable_short": executable_short,
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


def admitted(stats: dict[str, dict[str, Any]]) -> bool:
    support = (
        stats["fit"]["trades"] >= 80
        and stats["select_2023"]["trades"] >= 24
        and min(stats["select_2023_h1"]["trades"], stats["select_2023_h2"]["trades"]) >= 8
        and min(stats["fit"]["longs"], stats["fit"]["shorts"]) >= 10
        and min(stats["select_2023"]["longs"], stats["select_2023"]["shorts"]) >= 4
    )
    return bool(
        support
        and stats["fit"]["return_pct"] > 0.0
        and stats["fit"]["ratio"] >= 3.0
        and stats["select_2023"]["return_pct"] > 0.0
        and stats["select_2023"]["ratio"] >= 3.0
        and all(stats[name]["return_pct"] > 0.0 for name in ROBUSTNESS_SEGMENTS)
    )


def print_stats(title: str, stats: dict[str, dict[str, Any]]) -> None:
    print("\n" + title)
    for name in ("fit", "select_2023", *ROBUSTNESS_SEGMENTS, "quarantine_2022"):
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


def run(*, support_only: bool = False) -> dict[str, Any]:
    market, dates = load_pre2024()
    state = build_simplex_state(market, dates)
    thresholds = {
        name: fit_threshold(state, dates, name) for name in SCORE_VARIANTS
    }
    mask_bank = {
        name: policy_masks(state, name, thresholds[name]) for name in SCORE_VARIANTS
    }
    support = {
        name: {
            window: support_counts(dates, *masks, window=window)
            for window in WINDOWS
        }
        for name, masks in mask_bank.items()
    }
    coverage = {
        window: int(
            (
                window_mask(dates, window)
                & np.isfinite(state["oi_curvature_speed"].to_numpy(float))
            ).sum()
        )
        for window in WINDOWS
    }
    preflight = {"thresholds": thresholds, "support": support, "coverage": coverage}
    if support_only:
        return {"support_only": True, **preflight}

    extremes = (
        _future_extreme(market["low"].to_numpy(float), HOLD_BARS, "min"),
        _future_extreme(market["high"].to_numpy(float), HOLD_BARS, "max"),
    )
    primary_long, primary_short = mask_bank["oi_curvature_speed"]
    primary = simulate(market, dates, primary_long, primary_short, extremes)
    print_stats("PRIMARY simplex_curvature_fade", primary)
    controls: dict[str, dict[str, dict[str, Any]]] = {
        "direction_flip": simulate(
            market,
            dates,
            *policy_masks(
                state,
                "oi_curvature_speed",
                thresholds["oi_curvature_speed"],
                mapping="continuation",
            ),
            extremes,
        )
    }
    for name, masks in mask_bank.items():
        if name != "oi_curvature_speed":
            controls[name] = simulate(market, dates, masks[0], masks[1], extremes)
    for name, bars in (
        ("signal_delay_5m", 1),
        ("signal_delay_1h", 12),
        ("signal_delay_24h", 288),
        ("signal_delay_7d", 2016),
    ):
        controls[name] = simulate(
            market,
            dates,
            lag_boolean(primary_long, bars),
            lag_boolean(primary_short, bars),
            extremes,
        )
    for name, stats in controls.items():
        print_stats("CONTROL " + name, stats)

    cost_stress = {
        str(bp): simulate(
            market,
            dates,
            primary_long,
            primary_short,
            extremes,
            side_cost=bp / 10_000.0,
        )
        for bp in (0, 1, 3, 6, 10, 15)
    }
    primary_events = primary_long | primary_short
    event_overlap = {
        name: event_jaccard(primary_events, masks[0] | masks[1])
        for name, masks in mask_bank.items()
        if name != "oi_curvature_speed"
    }
    primary_score = state["oi_curvature_speed"].to_numpy(float)
    feature_overlap = {
        name: finite_spearman(primary_score, state[name].to_numpy(float))
        for name in SCORE_VARIANTS
        if name != "oi_curvature_speed"
    }
    novelty_pass = bool(
        max(event_overlap.values()) < 0.50
        and max(abs(value) for value in feature_overlap.values()) < 0.80
    )
    structural_controls = tuple(
        name for name in SCORE_VARIANTS if name != "oi_curvature_speed"
    )
    output = {
        "protocol": {
            "source_cutoff": "returned market and metrics frames hard-filtered strictly before 2024-01-01",
            "source_io_disclosure": "shared chunk parser may read and immediately discard later rows in the cutoff-crossing chunk; none enters returned frames or computation",
            "source_delay": "all Binance positioning fields delayed one complete 5m market row before hourly sampling",
            "mechanism": "6h three-cohort migration curvature x migration speed x positive 6h OI change; fit q80; fade net migration",
            "grid_size": 1,
            "hold": "fixed 12h",
            "quarantine": "2022 reported but excluded from admission because official top-trader fields have a large coverage gap",
            "entry": "completed minute-55 state enters next minute-00 open; 5m delay control enters minute-05",
            "leverage": 0.5,
            "cost": "6bp/side implementation cost",
            "strict_mdd": "favorable-first adverse-second OHLC high-water",
            "oos_opened": False,
            "contamination_note": "all pre-2024 outcomes exploratory; 2023 inspected internal selection; 2024+ excluded",
            "ontology_note": "curvature is observable cohort-ratio geometry; crowd exhaustion is a hypothesis, not an observed trader motive",
        },
        "support_only_preflight": {"performed_before_returns": True, **preflight},
        "primary": {"prelim_admitted": admitted(primary), "stats": primary},
        "controls": controls,
        "cost_stress": cost_stress,
        "novelty_audit": {
            "event_jaccard": event_overlap,
            "feature_spearman": feature_overlap,
            "novelty_pass": novelty_pass,
            "gate": "max event Jaccard <0.50 and max absolute score Spearman <0.80",
        },
        "final_admitted": bool(
            admitted(primary)
            and novelty_pass
            and admitted(controls["signal_delay_5m"])
            and not admitted(controls["direction_flip"])
            and not any(admitted(controls[name]) for name in structural_controls)
        ),
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    output = run(support_only=args.support_only)
    if args.support_only:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
