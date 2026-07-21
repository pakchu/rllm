"""Outcome-blind pure-clock exporter for the frozen CCHR PDLH comparator grid.

This module intentionally exposes reusable primitives only.  It has no CLI and
must not import the legacy outcome-bearing search module or read non-causal
artifacts.  Real source loading, when separately authorized, is limited to the
hash-bound causal allowlists in ``load_causal_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import itertools
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training.cchr_comparator_clock_common import (
    ClockCandidate,
    candidate_map_hash,
    read_hash_bound_prefix,
    schedule_candidates,
    validate_clock_frame,
)

MARKET_PATH = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA256 = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
MARKET_COLUMNS = ("date",)

METRICS_PATH = Path("data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz")
METRICS_SHA256 = "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
METRICS_COLUMNS = (
    "create_time",
    "sum_toptrader_long_short_ratio",
    "count_toptrader_long_short_ratio",
    "count_long_short_ratio",
)

CUTOFF = pd.Timestamp("2024-01-01T00:00:00Z")
QUARANTINE_START = pd.Timestamp("2022-01-01T00:00:00Z")
QUARANTINE_END = pd.Timestamp("2023-01-01T00:00:00Z")
PRIOR_Z_WINDOW = 8640
PRIOR_Z_MIN_PERIODS = 4320
ENTRY_Z = 1.5
CONTRACTION_FRACTION = 0.5
RESET_Z = 0.25
FIVE_MINUTES = cast(pd.Timedelta, pd.Timedelta(minutes=5))
MERGE_TOLERANCE = cast(pd.Timedelta, pd.Timedelta(minutes=5))
STALENESS_LIMIT = cast(pd.Timedelta, pd.Timedelta(minutes=10))
SOURCE_DELAY_BARS = 1
STATES = ("top_position_minus_global", "top_account_minus_global")
MIN_AGES = (144, 432)
TRIGGERS = ("contraction", "zero_cross")
HOLDS = (72, 216)


@dataclass(frozen=True)
class LifecycleSignal:
    """One first-resolution signal before hold-grid expansion."""

    state: str
    min_age: int
    trigger: str
    episode_start: datetime
    signal_time: datetime
    side: int


def pdlh_candidate_map() -> dict[str, dict[str, Any]]:
    """Return the byte-exact PDLH subset of the frozen CCHR member map."""

    members: dict[str, dict[str, Any]] = {}
    for state, min_age, trigger, hold in itertools.product(
        STATES, MIN_AGES, TRIGGERS, HOLDS
    ):
        candidate_id = f"pdlh:{state}:age={min_age}:trigger={trigger}:hold={hold}"
        members[candidate_id] = {
            "family": "pdlh",
            "parameters": {
                "disagreement": state,
                "min_age": min_age,
                "trigger": trigger,
            },
            "hold_bars": hold,
            "component_weight": None,
        }
    return dict(sorted(members.items()))


def expected_candidate_ids() -> tuple[str, ...]:
    return tuple(pdlh_candidate_map())


def pdlh_candidate_map_hash() -> str:
    return candidate_map_hash(pdlh_candidate_map())


def load_causal_inputs(
    *,
    market_path: str | Path = MARKET_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash-bind and load only the frozen causal input column allowlists."""

    market = read_hash_bound_prefix(
        market_path,
        expected_sha256=MARKET_SHA256,
        columns=MARKET_COLUMNS,
        date_column="date",
        end_exclusive="2024-01-01T00:00:00Z",
    )
    metrics = read_hash_bound_prefix(
        metrics_path,
        expected_sha256=METRICS_SHA256,
        columns=METRICS_COLUMNS,
        date_column="create_time",
        end_exclusive="2024-01-01T00:00:00Z",
    )
    return market, metrics


def _five_minute_utc_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="raise")
    series = pd.Series(values, index=frame.index, name=column)
    if bool(series.isna().to_numpy(dtype=bool).any()):
        raise ValueError(f"{column} contains missing timestamps")
    misaligned = (
        (series.dt.second != 0)
        | (series.dt.microsecond != 0)
        | (series.dt.nanosecond != 0)
        | (series.dt.minute % 5 != 0)
    )
    if bool(misaligned.to_numpy(dtype=bool).any()):
        raise ValueError(f"{column} must be aligned to the five-minute grid")
    return series


def attach_delayed_metrics(market: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach metrics delayed by one complete 5m bar with no interpolation/ffill."""

    market_dates = _five_minute_utc_series(market, "date")
    if bool(market_dates.duplicated().to_numpy(dtype=bool).any()):
        raise ValueError("market rows must have unique five-minute timestamps")
    ordered_market = (
        market.assign(date=market_dates).sort_values("date").reset_index(drop=True)
    )
    intervals = ordered_market["date"].diff().dropna()
    if len(intervals) and not bool((intervals == FIVE_MINUTES).all()):
        raise ValueError("market rows must form a complete five-minute grid")
    if bool((ordered_market["date"] >= CUTOFF).to_numpy(dtype=bool).any()):
        raise ValueError("post-2023 market rows are outside the PDLH pure-clock scope")

    metric_times = _five_minute_utc_series(metrics, "create_time")
    if bool(metric_times.duplicated().to_numpy(dtype=bool).any()):
        raise ValueError("metrics rows must have unique five-minute timestamps")
    value_columns = [column for column in METRICS_COLUMNS if column != "create_time"]
    ordered_metrics = metrics.assign(create_time=metric_times).sort_values(
        "create_time"
    )
    if bool((ordered_metrics["create_time"] >= CUTOFF).to_numpy(dtype=bool).any()):
        raise ValueError("post-2023 metrics rows are outside the PDLH pure-clock scope")

    joined = pd.merge_asof(
        ordered_market,
        ordered_metrics[["create_time", *value_columns]],
        left_on="date",
        right_on="create_time",
        direction="backward",
        tolerance=MERGE_TOLERANCE,
    )
    joined[value_columns] = joined[value_columns].shift(SOURCE_DELAY_BARS)
    joined["positioning_source_time"] = joined["create_time"].shift(SOURCE_DELAY_BARS)
    joined = joined.drop(columns=["create_time"])

    source_time = pd.to_datetime(joined["positioning_source_time"], utc=True)
    available = source_time.notna()
    latest_allowed = joined["date"] - SOURCE_DELAY_BARS * FIVE_MINUTES
    too_recent = available & (source_time > latest_allowed)
    too_stale = available & ((joined["date"] - source_time) > STALENESS_LIMIT)
    if bool(too_recent.to_numpy(dtype=bool).any()):
        raise RuntimeError("positioning source delay is shorter than one complete bar")
    if bool(too_stale.to_numpy(dtype=bool).any()):
        raise RuntimeError("positioning source is stale beyond ten minutes")

    numeric_complete = joined[value_columns].notna().all(axis=1)
    joined["positioning_available"] = (available & numeric_complete).astype(bool)
    return cast(pd.DataFrame, joined)


def prior_z(
    values: pd.Series, window: int = PRIOR_Z_WINDOW, minimum: int = PRIOR_Z_MIN_PERIODS
) -> pd.Series:
    """Prior-only z-score: the current row is excluded from mean/std estimates."""

    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    prior = numeric.shift(1)
    rolling = prior.rolling(window, min_periods=minimum)
    std = rolling.std(ddof=0).replace(0.0, np.nan)
    return cast(pd.Series, (numeric - rolling.mean()) / std)


def _positive_log(values: Any) -> pd.Series:
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    return cast(pd.Series, np.log(numeric.where(numeric > 0.0)))


def build_disagreement_states(market_with_metrics: pd.DataFrame) -> pd.DataFrame:
    top_position = _positive_log(market_with_metrics["sum_toptrader_long_short_ratio"])
    top_account = _positive_log(market_with_metrics["count_toptrader_long_short_ratio"])
    global_account = _positive_log(market_with_metrics["count_long_short_ratio"])
    states = pd.DataFrame(
        {
            "top_position_minus_global": prior_z(top_position - global_account),
            "top_account_minus_global": prior_z(top_account - global_account),
        },
        index=market_with_metrics.index,
    )
    return cast(pd.DataFrame, states.replace([np.inf, -np.inf], np.nan))


def positioning_valid_mask(dates: Any, available: Any) -> np.ndarray:
    date_values = pd.to_datetime(dates, utc=True, errors="raise")
    quarantine = (date_values >= QUARANTINE_START) & (date_values < QUARANTINE_END)
    pre_2024 = date_values < CUTOFF
    return (
        np.asarray(available, dtype=bool)
        & pre_2024.to_numpy(bool)
        & ~quarantine.to_numpy(bool)
    )


def lifecycle_signals_for_state(
    dates: Any,
    values: Any,
    valid: Any,
    *,
    state: str,
    min_age: int,
    trigger: str,
) -> list[LifecycleSignal]:
    """Emit legacy-primary first resolutions for one PDLH state/min-age/trigger."""

    if state not in STATES:
        raise KeyError(state)
    if min_age not in MIN_AGES:
        raise KeyError(min_age)
    if trigger not in TRIGGERS:
        raise KeyError(trigger)
    date_index = pd.to_datetime(pd.Series(dates), utc=True, errors="raise")
    disagreement_z = np.asarray(values, dtype=float)
    valid_mask = np.asarray(valid, dtype=bool)
    if len(date_index) != len(disagreement_z) or len(date_index) != len(valid_mask):
        raise ValueError("dates, values, and valid masks must have identical lengths")

    signals: list[LifecycleSignal] = []
    active = False
    episode_side = 0
    episode_start: pd.Timestamp | None = None
    age = 0
    peak = 0.0
    fired = False
    previous_time: pd.Timestamp | None = None

    for position, stamp in enumerate(date_index):
        if previous_time is not None and stamp - previous_time != FIVE_MINUTES:
            active = False
            episode_side = 0
            episode_start = None
            age = 0
            peak = 0.0
            fired = False
        previous_time = stamp

        value = float(disagreement_z[position])
        if not valid_mask[position] or not np.isfinite(value):
            active = False
            episode_side = 0
            episode_start = None
            age = 0
            peak = 0.0
            fired = False
            continue

        current_side = int(np.sign(value))
        if not active:
            if abs(value) >= ENTRY_Z:
                active = True
                episode_side = current_side
                episode_start = stamp
                age = 1
                peak = abs(value)
                fired = False
            continue

        age += 1
        peak = max(peak, abs(value))
        crossed = current_side != 0 and current_side != episode_side
        contracted = (
            current_side == episode_side
            and peak > 0.0
            and abs(value) <= CONTRACTION_FRACTION * peak
        )
        if trigger == "zero_cross":
            resolved = crossed
        elif trigger == "contraction":
            resolved = contracted
        else:
            raise KeyError(trigger)

        if resolved and age >= min_age and not fired:
            side = -episode_side
            if episode_start is None:
                raise RuntimeError("active PDLH episode lost its start timestamp")
            signals.append(
                LifecycleSignal(
                    state=state,
                    min_age=min_age,
                    trigger=trigger,
                    episode_start=episode_start.to_pydatetime(),
                    signal_time=stamp.to_pydatetime(),
                    side=side,
                )
            )
            fired = True

        if crossed:
            if abs(value) >= ENTRY_Z:
                active = True
                episode_side = current_side
                episode_start = stamp
                age = 1
                peak = abs(value)
                fired = False
            else:
                active = False
                episode_side = 0
                episode_start = None
                age = 0
                peak = 0.0
                fired = False
        elif fired and abs(value) < RESET_Z:
            active = False
            episode_side = 0
            episode_start = None
            age = 0
            peak = 0.0
            fired = False

    return signals


def raw_clock_candidates_from_states(
    dates: Any,
    states: pd.DataFrame,
    valid: Any,
) -> list[ClockCandidate]:
    raw: list[ClockCandidate] = []
    for state, min_age, trigger, hold in itertools.product(
        STATES, MIN_AGES, TRIGGERS, HOLDS
    ):
        signals = lifecycle_signals_for_state(
            dates,
            states[state].to_numpy(dtype=float),
            valid,
            state=state,
            min_age=min_age,
            trigger=trigger,
        )
        candidate_id = f"pdlh:{state}:age={min_age}:trigger={trigger}:hold={hold}"
        for signal in signals:
            signal_time = cast(
                pd.Timestamp, pd.Timestamp(signal.signal_time).tz_convert("UTC")
            )
            entry_time = cast(pd.Timestamp, signal_time + FIVE_MINUTES)
            raw.append(
                ClockCandidate(
                    candidate_id=candidate_id,
                    causal_origins=(signal.episode_start,),
                    signal_time=cast(datetime, signal_time.to_pydatetime()),
                    decision_time=cast(datetime, entry_time.to_pydatetime()),
                    entry_time=cast(datetime, entry_time.to_pydatetime()),
                    exit_time=cast(
                        datetime,
                        cast(
                            pd.Timestamp, entry_time + hold * FIVE_MINUTES
                        ).to_pydatetime(),
                    ),
                    side=signal.side,
                )
            )
    return raw


def build_pdlh_clock_from_states(
    dates: Any,
    states: pd.DataFrame,
    valid: Any,
) -> pd.DataFrame:
    """Build the scheduled six-column PDLH clock from causal state arrays."""

    if tuple(states.columns) != STATES:
        raise ValueError("PDLH states must use the exact frozen state column order")
    frame = schedule_candidates(raw_clock_candidates_from_states(dates, states, valid))
    return validate_clock_frame(frame)


def build_pdlh_clock(market: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the scheduled six-column PDLH clock from causal input frames."""

    joined = attach_delayed_metrics(market, metrics)
    states = build_disagreement_states(joined)
    valid = positioning_valid_mask(joined["date"], joined["positioning_available"])
    return build_pdlh_clock_from_states(joined["date"], states, valid)


def export_manifest(
    clock: pd.DataFrame, clock_sha256: str | None = None
) -> dict[str, Any]:
    """Describe a produced clock without embedding real source rows or outcomes."""

    return {
        "family": "pdlh",
        "source_bindings": {
            "market": {
                "path": str(MARKET_PATH),
                "sha256": MARKET_SHA256,
                "columns": list(MARKET_COLUMNS),
            },
            "metrics": {
                "path": str(METRICS_PATH),
                "sha256": METRICS_SHA256,
                "columns": list(METRICS_COLUMNS),
            },
        },
        "protocol": {
            "pre_2024_only": True,
            "metrics_delay_bars": SOURCE_DELAY_BARS,
            "metrics_merge_tolerance_minutes": 5,
            "metrics_staleness_limit_minutes": 10,
            "prior_z_window": PRIOR_Z_WINDOW,
            "prior_z_min_periods": PRIOR_Z_MIN_PERIODS,
            "entry_z": ENTRY_Z,
            "contraction_fraction": CONTRACTION_FRACTION,
            "reset_z": RESET_Z,
            "flip": False,
            "ignore_age": False,
            "schedule": "split containment first, then independent non-overlap per candidate_id",
        },
        "candidate_map": pdlh_candidate_map(),
        "candidate_map_hash": pdlh_candidate_map_hash(),
        "clock_rows": int(len(clock)),
        "clock_sha256": clock_sha256,
        "outcome_boundary": {
            "outcome_json_parsed": 0,
            "evaluator_imports": 0,
            "price_path_columns_loaded": 0,
            "payoff_fields_loaded": 0,
            "real_execution_cli": False,
        },
    }
