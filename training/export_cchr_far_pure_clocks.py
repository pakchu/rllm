# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportReturnType=false, reportOperatorIssue=false, reportInvalidTypeForm=false, reportIndexIssue=false
"""Export outcome-blind FAR pure clocks for the frozen CCHR comparator map.

This module is a clean-room clock exporter for the funding-age rollover (FAR)
comparator family.  It reproduces only the causal feature semantics from
``training/search_funding_age_rollover_transfer_alpha.py`` and intentionally
never imports legacy search/evaluation modules or reads outcome/result JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from training.cchr_comparator_clock_common import (
    FIVE_MINUTES,
    ClockCandidate,
    read_hash_bound_prefix,
    schedule_candidates,
)


MARKET_PATH = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
MARKET_SHA256 = "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
MARKET_COLUMNS = (
    "date",
    "close",
    "quote_asset_volume",
    "taker_buy_quote",
    "open_interest",
    "open_interest_value",
)
METRICS_PATH = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
METRICS_SHA256 = "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
METRICS_COLUMNS = (
    "create_time",
    "sum_open_interest",
    "count_long_short_ratio",
)
FUNDING_PATH = (
    "data/binance_um_aux_btc_2020_2026/BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
FUNDING_SHA256 = "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
FUNDING_COLUMNS = ("date", "funding_time", "funding_rate")

PRE2024_END = pd.Timestamp("2024-01-01", tz="UTC")
FIT_START = pd.Timestamp("2020-10-15", tz="UTC")
FIT_END = pd.Timestamp("2023-01-01", tz="UTC")
GRID_MIN_AGE_SETTLEMENTS = (1, 3, 6)
GRID_HALF_LIFE_BARS = (288, 864)
GRID_HOLD_BARS = (72, 144)
THRESHOLD_Q = 0.90


@dataclass(frozen=True)
class FarFeatureBank:
    dates: pd.Series
    score: pd.Series
    causal_origin: pd.Series


def far_candidate_id(
    min_age_settlements: int, half_life_bars: int, hold_bars: int
) -> str:
    return (
        f"far:age={min_age_settlements}:half_life={half_life_bars}:"
        f"q={THRESHOLD_Q:.2f}:hold={hold_bars}"
    )


def far_candidate_map() -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for age in GRID_MIN_AGE_SETTLEMENTS:
        for half_life in GRID_HALF_LIFE_BARS:
            for hold in GRID_HOLD_BARS:
                members[far_candidate_id(age, half_life, hold)] = {
                    "family": "far",
                    "parameters": {
                        "min_age_settlements": age,
                        "half_life_bars": half_life,
                        "q": THRESHOLD_Q,
                    },
                    "hold_bars": hold,
                    "component_weight": None,
                }
    return dict(sorted(members.items()))


def _utc_series(values: pd.Series, *, label: str) -> pd.Series:
    stamps = pd.to_datetime(values, utc=True, errors="raise")
    if bool(stamps.isna().to_numpy(dtype=bool).any()):
        raise ValueError(f"{label} contains null timestamps")
    seconds = cast(pd.Series, stamps.dt.second)
    micros = cast(pd.Series, stamps.dt.microsecond)
    nanos = cast(pd.Series, stamps.dt.nanosecond)
    minutes = cast(pd.Series, stamps.dt.minute)
    aligned = seconds.eq(0) & micros.eq(0) & nanos.eq(0) & minutes.mod(5).eq(0)
    if not bool(aligned.to_numpy(dtype=bool).all()):
        raise ValueError(f"{label} must be aligned to the five-minute grid")
    return cast(pd.Series, stamps)


def _assert_complete_5m_grid(dates: pd.Series, *, label: str) -> None:
    if bool(dates.duplicated().to_numpy(dtype=bool).any()):
        raise ValueError(f"{label} rows must have unique timestamps")
    intervals = dates.diff().dropna()
    if len(intervals) and not bool(
        intervals.eq(FIVE_MINUTES).to_numpy(dtype=bool).all()
    ):
        raise ValueError(f"{label} rows must form a complete 5-minute grid")


def load_hash_bound_pre2024() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    market = read_hash_bound_prefix(
        MARKET_PATH,
        expected_sha256=MARKET_SHA256,
        columns=MARKET_COLUMNS,
        date_column="date",
        end_exclusive=PRE2024_END,
    )
    metrics = read_hash_bound_prefix(
        METRICS_PATH,
        expected_sha256=METRICS_SHA256,
        columns=METRICS_COLUMNS,
        date_column="create_time",
        end_exclusive=PRE2024_END,
    )
    funding = read_hash_bound_prefix(
        FUNDING_PATH,
        expected_sha256=FUNDING_SHA256,
        columns=FUNDING_COLUMNS,
        date_column="date",
        end_exclusive=PRE2024_END,
    )
    return prepare_pre2024_inputs(market, metrics, funding)


def prepare_pre2024_inputs(
    market: pd.DataFrame,
    metrics: pd.DataFrame,
    funding: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    causal_market = _attach_delayed_metrics(market, metrics)
    dates = _utc_series(causal_market["date"], label="market date")
    causal_market = causal_market.loc[dates < PRE2024_END].reset_index(drop=True)
    dates = cast(pd.Series, dates.loc[dates < PRE2024_END].reset_index(drop=True))
    if len(dates) and dates.max() >= PRE2024_END:
        raise RuntimeError("future market opened")
    source = pd.to_datetime(
        causal_market["positioning_source_time"], utc=True, errors="coerce"
    )
    if bool(source.notna().to_numpy(dtype=bool).any()) and source.max() >= PRE2024_END:
        raise RuntimeError("future metrics opened")
    event_rate = _funding_event_rate(dates, funding)
    return causal_market, dates, event_rate


def _attach_delayed_metrics(
    market: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    left = market.loc[:, list(MARKET_COLUMNS)].copy()
    left["date"] = _utc_series(left["date"], label="market date")
    left = left.sort_values("date").reset_index(drop=True)
    _assert_complete_5m_grid(cast(pd.Series, left["date"]), label="market")

    right = metrics.loc[:, list(METRICS_COLUMNS)].copy()
    right["create_time"] = _utc_series(
        right["create_time"], label="metrics create_time"
    )
    right = right.sort_values("create_time").reset_index(drop=True)
    if bool(right["create_time"].duplicated().to_numpy(dtype=bool).any()):
        raise ValueError("metrics rows must have unique timestamps")

    joined = pd.merge(
        left,
        right,
        left_on="date",
        right_on="create_time",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    value_columns = ["sum_open_interest", "count_long_short_ratio"]
    joined[value_columns] = joined[value_columns].shift(1)
    joined["positioning_source_time"] = joined["create_time"].shift(1)
    valid_source = pd.to_datetime(joined["positioning_source_time"], utc=True).notna()
    latest_allowed = cast(pd.Series, joined["date"]) - FIVE_MINUTES
    source = pd.to_datetime(
        joined["positioning_source_time"], utc=True, errors="coerce"
    )
    if bool(
        (source.loc[valid_source] > latest_allowed.loc[valid_source])
        .to_numpy(dtype=bool)
        .any()
    ):
        raise RuntimeError("positioning source delay is shorter than one complete bar")
    if bool(
        (source.loc[valid_source] != latest_allowed.loc[valid_source])
        .to_numpy(dtype=bool)
        .any()
    ):
        raise RuntimeError(
            "positioning source is stale; exact one-complete-bar delay required"
        )
    joined["positioning_available"] = (
        joined[value_columns].notna().all(axis=1) & valid_source
    ).astype(float)
    return joined.drop(columns=["create_time"])


def _funding_event_rate(dates: pd.Series, funding: pd.DataFrame) -> np.ndarray:
    event_rate = np.full(len(dates), np.nan, dtype=float)
    funding_dates = _utc_series(funding["date"], label="funding date")
    pre2024_funding = funding.loc[funding_dates < PRE2024_END].copy()
    if pre2024_funding.empty:
        return event_rate
    exact = pd.to_datetime(
        pd.to_numeric(pre2024_funding["funding_time"], errors="raise"),
        unit="ms",
        utc=True,
        errors="raise",
    )
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for timestamp, rate in zip(
        exact,
        pd.to_numeric(pre2024_funding["funding_rate"], errors="coerce"),
        strict=True,
    ):
        known = pd.Timestamp(timestamp).ceil("5min")
        pos = int(np.searchsorted(date_values, known.to_datetime64(), side="left"))
        if pos < len(event_rate) and np.isfinite(rate):
            event_rate[pos] = float(rate)
    return event_rate


def prior_z(values: pd.Series, window: int = 2016) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    mean = prior.rolling(window, min_periods=max(288, window // 2)).mean()
    std = (
        prior.rolling(window, min_periods=max(288, window // 2))
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    return (numeric - mean) / std


def owner_state(market: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ratio = pd.to_numeric(market["count_long_short_ratio"], errors="coerce")
    global_ratio = np.log(ratio.where(ratio > 0))
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    flow = (
        ((2.0 * taker_buy / quote.replace(0.0, np.nan)) - 1.0)
        .rolling(12, min_periods=12)
        .mean()
    )
    owner = np.tanh(0.5 * prior_z(global_ratio, 288) + 0.5 * prior_z(flow, 288))
    return owner, owner - owner.shift(12)


def build_features(
    market: pd.DataFrame,
    dates: pd.Series,
    event_rate: np.ndarray,
    *,
    min_age_settlements: int,
    half_life_bars: int,
) -> FarFeatureBank:
    if min_age_settlements < 0:
        raise ValueError("min_age_settlements must be non-negative")
    if half_life_bars <= 0:
        raise ValueError("half_life_bars must be positive")
    if len(market) != len(dates) or len(market) != len(event_rate):
        raise ValueError("market, dates, and event_rate lengths must match")

    owner, owner_change = owner_state(market)
    owner_np = owner.to_numpy(float)
    owner_change_np = owner_change.to_numpy(float)
    oi = pd.to_numeric(market["sum_open_interest"], errors="coerce").to_numpy(float)
    log_price = np.log(pd.to_numeric(market["close"], errors="coerce").to_numpy(float))
    max_age = 24
    long_w = np.zeros(max_age + 1)
    short_w = np.zeros(max_age + 1)
    long_entry = np.zeros(max_age + 1)
    short_entry = np.zeros(max_age + 1)
    long_funding = np.zeros(max_age + 1)
    short_funding = np.zeros(max_age + 1)
    settlement_origins: list[pd.Timestamp | None] = [None] * (max_age + 1)
    decay = float(np.exp(-np.log(2.0) / half_life_bars))
    signed_transfer = np.full(len(market), np.nan)
    causal_origins: list[pd.Timestamp | pd.NaT] = [pd.NaT] * len(market)

    def age_buckets(settlement_time: pd.Timestamp) -> None:
        for arr in (
            long_w,
            short_w,
            long_entry,
            short_entry,
            long_funding,
            short_funding,
        ):
            arr[-1] += arr[-2]
            arr[1:-1] = arr[:-2]
            arr[0] = 0.0
        settlement_origins[-1] = _oldest_origin(
            settlement_origins[-1], settlement_origins[-2]
        )
        settlement_origins[1:-1] = settlement_origins[:-2]
        settlement_origins[0] = None
        for index in range(1, max_age + 1):
            if (
                settlement_origins[index] is None
                and (long_w[index] + short_w[index]) > 1e-15
            ):
                settlement_origins[index] = settlement_time

    for i in range(1, len(market)):
        if not (
            np.isfinite(log_price[i])
            and np.isfinite(oi[i])
            and oi[i] > 0
            and np.isfinite(oi[i - 1])
            and oi[i - 1] > 0
        ):
            continue
        retention = decay * min(1.0, oi[i] / oi[i - 1])
        for arr in (
            long_w,
            short_w,
            long_entry,
            short_entry,
            long_funding,
            short_funding,
        ):
            arr *= retention
        if np.isfinite(event_rate[i]):
            age_buckets(pd.Timestamp(dates.iloc[i]))
            long_funding += long_w * float(event_rate[i])
            short_funding -= short_w * float(event_rate[i])

        ages = slice(min_age_settlements, None)
        lw = long_w[ages]
        sw = short_w[ages]
        lb = np.divide(
            long_entry[ages], lw, out=np.full_like(lw, np.nan), where=lw > 1e-15
        )
        sb = np.divide(
            short_entry[ages], sw, out=np.full_like(sw, np.nan), where=sw > 1e-15
        )
        lf = np.divide(
            long_funding[ages], lw, out=np.full_like(lw, np.nan), where=lw > 1e-15
        )
        sf = np.divide(
            short_funding[ages], sw, out=np.full_like(sw, np.nan), where=sw > 1e-15
        )
        long_per_weight = np.maximum(lf - (log_price[i] - lb), 0.0)
        short_per_weight = np.maximum(sf + (log_price[i] - sb), 0.0)
        long_burden = float(np.nansum(lw * long_per_weight))
        short_burden = float(np.nansum(sw * short_per_weight))
        if np.isfinite(owner_change_np[i]):
            long_transfer = long_burden * max(-owner_change_np[i], 0.0)
            short_transfer = short_burden * max(owner_change_np[i], 0.0)
            signed_transfer[i] = short_transfer - long_transfer
            origin = _oldest_live_origin(
                settlement_origins[min_age_settlements:],
                long_w[min_age_settlements:],
                short_w[min_age_settlements:],
            )
            if origin is not None:
                causal_origins[i] = origin

        delta = max(oi[i] - oi[i - 1], 0.0) / oi[i - 1]
        if delta > 0 and np.isfinite(owner_np[i]):
            long_add = delta * max(owner_np[i], 0.0)
            short_add = delta * max(-owner_np[i], 0.0)
            long_w[0] += long_add
            short_w[0] += short_add
            long_entry[0] += long_add * log_price[i]
            short_entry[0] += short_add * log_price[i]

    transfer_series = pd.Series(signed_transfer, index=market.index)
    intensity_z = prior_z(transfer_series.abs(), 2016)
    score = np.sign(transfer_series) * intensity_z.clip(lower=0.0)
    return FarFeatureBank(
        dates=dates.reset_index(drop=True),
        score=score.reset_index(drop=True),
        causal_origin=pd.Series(causal_origins),
    )


def _oldest_origin(
    left: pd.Timestamp | None, right: pd.Timestamp | None
) -> pd.Timestamp | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if left <= right else right


def _oldest_live_origin(
    origins: Sequence[pd.Timestamp | None], long_w: np.ndarray, short_w: np.ndarray
) -> pd.Timestamp | None:
    live: list[pd.Timestamp] = []
    for origin, weight in zip(origins, long_w + short_w, strict=True):
        if origin is not None and weight > 1e-15:
            live.append(origin)
    return min(live) if live else None


def fit_abs_threshold(
    score: pd.Series,
    dates: pd.Series,
    *,
    q: float = THRESHOLD_Q,
    min_observations: int = 10_000,
) -> float:
    mask = (dates >= FIT_START) & (dates < FIT_END)
    values = score.loc[mask].abs().dropna()
    if len(values) < min_observations:
        raise ValueError(f"insufficient fit scores {len(values)}")
    return float(values.quantile(q))


def signal_arrays(
    features: FarFeatureBank, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    score = features.score.to_numpy(float)
    active = np.isfinite(score) & (np.abs(score) >= threshold)
    onset = active & ~np.r_[False, active[:-1]]
    side = np.sign(score)
    return onset & (side > 0), onset & (side < 0)


def raw_candidates_for_member(
    features: FarFeatureBank,
    *,
    candidate_id: str,
    threshold: float,
    hold_bars: int,
) -> list[ClockCandidate]:
    long_active, short_active = signal_arrays(features, threshold)
    candidates: list[ClockCandidate] = []
    for index, side in _signal_indices(long_active, short_active):
        signal_time = pd.Timestamp(features.dates.iloc[index]).to_pydatetime()
        causal_origin = features.causal_origin.iloc[index]
        if pd.isna(causal_origin):
            causal_origin = pd.Timestamp(signal_time)
        candidates.append(
            ClockCandidate(
                candidate_id=candidate_id,
                causal_origins=(pd.Timestamp(causal_origin).to_pydatetime(),),
                signal_time=signal_time,
                decision_time=signal_time + FIVE_MINUTES.to_pytimedelta(),
                entry_time=signal_time + FIVE_MINUTES.to_pytimedelta(),
                exit_time=signal_time + (hold_bars + 1) * FIVE_MINUTES,
                side=side,
            )
        )
    return candidates


def _signal_indices(
    long_active: np.ndarray, short_active: np.ndarray
) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for index, is_long in enumerate(long_active):
        if bool(is_long):
            rows.append((index, 1))
    for index, is_short in enumerate(short_active):
        if bool(is_short):
            rows.append((index, -1))
    return sorted(rows)


def build_far_clock_frame(
    market: pd.DataFrame,
    dates: pd.Series,
    event_rate: np.ndarray,
    *,
    min_fit_observations: int = 10_000,
) -> pd.DataFrame:
    raw: list[ClockCandidate] = []
    for age in GRID_MIN_AGE_SETTLEMENTS:
        for half_life in GRID_HALF_LIFE_BARS:
            features = build_features(
                market,
                dates,
                event_rate,
                min_age_settlements=age,
                half_life_bars=half_life,
            )
            threshold = fit_abs_threshold(
                features.score,
                dates,
                q=THRESHOLD_Q,
                min_observations=min_fit_observations,
            )
            for hold in GRID_HOLD_BARS:
                raw.extend(
                    raw_candidates_for_member(
                        features,
                        candidate_id=far_candidate_id(age, half_life, hold),
                        threshold=threshold,
                        hold_bars=hold,
                    )
                )
    return schedule_candidates(raw)
