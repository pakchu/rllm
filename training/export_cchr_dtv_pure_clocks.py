# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportReturnType=false, reportCallIssue=false
"""Outcome-blind pure-clock exporter for frozen CCHR DTV comparators.

This module intentionally materializes only causal 5-minute inputs, derives the
primary debt-transfer-velocity signal, and exports six-column comparator clocks.
It does not import legacy search/evaluation modules and does not read outcomes.
"""

from __future__ import annotations

import itertools
from datetime import timezone
from typing import Any, Final, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from training import cchr_comparator_clock_common as clocks


MARKET_PATH: Final = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
MARKET_SHA256: Final = (
    "dbc9e53b09551b469168fe19cc750c5c3ea86278db3055d079103f7654050192"
)
MARKET_COLUMNS: Final = (
    "date",
    "close",
    "quote_asset_volume",
    "taker_buy_quote",
    "open_interest",
    "open_interest_value",
)
SPOT_PATH: Final = "data/cache_spot_premium_5m_2020-01-01_2026-06-01.csv.gz"
SPOT_SHA256: Final = "c21ed3f52804a3f879ef5167b6b81ee0fd7dd262f2e9d87ee5c1c25dbad73617"
SPOT_COLUMNS: Final = ("date", "spot_close", "spot_volume", "spot_rows")
METRICS_PATH: Final = "data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
METRICS_SHA256: Final = (
    "d391022352d5b14dea7ffd207a9d1f84f603d06ddae42da55dd792f722fc0106"
)
METRICS_COLUMNS: Final = (
    "create_time",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_long_short_ratio",
)
SOURCE_BINDINGS: Final = {
    "market": {
        "path": MARKET_PATH,
        "sha256": MARKET_SHA256,
        "columns": MARKET_COLUMNS,
    },
    "spot": {"path": SPOT_PATH, "sha256": SPOT_SHA256, "columns": SPOT_COLUMNS},
    "metrics": {
        "path": METRICS_PATH,
        "sha256": METRICS_SHA256,
        "columns": METRICS_COLUMNS,
    },
}

FIT_START: Final = pd.Timestamp("2020-10-15")
FIT_END: Final = pd.Timestamp("2023-01-01")
PRE2024_END: Final = pd.Timestamp("2024-01-01")
MEMORIES: Final = (72, 288)
ACCEPTANCE_HORIZONS: Final = (72, 288)
QUANTILES: Final = (0.90, 0.95)
HOLDS: Final = (72, 144, 288)
VARIANT: Final = "primary"


def candidate_id(
    memory: int, acceptance_horizon: int, quantile: float, hold: int
) -> str:
    """Return the byte-exact CCHR DTV member ID."""

    return (
        f"dtv:memory={memory}:accept={acceptance_horizon}:q={quantile:.2f}:hold={hold}"
    )


def comparator_candidate_map() -> dict[str, dict[str, Any]]:
    """Return the frozen CCHR prereg DTV family subset."""

    members: dict[str, dict[str, Any]] = {}
    for memory, acceptance, quantile, hold in itertools.product(
        MEMORIES, ACCEPTANCE_HORIZONS, QUANTILES, HOLDS
    ):
        members[candidate_id(memory, acceptance, quantile, hold)] = {
            "family": "dtv",
            "parameters": {
                "memory": memory,
                "acceptance_horizon": acceptance,
                "q": quantile,
            },
            "hold_bars": hold,
            "component_weight": None,
        }
    return dict(sorted(members.items()))


def candidate_ids() -> tuple[str, ...]:
    return tuple(comparator_candidate_map())


def candidate_map_hash() -> str:
    return clocks.candidate_map_hash(comparator_candidate_map())


def _utc_naive_series(values: pd.Series, *, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if not isinstance(parsed, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    return parsed.dt.tz_convert(None)


def _pre2024(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    dates = _utc_naive_series(cast(pd.Series, frame[column]), label=column)
    return frame.loc[dates < PRE2024_END].copy()


def _require_complete_five_minute_grid(frame: pd.DataFrame) -> None:
    intervals = cast(pd.Series, frame["date"]).diff().dropna()
    if len(intervals) and not bool(intervals.eq(pd.Timedelta("5min")).all()):
        raise ValueError("market rows must form a complete 5-minute grid before delay")


def attach_delayed_metrics(
    market: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    tolerance: str = "5min",
    delay_bars: int = 1,
) -> pd.DataFrame:
    """Attach metrics exactly as the legacy causal one-complete-bar delay did."""

    if delay_bars < 1:
        raise ValueError("delay_bars must be at least one complete source bar")
    left = market.copy()
    left["date"] = _utc_naive_series(cast(pd.Series, left["date"]), label="date")
    left = left.sort_values("date").reset_index(drop=True)
    if bool(cast(pd.Series, left["date"]).duplicated().any()):
        raise ValueError("market rows must have unique timestamps before delay")
    _require_complete_five_minute_grid(left)

    right = metrics.copy()
    right["create_time"] = _utc_naive_series(
        cast(pd.Series, right["create_time"]), label="create_time"
    )
    value_columns = [column for column in right.columns if column != "create_time"]
    merged = pd.merge_asof(
        left,
        right[["create_time", *value_columns]].sort_values("create_time"),
        left_on="date",
        right_on="create_time",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    delay = int(delay_bars)
    merged[value_columns] = merged[value_columns].shift(delay)
    merged["positioning_source_time"] = merged["create_time"].shift(delay)
    valid_source = cast(pd.Series, merged["positioning_source_time"]).notna()
    latest_allowed = cast(pd.Series, merged["date"]) - pd.Timedelta("5min") * delay
    too_recent = (
        cast(pd.Series, merged.loc[valid_source, "positioning_source_time"])
        > latest_allowed.loc[valid_source]
    )
    if bool(too_recent.any()):
        raise RuntimeError("positioning source delay is shorter than one complete bar")
    maximum_age = pd.Timedelta(tolerance) + pd.Timedelta("5min") * delay
    too_stale = (
        cast(pd.Series, merged.loc[valid_source, "date"])
        - cast(pd.Series, merged.loc[valid_source, "positioning_source_time"])
    ) > maximum_age
    if bool(too_stale.any()):
        raise RuntimeError("positioning source exceeded the bounded staleness contract")
    merged["positioning_available"] = (
        merged[value_columns].notna().all(axis=1) & valid_source
    ).astype(float)
    return merged.drop(columns=["create_time"])


def load_pre2024() -> pd.DataFrame:
    """Hash-bind and join only the frozen causal source columns before 2024."""

    market = clocks.read_hash_bound_prefix(
        MARKET_PATH,
        expected_sha256=MARKET_SHA256,
        columns=MARKET_COLUMNS,
        date_column="date",
        end_exclusive=PRE2024_END,
    )
    spot = clocks.read_hash_bound_prefix(
        SPOT_PATH,
        expected_sha256=SPOT_SHA256,
        columns=SPOT_COLUMNS,
        date_column="date",
        end_exclusive=PRE2024_END,
    )
    metrics = clocks.read_hash_bound_prefix(
        METRICS_PATH,
        expected_sha256=METRICS_SHA256,
        columns=METRICS_COLUMNS,
        date_column="create_time",
        end_exclusive=PRE2024_END,
    )
    market = _pre2024(market, "date")
    spot = _pre2024(spot, "date")
    metrics = _pre2024(metrics, "create_time")

    market["date"] = _utc_naive_series(cast(pd.Series, market["date"]), label="date")
    spot["date"] = _utc_naive_series(cast(pd.Series, spot["date"]), label="date")
    market = market.sort_values("date").drop_duplicates("date", keep="last")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last")
    joined = market.merge(
        spot[["date", "spot_close", "spot_volume", "spot_rows"]],
        on="date",
        how="left",
        validate="one_to_one",
    ).reset_index(drop=True)
    joined = attach_delayed_metrics(joined, metrics, tolerance="5min", delay_bars=1)
    dates = cast(pd.Series, pd.to_datetime(joined["date"], errors="raise"))
    if bool((dates >= PRE2024_END).any()):
        raise RuntimeError("future market rows opened")
    source = pd.to_datetime(joined["positioning_source_time"], errors="coerce")
    if isinstance(source, pd.Series) and bool((source.dropna() >= PRE2024_END).any()):
        raise RuntimeError("future metrics rows opened")
    return joined


def prior_z(values: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    minimum = max(288, window // 2) if minimum is None else minimum
    mean = prior.rolling(window, min_periods=minimum).mean()
    std = prior.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (numeric - mean) / std


def build_features(
    market: pd.DataFrame,
    memory: int,
    acceptance_horizon: int,
    variant: str = VARIANT,
) -> pd.DataFrame:
    """Reproduce the legacy primary DTV feature formula without outcomes."""

    if variant != VARIANT:
        raise ValueError("only the primary DTV variant is authorized for this exporter")
    close = pd.to_numeric(market["close"], errors="coerce")
    log_price = np.log(close.where(close > 0))
    oi = np.log(
        pd.to_numeric(market["sum_open_interest"], errors="coerce").where(
            lambda x: x > 0
        )
    )
    oi_value = np.log(
        pd.to_numeric(market["sum_open_interest_value"], errors="coerce").where(
            lambda x: x > 0
        )
    )
    global_ratio = np.log(
        pd.to_numeric(market["count_long_short_ratio"], errors="coerce").where(
            lambda x: x > 0
        )
    )
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")
    taker_flow = (
        ((2.0 * taker_buy / quote.replace(0.0, np.nan)) - 1.0)
        .rolling(12, min_periods=12)
        .mean()
    )
    owner = np.tanh(0.5 * prior_z(global_ratio, 288) + 0.5 * prior_z(taker_flow, 288))
    owner_change = owner - owner.shift(12)
    new_debt = (oi - oi.shift(1)).clip(lower=0.0)
    impulse = new_debt * owner_change
    transfer_velocity = impulse.ewm(
        halflife=memory, min_periods=memory, adjust=False
    ).mean()

    spot_close = pd.to_numeric(market["spot_close"], errors="coerce")
    spot_volume = pd.to_numeric(market["spot_volume"], errors="coerce")
    complete_spot = pd.to_numeric(market["spot_rows"], errors="coerce").eq(5)
    spot_notional_24h = (
        (spot_close * spot_volume)
        .where(complete_spot)
        .rolling(288, min_periods=276)
        .sum()
    )
    spot_liquidity_growth = np.log(spot_notional_24h / spot_notional_24h.shift(288))
    debt_growth = oi_value - oi_value.shift(288)
    cash_gap = prior_z(debt_growth - spot_liquidity_growth, 2016)
    acceptance = np.sign(transfer_velocity) * (
        log_price - log_price.shift(acceptance_horizon)
    )
    acceptance_z = prior_z(acceptance, 288)
    transfer_intensity_z = prior_z(transfer_velocity.abs(), 2016)
    cash_term = cash_gap.clip(lower=0.0)
    score = (
        transfer_intensity_z.clip(lower=0.0) * cash_term * acceptance_z.clip(lower=0.0)
    )
    source_valid = (
        pd.to_numeric(market["sum_open_interest"], errors="coerce").gt(0)
        & pd.to_numeric(market["sum_open_interest_value"], errors="coerce").gt(0)
        & pd.to_numeric(market["count_long_short_ratio"], errors="coerce").gt(0)
        & pd.to_datetime(market["positioning_source_time"], errors="coerce").notna()
        & complete_spot
    )
    score = score.where(source_valid)
    return pd.DataFrame(
        {
            "owner": owner,
            "owner_change": owner_change,
            "new_debt": new_debt,
            "transfer_velocity": transfer_velocity,
            "transfer_intensity_z": transfer_intensity_z,
            "cash_gap": cash_gap,
            "acceptance_z": acceptance_z,
            "score": score,
        }
    )


def fit_threshold(
    score: pd.Series,
    dates: pd.Series,
    quantile: float,
    *,
    min_positive: int = 1_000,
) -> float:
    """Fit positive-score threshold on [2020-10-15, 2023-01-01) only."""

    normalized_dates = _utc_naive_series(dates, label="dates")
    mask = (normalized_dates >= FIT_START) & (normalized_dates < FIT_END)
    values = pd.to_numeric(score.loc[mask], errors="coerce").dropna()
    values = values[values > 0]
    if len(values) < min_positive:
        raise ValueError(f"insufficient positive fit scores: {len(values)}")
    return float(values.quantile(quantile))


def signal_sides(features: pd.DataFrame, threshold: float) -> pd.Series:
    score = features["score"].to_numpy(float)
    velocity = features["transfer_velocity"].to_numpy(float)
    active = np.isfinite(score) & np.isfinite(velocity) & (score >= threshold)
    onset = active & ~np.r_[False, active[:-1]]
    side = -np.sign(velocity)
    side = np.where(onset & (side != 0), side, 0).astype(int)
    return pd.Series(side, index=features.index, name="side")


def raw_clock_candidates_for_member(
    market: pd.DataFrame,
    *,
    memory: int,
    acceptance_horizon: int,
    quantile: float,
    hold: int,
    min_positive: int = 1_000,
) -> list[clocks.ClockCandidate]:
    dates = _utc_naive_series(cast(pd.Series, market["date"]), label="date")
    features = build_features(market, memory, acceptance_horizon)
    threshold = fit_threshold(
        features["score"], dates, quantile, min_positive=min_positive
    )
    sides = signal_sides(features, threshold)
    member_id = candidate_id(memory, acceptance_horizon, quantile, hold)
    candidates: list[clocks.ClockCandidate] = []
    for row_index, side in sides[sides.ne(0)].items():
        signal_time = pd.Timestamp(
            dates.loc[row_index], tz=timezone.utc
        ).to_pydatetime()
        entry_time = signal_time + clocks.FIVE_MINUTES
        exit_time = entry_time + pd.Timedelta(minutes=5 * hold)
        candidates.append(
            clocks.ClockCandidate(
                candidate_id=member_id,
                causal_origins=(signal_time,),
                signal_time=signal_time,
                decision_time=entry_time,
                entry_time=entry_time,
                exit_time=exit_time,
                side=int(side),
            )
        )
    return candidates


def raw_clock_candidates(
    market: pd.DataFrame,
    *,
    min_positive: int = 1_000,
) -> list[clocks.ClockCandidate]:
    candidates: list[clocks.ClockCandidate] = []
    for memory, acceptance, quantile, hold in itertools.product(
        MEMORIES, ACCEPTANCE_HORIZONS, QUANTILES, HOLDS
    ):
        candidates.extend(
            raw_clock_candidates_for_member(
                market,
                memory=memory,
                acceptance_horizon=acceptance,
                quantile=quantile,
                hold=hold,
                min_positive=min_positive,
            )
        )
    return candidates


def build_clock_frame(
    market: pd.DataFrame,
    *,
    min_positive: int = 1_000,
) -> pd.DataFrame:
    """Build scheduled DTV clocks after split containment and per-ID non-overlap."""

    frame = clocks.schedule_candidates(
        raw_clock_candidates(market, min_positive=min_positive)
    )
    observed = set(cast(pd.Series, frame["candidate_id"]))
    unexpected = observed - set(candidate_ids())
    if unexpected:
        raise ValueError(f"unexpected DTV candidate IDs: {sorted(unexpected)}")
    return frame


def export_clock(
    market: pd.DataFrame,
    output_path: str,
    *,
    min_positive: int = 1_000,
    require_all_members: bool = True,
) -> str:
    frame = build_clock_frame(market, min_positive=min_positive)
    expected: Sequence[str] | None = candidate_ids() if require_all_members else None
    return clocks.write_deterministic_gzip_clock(
        frame, output_path, expected_candidate_ids=expected
    )


def source_bindings() -> Mapping[str, Mapping[str, Any]]:
    return SOURCE_BINDINGS
