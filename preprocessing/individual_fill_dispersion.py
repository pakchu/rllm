"""Frozen IFDA-72 five-minute features from raw Binance fills.

The public API is intentionally small for streaming builders: validate one UTC
calendar day of canonical raw individual trades plus Binance aggregate trades and
return a complete 288-row five-minute grid.  The features are outcome-blind and
contain no OHLC fields.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Final, Iterable, cast

import numpy as np
import pandas as pd

FIVE_MINUTES: Final[str] = "5min"
FIVE_MINUTES_MS: Final[int] = 5 * 60 * 1_000
DAY_MS: Final[int] = 24 * 60 * 60 * 1_000
QUOTE_QTY_RECONCILIATION_TOLERANCE: Final[float] = 0.011

RAW_TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "trade_id",
    "price",
    "quantity",
    "quote_qty",
    "transact_time",
    "is_buyer_maker",
)

RAW_AGGTRADE_COLUMNS: Final[tuple[str, ...]] = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)

_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "source_observed",
    "source_complete",
    "source_gap_day",
    "verified_zero_volume_empty",
    "post_gap_quarantine",
)

_FEATURE_SUFFIXES: Final[tuple[str, ...]] = (
    "first_trade_id",
    "last_trade_id",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "fill_count",
    "buy_fill_count",
    "sell_fill_count",
    "quote_notional",
    "buy_quote_notional",
    "sell_quote_notional",
    "signed_quote_notional",
    "flow_coherence",
    "buy_fill_hhi",
    "buy_fill_effective_count",
    "buy_fill_normalized_effective_count",
    "sell_fill_hhi",
    "sell_fill_effective_count",
    "sell_fill_normalized_effective_count",
    "all_fill_hhi",
    "all_fill_effective_count",
    "all_fill_normalized_effective_count",
    "dominant_side",
    "dominant_quote_notional",
    "opposing_quote_notional",
    "dominant_fill_count",
    "opposing_fill_count",
    "dominant_equalization",
    "opposing_equalization",
    "equalization_gap",
    "ifda_score",
    "buy_avg_quote_notional",
    "sell_avg_quote_notional",
    "all_fill_avg_quote_notional",
)

_AGG_FEATURE_SUFFIXES: Final[tuple[str, ...]] = tuple(
    column
    .replace("fill_count", "event_count")
    .replace("fill_hhi", "event_hhi")
    .replace("fill_effective_count", "event_effective_count")
    .replace("fill_normalized_effective_count", "event_normalized_effective_count")
    .replace("fill_avg_quote_notional", "event_avg_quote_notional")
    for column in _FEATURE_SUFFIXES
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    _SOURCE_COLUMNS + _FEATURE_SUFFIXES + tuple(f"agg_{column}" for column in _AGG_FEATURE_SUFFIXES)
)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _as_utc_midnight(day: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = cast(pd.Timestamp, pd.Timestamp(day))
    if pd.isna(timestamp):
        raise ValueError("day must be a valid timestamp")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.normalize()


def _empty_grid(day_start: pd.Timestamp) -> pd.DataFrame:
    grid = pd.date_range(day_start, periods=288, freq=FIVE_MINUTES, tz=timezone.utc)
    output = pd.DataFrame({"date": grid})
    output["source_observed"] = False
    output["source_complete"] = True
    output["source_gap_day"] = False
    output["verified_zero_volume_empty"] = True
    output["post_gap_quarantine"] = False
    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = 0
    return cast(pd.DataFrame, output.loc[:, OUTPUT_COLUMNS])


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} frame is missing columns: {missing}")


def _bool_array(values: pd.Series, column: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool, copy=False)
    text = values.astype("string").str.strip().str.lower()
    if not text.isin(["true", "false"]).all():
        raise ValueError(f"{column} must contain only boolean maker flags")
    return text.eq("true").to_numpy(dtype=bool, copy=False)


def _numeric_array(frame: pd.DataFrame, column: str, *, positive: bool) -> np.ndarray:
    numeric_series = cast(
        pd.Series, pd.to_numeric(_series(frame, column), errors="raise")
    )
    numeric = cast(
        np.ndarray, numeric_series.to_numpy(dtype="float64", copy=False)
    )
    if not np.isfinite(numeric).all() or (positive and (numeric <= 0.0).any()):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{column} must contain {qualifier} values")
    return numeric


def _int_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    numeric = cast(pd.Series, pd.to_numeric(_series(frame, column), errors="raise"))
    as_float = cast(np.ndarray, numeric.to_numpy(dtype="float64", copy=False))
    if not np.isfinite(as_float).all() or not np.equal(as_float, np.floor(as_float)).all():
        raise ValueError(f"{column} must contain finite integer values")
    return cast(np.ndarray, as_float.astype(np.int64))


def _validate_day_timestamps(
    timestamps: np.ndarray, *, day_start_ms: int, name: str
) -> None:
    if len(timestamps) == 0:
        return
    if (timestamps < day_start_ms).any() or (timestamps >= day_start_ms + DAY_MS).any():
        raise ValueError(f"{name} timestamps must be within the requested UTC day")
    if (np.diff(timestamps) < 0).any():
        raise ValueError(f"{name} timestamps must be monotone nondecreasing")


def _validate_unique_monotone_ids(ids: np.ndarray, *, name: str) -> None:
    if len(ids) == 0:
        return
    if (ids < 0).any() or (np.diff(ids) <= 0).any():
        raise ValueError(f"{name} IDs must be monotone, unique, and nonnegative")


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0.0)
    return result


def _feature_frame(
    events: pd.DataFrame,
    *,
    grid: pd.DatetimeIndex,
    id_column: str,
    output_suffixes: tuple[str, ...],
    prefix: str = "",
) -> pd.DataFrame:
    columns = [f"{prefix}{column}" for column in output_suffixes]
    if events.empty:
        return pd.DataFrame(0, index=grid, columns=pd.Index(columns))

    grouped = events.groupby("date", sort=True, observed=True)
    bar = cast(
        pd.DataFrame,
        grouped.agg(
            first_trade_id=(id_column, "first"),
            last_trade_id=(id_column, "last"),
            first_transact_time_ms=("transact_time", "first"),
            last_transact_time_ms=("transact_time", "last"),
            fill_count=(id_column, "size"),
            buy_fill_count=("buy_fill", "sum"),
            sell_fill_count=("sell_fill", "sum"),
            quote_notional=("quote_notional", "sum"),
            buy_quote_notional=("buy_quote_notional", "sum"),
            sell_quote_notional=("sell_quote_notional", "sum"),
            signed_quote_notional=("signed_quote_notional", "sum"),
            buy_fill_hhi_numerator=("buy_quote_notional_squared", "sum"),
            sell_fill_hhi_numerator=("sell_quote_notional_squared", "sum"),
            all_fill_hhi_numerator=("quote_notional_squared", "sum"),
        ),
    )
    total = _series(bar, "quote_notional").to_numpy(dtype=np.float64)
    buy = _series(bar, "buy_quote_notional").to_numpy(dtype=np.float64)
    sell = _series(bar, "sell_quote_notional").to_numpy(dtype=np.float64)
    signed = _series(bar, "signed_quote_notional").to_numpy(dtype=np.float64)
    counts = _series(bar, "fill_count").to_numpy(dtype=np.float64)
    buy_counts = _series(bar, "buy_fill_count").to_numpy(dtype=np.float64)
    sell_counts = _series(bar, "sell_fill_count").to_numpy(dtype=np.float64)

    bar["flow_coherence"] = _safe_divide(np.abs(signed), total)
    bar["buy_fill_hhi"] = _safe_divide(
        _series(bar, "buy_fill_hhi_numerator").to_numpy(dtype=np.float64), buy * buy
    )
    buy_hhi = _series(bar, "buy_fill_hhi").to_numpy(dtype=np.float64)
    bar["buy_fill_effective_count"] = _safe_divide(np.ones_like(buy_hhi), buy_hhi)
    bar["buy_fill_normalized_effective_count"] = _safe_divide(
        _series(bar, "buy_fill_effective_count").to_numpy(dtype=np.float64), buy_counts
    )
    bar["sell_fill_hhi"] = _safe_divide(
        _series(bar, "sell_fill_hhi_numerator").to_numpy(dtype=np.float64), sell * sell
    )
    sell_hhi = _series(bar, "sell_fill_hhi").to_numpy(dtype=np.float64)
    bar["sell_fill_effective_count"] = _safe_divide(np.ones_like(sell_hhi), sell_hhi)
    bar["sell_fill_normalized_effective_count"] = _safe_divide(
        _series(bar, "sell_fill_effective_count").to_numpy(dtype=np.float64), sell_counts
    )
    bar["all_fill_hhi"] = _safe_divide(
        _series(bar, "all_fill_hhi_numerator").to_numpy(dtype=np.float64), total * total
    )
    all_hhi = _series(bar, "all_fill_hhi").to_numpy(dtype=np.float64)
    bar["all_fill_effective_count"] = _safe_divide(np.ones_like(all_hhi), all_hhi)
    bar["all_fill_normalized_effective_count"] = _safe_divide(
        _series(bar, "all_fill_effective_count").to_numpy(dtype=np.float64), counts
    )

    dominant_is_buy = buy >= sell
    has_flow = total > 0.0
    tie = np.isclose(buy, sell, atol=0.0, rtol=0.0)
    bar["dominant_side"] = np.where(has_flow & ~tie, np.where(dominant_is_buy, 1, -1), 0)
    dominant_quote = np.maximum(buy, sell)
    opposing_quote = np.minimum(buy, sell)
    dominant_count = np.where(dominant_is_buy, buy_counts, sell_counts)
    opposing_count = np.where(dominant_is_buy, sell_counts, buy_counts)
    dominant_equalization = np.where(
        dominant_is_buy,
        _series(bar, "buy_fill_normalized_effective_count").to_numpy(dtype=np.float64),
        _series(bar, "sell_fill_normalized_effective_count").to_numpy(dtype=np.float64),
    )
    opposing_equalization = np.where(
        dominant_is_buy,
        _series(bar, "sell_fill_normalized_effective_count").to_numpy(dtype=np.float64),
        _series(bar, "buy_fill_normalized_effective_count").to_numpy(dtype=np.float64),
    )
    dominant_count = np.where(has_flow, dominant_count, 0.0)
    opposing_count = np.where(has_flow, opposing_count, 0.0)
    dominant_equalization = np.where(has_flow & ~tie, dominant_equalization, 0.0)
    opposing_equalization = np.where(has_flow & ~tie, opposing_equalization, 0.0)
    bar["dominant_quote_notional"] = dominant_quote
    bar["opposing_quote_notional"] = opposing_quote
    bar["dominant_fill_count"] = dominant_count
    bar["opposing_fill_count"] = opposing_count
    bar["dominant_equalization"] = dominant_equalization
    bar["opposing_equalization"] = opposing_equalization
    bar["equalization_gap"] = np.maximum(dominant_equalization - opposing_equalization, 0.0)
    bar["ifda_score"] = (
        _series(bar, "flow_coherence")
        * _series(bar, "dominant_equalization")
        * _series(bar, "equalization_gap")
    )
    bar["buy_avg_quote_notional"] = _safe_divide(buy, buy_counts)
    bar["sell_avg_quote_notional"] = _safe_divide(sell, sell_counts)
    bar["all_fill_avg_quote_notional"] = _safe_divide(total, counts)

    bar = cast(pd.DataFrame, bar.loc[:, _FEATURE_SUFFIXES])
    bar = cast(pd.DataFrame, bar.reindex(grid, fill_value=0))
    bar.columns = columns
    return cast(pd.DataFrame, bar.loc[:, columns])


def _prepare_individual(frame: pd.DataFrame, *, day_start_ms: int) -> pd.DataFrame:
    _require_columns(frame, RAW_TRADE_COLUMNS, "individual trades")
    if frame.empty:
        return pd.DataFrame(columns=pd.Index([*RAW_TRADE_COLUMNS, "date"]))

    ids = _int_array(frame, "trade_id")
    _validate_unique_monotone_ids(ids, name="individual trade")
    if len(ids) > 1 and not (np.diff(ids) == 1).all():
        raise ValueError("individual trade IDs must be exactly +1 continuous")

    timestamps = _int_array(frame, "transact_time")
    _validate_day_timestamps(timestamps, day_start_ms=day_start_ms, name="individual trade")
    prices = _numeric_array(frame, "price", positive=True)
    quantities = _numeric_array(frame, "quantity", positive=True)
    quote_qty = _numeric_array(frame, "quote_qty", positive=True)
    reconstructed = prices * quantities
    if (np.abs(quote_qty - reconstructed) > QUOTE_QTY_RECONCILIATION_TOLERANCE).any():
        raise ValueError("quote_qty must reconcile to price * quantity within 0.011 USDT")
    maker = _bool_array(_series(frame, "is_buyer_maker"), "is_buyer_maker")

    work = pd.DataFrame(
        {
            "trade_id": ids,
            "transact_time": timestamps,
            "quote_notional": quote_qty,
            "is_buyer_maker": maker,
        }
    )
    work["date"] = pd.to_datetime(work["transact_time"], unit="ms", utc=True).dt.floor(
        FIVE_MINUTES
    )
    work["side"] = np.where(work["is_buyer_maker"].to_numpy(bool), -1, 1).astype(np.int8)
    work["buy_fill"] = (work["side"].to_numpy(np.int8) == 1).astype(np.int8)
    work["sell_fill"] = (work["side"].to_numpy(np.int8) == -1).astype(np.int8)
    work["buy_quote_notional"] = np.where(work["buy_fill"], work["quote_notional"], 0.0)
    work["sell_quote_notional"] = np.where(work["sell_fill"], work["quote_notional"], 0.0)
    work["signed_quote_notional"] = work["quote_notional"] * work["side"]
    work["quote_notional_squared"] = work["quote_notional"] ** 2
    work["buy_quote_notional_squared"] = work["buy_quote_notional"] ** 2
    work["sell_quote_notional_squared"] = work["sell_quote_notional"] ** 2
    return work


def _prepare_aggregate(frame: pd.DataFrame, *, day_start_ms: int) -> pd.DataFrame:
    _require_columns(frame, RAW_AGGTRADE_COLUMNS, "aggregate trades")
    if frame.empty:
        return pd.DataFrame(columns=pd.Index([*RAW_AGGTRADE_COLUMNS, "date"]))

    ids = _int_array(frame, "agg_trade_id")
    _validate_unique_monotone_ids(ids, name="aggregate trade")
    if len(ids) > 1 and not (np.diff(ids) == 1).all():
        raise ValueError("aggregate trade IDs must be exactly +1 continuous")
    first_ids = _int_array(frame, "first_trade_id")
    last_ids = _int_array(frame, "last_trade_id")
    if (first_ids < 0).any() or (last_ids < first_ids).any():
        raise ValueError("aggregate trade first/last trade IDs must be valid ranges")
    if len(first_ids) > 1 and (first_ids[1:] <= last_ids[:-1]).any():
        raise ValueError(
            "aggregate trade underlying ID ranges must be monotone and non-overlapping"
        )

    timestamps = _int_array(frame, "transact_time")
    _validate_day_timestamps(timestamps, day_start_ms=day_start_ms, name="aggregate trade")
    prices = _numeric_array(frame, "price", positive=True)
    quantities = _numeric_array(frame, "quantity", positive=True)
    maker = _bool_array(_series(frame, "is_buyer_maker"), "is_buyer_maker")

    work = pd.DataFrame(
        {
            "agg_trade_id": ids,
            "first_trade_id": first_ids,
            "last_trade_id": last_ids,
            "transact_time": timestamps,
            "quote_notional": prices * quantities,
            "is_buyer_maker": maker,
        }
    )
    work["date"] = pd.to_datetime(work["transact_time"], unit="ms", utc=True).dt.floor(
        FIVE_MINUTES
    )
    work["side"] = np.where(work["is_buyer_maker"].to_numpy(bool), -1, 1).astype(np.int8)
    work["buy_fill"] = (work["side"].to_numpy(np.int8) == 1).astype(np.int8)
    work["sell_fill"] = (work["side"].to_numpy(np.int8) == -1).astype(np.int8)
    work["buy_quote_notional"] = np.where(work["buy_fill"], work["quote_notional"], 0.0)
    work["sell_quote_notional"] = np.where(work["sell_fill"], work["quote_notional"], 0.0)
    work["signed_quote_notional"] = work["quote_notional"] * work["side"]
    work["quote_notional_squared"] = work["quote_notional"] ** 2
    work["buy_quote_notional_squared"] = work["buy_quote_notional"] ** 2
    work["sell_quote_notional_squared"] = work["sell_quote_notional"] ** 2
    return work


def aggregate_day(
    individual_frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    day: date | datetime | str | pd.Timestamp,
) -> pd.DataFrame:
    """Return a complete UTC five-minute IFDA-72 grid for ``day``.

    Maker semantics follow Binance: ``is_buyer_maker=False`` means buyer
    aggressor (side ``+1``), and ``True`` means seller aggressor (side ``-1``).
    Raw individual ``quote_qty`` is the authoritative notional.  The official
    raw trade archive can round/truncate quote quantities to cents, so
    reconciliation is frozen to an absolute ``<= 0.011`` USDT tolerance.
    """

    day_start = _as_utc_midnight(day)
    day_start_ms = int(day_start.timestamp() * 1_000)
    grid = pd.date_range(day_start, periods=288, freq=FIVE_MINUTES, tz=timezone.utc)

    individual = _prepare_individual(individual_frame, day_start_ms=day_start_ms)
    aggregate = _prepare_aggregate(aggregate_frame, day_start_ms=day_start_ms)

    output = _empty_grid(day_start)
    primary = _feature_frame(
        individual, grid=grid, id_column="trade_id", output_suffixes=_FEATURE_SUFFIXES
    )
    control = _feature_frame(
        aggregate,
        grid=grid,
        id_column="agg_trade_id",
        output_suffixes=_AGG_FEATURE_SUFFIXES,
        prefix="agg_",
    )
    output = pd.concat(
        [output.loc[:, _SOURCE_COLUMNS].set_index("date"), primary, control], axis=1
    ).reset_index(names="date")

    observed_index = pd.Index([])
    if not individual.empty:
        observed_index = observed_index.union(pd.Index(individual["date"]))
    if not aggregate.empty:
        observed_index = observed_index.union(pd.Index(aggregate["date"]))
    observed = _series(output, "date").isin(observed_index.to_numpy())
    output["source_observed"] = observed.to_numpy(dtype=bool)
    output["verified_zero_volume_empty"] = ~output["source_observed"].to_numpy(dtype=bool)
    output["source_complete"] = True
    output["source_gap_day"] = False
    output["post_gap_quarantine"] = False

    output = cast(pd.DataFrame, output.loc[:, OUTPUT_COLUMNS])
    numeric = output.drop(
        columns=[
            "date",
            "source_observed",
            "source_complete",
            "source_gap_day",
            "verified_zero_volume_empty",
            "post_gap_quarantine",
        ]
    )
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("IFDA aggregation produced non-finite values")
    if len(output) != 288:
        raise ValueError("IFDA aggregation did not produce a complete daily grid")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("IFDA grid is duplicate or unordered")
    return output
