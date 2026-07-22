"""Outcome-blind same-millisecond cascade features from Binance aggTrades."""
from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd


BAR_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "source_observed",
    "source_complete",
    "source_gap_day",
    "verified_zero_volume_empty",
    "post_gap_quarantine",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "agg_trade_count",
    "underlying_trade_count",
    "millisecond_group_count",
    "collision_group_count",
    "first_price",
    "last_price",
    "quote_notional",
    "signed_quote_notional",
    "collision_quote_notional",
    "collision_notional_share",
    "max_ms_transact_time",
    "max_ms_event_count",
    "max_ms_underlying_trade_count",
    "max_ms_quote_notional",
    "max_ms_signed_quote_notional",
    "max_ms_notional_share",
    "max_ms_coherence",
    "max_ms_side",
    "max_ms_pre_group_price",
    "max_ms_first_price",
    "max_ms_last_price",
    "max_ms_sweep_bp",
    "max_ms_score",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).divide(denominator.astype(float).replace(0.0, np.nan)).fillna(0.0)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def aggregate_same_millisecond_five_minute(frame: pd.DataFrame) -> pd.DataFrame:
    """Select each completed bar's largest exact-millisecond notional group.

    The selected group is the largest by quote notional; ties resolve to the
    earliest transaction millisecond.  Its pre-group price is the preceding
    exact-millisecond group's last price inside the same completed five-minute
    bar.  The first group of a bar has no causal within-bar predecessor and is
    assigned zero sweep/score rather than borrowing state across a boundary.
    """
    if frame.empty:
        raise ValueError("aggTrades frame is empty")
    required = {
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"aggTrades frame is missing columns: {sorted(missing)}")

    work = frame.copy()
    work["date"] = (
        pd.to_datetime(work["transact_time"], unit="ms", utc=True, errors="raise")
        .dt.floor("5min")
        .dt.tz_localize(None)
    )
    work["side"] = np.where(work["is_buyer_maker"].to_numpy(bool), -1.0, 1.0)
    work["event_notional"] = work["price"].to_numpy(float) * work["quantity"].to_numpy(float)
    work["signed_notional"] = work["event_notional"] * work["side"]
    work["underlying_count"] = (
        work["last_trade_id"].to_numpy(np.int64)
        - work["first_trade_id"].to_numpy(np.int64)
        + 1
    )

    millisecond = cast(
        pd.DataFrame,
        work.groupby(["date", "transact_time"], sort=True, observed=True).agg(
            event_count=("agg_trade_id", "size"),
            underlying_trade_count=("underlying_count", "sum"),
            quote_notional=("event_notional", "sum"),
            signed_quote_notional=("signed_notional", "sum"),
            first_price=("price", "first"),
            last_price=("price", "last"),
        ),
    )
    millisecond["pre_group_price"] = millisecond.groupby(level="date", sort=False)[
        "last_price"
    ].shift()
    millisecond["coherence"] = _safe_ratio(
        _series(millisecond, "signed_quote_notional").abs(),
        _series(millisecond, "quote_notional"),
    )
    millisecond["side"] = np.sign(_series(millisecond, "signed_quote_notional")).astype(np.int8)
    valid_predecessor = _series(millisecond, "pre_group_price").notna()
    raw_sweep = (
        10_000.0
        * _series(millisecond, "side")
        * np.log(
            _series(millisecond, "last_price")
            / _series(millisecond, "pre_group_price")
        )
    )
    millisecond["sweep_bp"] = raw_sweep.where(valid_predecessor, 0.0).fillna(0.0)

    bar = cast(
        pd.DataFrame,
        work.groupby("date", sort=True, observed=True).agg(
            first_transact_time_ms=("transact_time", "first"),
            last_transact_time_ms=("transact_time", "last"),
            agg_trade_count=("agg_trade_id", "size"),
            underlying_trade_count=("underlying_count", "sum"),
            first_price=("price", "first"),
            last_price=("price", "last"),
            quote_notional=("event_notional", "sum"),
            signed_quote_notional=("signed_notional", "sum"),
        ),
    )
    by_bar = millisecond.groupby(level="date", sort=True)
    bar["millisecond_group_count"] = by_bar.size()
    collision_mask = _series(millisecond, "event_count").ge(2)
    bar["collision_group_count"] = collision_mask.groupby(level="date").sum()
    bar["collision_quote_notional"] = (
        _series(millisecond, "quote_notional")
        .where(collision_mask, 0.0)
        .groupby(level="date")
        .sum()
    )
    bar["collision_notional_share"] = _safe_ratio(
        _series(bar, "collision_quote_notional"), _series(bar, "quote_notional")
    )

    ranked = millisecond.reset_index().sort_values(
        ["date", "quote_notional", "transact_time"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = cast(
        pd.DataFrame,
        ranked.drop_duplicates("date", keep="first").set_index("date"),
    )
    bar["max_ms_transact_time"] = selected["transact_time"]
    bar["max_ms_event_count"] = selected["event_count"]
    bar["max_ms_underlying_trade_count"] = selected["underlying_trade_count"]
    bar["max_ms_quote_notional"] = selected["quote_notional"]
    bar["max_ms_signed_quote_notional"] = selected["signed_quote_notional"]
    bar["max_ms_notional_share"] = _safe_ratio(
        _series(selected, "quote_notional"), _series(bar, "quote_notional")
    )
    bar["max_ms_coherence"] = selected["coherence"]
    bar["max_ms_side"] = selected["side"]
    bar["max_ms_pre_group_price"] = selected["pre_group_price"].fillna(selected["first_price"])
    bar["max_ms_first_price"] = selected["first_price"]
    bar["max_ms_last_price"] = selected["last_price"]
    bar["max_ms_sweep_bp"] = selected["sweep_bp"]
    bar["max_ms_score"] = (
        bar["max_ms_notional_share"]
        * bar["max_ms_coherence"]
        * _series(bar, "max_ms_sweep_bp").clip(lower=0.0)
    )

    output = cast(pd.DataFrame, bar.reset_index())
    output["source_observed"] = True
    output["source_complete"] = True
    output["source_gap_day"] = False
    output["verified_zero_volume_empty"] = False
    output["post_gap_quarantine"] = False
    output = cast(pd.DataFrame, output.loc[:, BAR_COLUMNS])
    numeric = cast(pd.DataFrame, output.drop(columns="date"))
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("same-millisecond aggregation produced non-finite values")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("same-millisecond bars are duplicate or unordered")
    return output
