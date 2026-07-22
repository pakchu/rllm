"""Outcome-blind BTC quantity-lattice cohort features from Binance aggTrades."""
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
    "agg_trade_count",
    "total_quantity_mbtc",
    "total_signed_quantity_mbtc",
    "coarse_event_count",
    "coarse_quantity_mbtc",
    "coarse_signed_quantity_mbtc",
    "medium_event_count",
    "medium_quantity_mbtc",
    "medium_signed_quantity_mbtc",
    "fine_event_count",
    "fine_quantity_mbtc",
    "fine_signed_quantity_mbtc",
    "coarse_quantity_share",
    "coarse_coherence",
    "fine_signed_share",
    "coarse_side",
    "cohort_opposition",
    "qlcd_score",
)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        numerator.astype(float)
        .divide(denominator.astype(float).replace(0.0, np.nan))
        .fillna(0.0)
    )


def quantity_mbtc(quantity: pd.Series) -> np.ndarray:
    """Convert exact 0.001-BTC contract quantities to integer milli-BTC."""
    numeric = cast(pd.Series, pd.to_numeric(quantity, errors="raise"))
    values = numeric.astype(float).to_numpy()
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("aggTrade quantity must be finite and positive")
    scaled = values * 1_000.0
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-9, rtol=0.0):
        raise ValueError("aggTrade quantity is not on the frozen 0.001-BTC lattice")
    if np.any(rounded > np.iinfo(np.int64).max):
        raise ValueError("aggTrade quantity exceeds int64 milli-BTC range")
    return rounded.astype(np.int64)


def aggregate_quantity_lattice_five_minute(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coarse, medium, and fine BTC-size cohorts by completed 5m bar.

    Cohorts are mutually exclusive on the exact milli-BTC integer lattice:

    - coarse: multiples of 100 mBTC (0.1 BTC),
    - medium: multiples of 10 mBTC but not 100 mBTC,
    - fine: all remaining 1 mBTC increments.

    QLCD measures disagreement between coarse aggressive quantity and the fine
    cohort.  Medium events are retained as a frozen diagnostic/control cohort
    but do not enter the primary score.
    """
    if frame.empty:
        raise ValueError("aggTrades frame is empty")
    required = {
        "agg_trade_id",
        "quantity",
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
    work["quantity_mbtc"] = quantity_mbtc(_series(work, "quantity"))
    maker = _series(work, "is_buyer_maker")
    if pd.api.types.is_bool_dtype(maker.dtype):
        maker_flag = maker.astype(bool).to_numpy()
    else:
        maker_text = maker.astype("string").str.strip().str.lower()
        if not maker_text.isin(["true", "false"]).all():
            raise ValueError("is_buyer_maker contains an unknown value")
        maker_flag = maker_text.eq("true").to_numpy()
    work["side"] = np.where(maker_flag, -1, 1).astype(np.int8)
    work["signed_quantity_mbtc"] = (
        work["quantity_mbtc"].to_numpy(np.int64)
        * work["side"].to_numpy(np.int64)
    )
    coarse = work["quantity_mbtc"].to_numpy(np.int64) % 100 == 0
    medium = (~coarse) & (work["quantity_mbtc"].to_numpy(np.int64) % 10 == 0)
    fine = ~(coarse | medium)
    for name, mask in (("coarse", coarse), ("medium", medium), ("fine", fine)):
        work[f"{name}_event"] = mask.astype(np.int8)
        work[f"{name}_quantity_mbtc"] = np.where(
            mask, work["quantity_mbtc"].to_numpy(np.int64), 0
        )
        work[f"{name}_signed_quantity_mbtc"] = np.where(
            mask, work["signed_quantity_mbtc"].to_numpy(np.int64), 0
        )

    grouped = work.groupby("date", sort=True, observed=True)
    bar = cast(
        pd.DataFrame,
        grouped.agg(
            agg_trade_count=("agg_trade_id", "size"),
            total_quantity_mbtc=("quantity_mbtc", "sum"),
            total_signed_quantity_mbtc=("signed_quantity_mbtc", "sum"),
            coarse_event_count=("coarse_event", "sum"),
            coarse_quantity_mbtc=("coarse_quantity_mbtc", "sum"),
            coarse_signed_quantity_mbtc=("coarse_signed_quantity_mbtc", "sum"),
            medium_event_count=("medium_event", "sum"),
            medium_quantity_mbtc=("medium_quantity_mbtc", "sum"),
            medium_signed_quantity_mbtc=("medium_signed_quantity_mbtc", "sum"),
            fine_event_count=("fine_event", "sum"),
            fine_quantity_mbtc=("fine_quantity_mbtc", "sum"),
            fine_signed_quantity_mbtc=("fine_signed_quantity_mbtc", "sum"),
        ),
    )
    bar["coarse_quantity_share"] = _safe_ratio(
        _series(bar, "coarse_quantity_mbtc"), _series(bar, "total_quantity_mbtc")
    )
    bar["coarse_coherence"] = _safe_ratio(
        _series(bar, "coarse_signed_quantity_mbtc").abs(),
        _series(bar, "coarse_quantity_mbtc"),
    )
    bar["fine_signed_share"] = _safe_ratio(
        _series(bar, "fine_signed_quantity_mbtc"),
        _series(bar, "fine_quantity_mbtc"),
    )
    bar["coarse_side"] = np.sign(
        _series(bar, "coarse_signed_quantity_mbtc").to_numpy(np.int64)
    ).astype(np.int8)
    bar["cohort_opposition"] = np.clip(
        -_series(bar, "coarse_side").to_numpy(float)
        * _series(bar, "fine_signed_share").to_numpy(float),
        0.0,
        1.0,
    )
    bar["qlcd_score"] = (
        _series(bar, "coarse_quantity_share")
        * _series(bar, "coarse_coherence")
        * _series(bar, "cohort_opposition")
    )

    output = cast(pd.DataFrame, bar.reset_index())
    output["source_observed"] = True
    output["source_complete"] = True
    output["source_gap_day"] = False
    output["verified_zero_volume_empty"] = False
    output["post_gap_quarantine"] = False
    output = cast(pd.DataFrame, output.loc[:, BAR_COLUMNS])
    if not np.isfinite(output.drop(columns="date").to_numpy(float)).all():
        raise ValueError("quantity-lattice aggregation produced non-finite values")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("quantity-lattice bars are duplicate or unordered")
    return output
