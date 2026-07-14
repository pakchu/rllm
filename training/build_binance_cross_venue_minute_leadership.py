"""Build outcome-blind Spot/USD-M minute-order descriptors.

Official monthly Binance one-minute kline archives are checksum-verified and
aligned on exact UTC open times.  Only transitions contained in a completed
five-minute bar are used; the minute-4 to next-bar-minute-0 edge is excluded.
Raw archives are never persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _month_starts,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


SPOT_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
UM_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
INTERVAL = "1m"
SCHEMA_VERSION = 1
SEALED_END_EXCLUSIVE = date(2024, 1, 1)

RAW_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time",
    "quote_notional",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)

_HEADER_ALIASES = {
    "volume": "base_volume",
    "quote_volume": "quote_notional",
    "quote_asset_volume": "quote_notional",
    "count": "trade_count",
    "number_of_trades": "trade_count",
    "taker_buy_volume": "taker_buy_base",
    "taker_buy_base_asset_volume": "taker_buy_base",
    "taker_buy_quote_volume": "taker_buy_quote",
    "taker_buy_quote_asset_volume": "taker_buy_quote",
}

AUDIT_COLUMNS = (
    "spot_rows",
    "um_rows",
    "spot_missing_minutes",
    "um_missing_minutes",
    "spot_invalid_source_minutes",
    "um_invalid_source_minutes",
    "lagged_pair_count",
    "simultaneous_flow_pair_count",
    "simultaneous_return_pair_count",
)

FEATURE_COLUMNS = (
    "spot_quote_notional",
    "um_quote_notional",
    "spot_trade_count",
    "um_trade_count",
    "spot_signed_quote_notional",
    "um_signed_quote_notional",
    "spot_flow_fraction",
    "um_flow_fraction",
    "spot_flow_coherence",
    "um_flow_coherence",
    "spot_log_return_5m",
    "um_log_return_5m",
    "spot_abs_path_return_bp",
    "um_abs_path_return_bp",
    "spot_activity_time_centroid",
    "um_activity_time_centroid",
    "um_minus_spot_activity_time_centroid",
    "spot_flow_time_centroid",
    "um_flow_time_centroid",
    "um_minus_spot_flow_time_centroid",
    "spot_return_time_centroid",
    "um_return_time_centroid",
    "um_minus_spot_return_time_centroid",
    "spot_to_um_lagged_flow_response_bp",
    "um_to_spot_lagged_flow_response_bp",
    "lagged_flow_response_diff_bp",
    "spot_to_um_lagged_directional_alignment",
    "um_to_spot_lagged_directional_alignment",
    "lagged_directional_alignment_diff",
    "flow_transfer_asymmetry",
    "return_leadership_asymmetry",
    "simultaneous_flow_sign_agreement",
    "simultaneous_return_sign_agreement",
    "open_basis_bp",
    "close_basis_bp",
    "basis_change_bp",
    "log_spot_um_quote_ratio",
)

OUTPUT_COLUMNS = (
    "date",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    *AUDIT_COLUMNS,
    *FEATURE_COLUMNS,
    "source_complete",
    "cross_venue_feature_valid",
)


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_venue_minute_leadership_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


def spot_archive_url(symbol: str, month: date) -> str:
    stem = f"{symbol}-{INTERVAL}-{month:%Y-%m}.zip"
    return f"{SPOT_BASE_URL}/{symbol}/{INTERVAL}/{stem}"


def um_archive_url(symbol: str, month: date) -> str:
    stem = f"{symbol}-{INTERVAL}-{month:%Y-%m}.zip"
    return f"{UM_BASE_URL}/{symbol}/{INTERVAL}/{stem}"


def _checksum_url(archive: str) -> str:
    return archive + ".CHECKSUM"


def _canonicalize_header(columns: pd.Index) -> list[str]:
    normalized = [str(column).strip().lower() for column in columns]
    return [_HEADER_ALIASES.get(column, column) for column in normalized]


def _epoch_values_to_ms(values: pd.Series, *, field: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{field} contains non-finite timestamps")
    integer = numeric.astype("int64")
    if not np.equal(numeric.to_numpy(float), integer.to_numpy(float)).all():
        raise ValueError(f"{field} contains non-integer timestamps")
    microsecond = integer.abs().gt(100_000_000_000_000)
    if microsecond.any() and not microsecond.all():
        raise ValueError(f"{field} mixes millisecond and microsecond timestamps")
    if microsecond.all():
        integer = integer.floordiv(1_000)
    return integer


def read_archive(payload: bytes, *, venue: str) -> pd.DataFrame:
    """Read a Spot or USD-M kline ZIP into the shared canonical schema."""
    if venue not in {"spot", "um"}:
        raise ValueError("venue must be 'spot' or 'um'")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in archive, found {members}")
        with archive.open(members[0]) as handle:
            first_line = handle.readline().lstrip(b"\xef\xbb\xbf").lower()
        has_header = first_line.startswith(b"open_time,")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(RAW_COLUMNS),
                low_memory=False,
            )
    if has_header:
        frame.columns = _canonicalize_header(frame.columns)
    if len(set(frame.columns)) != len(frame.columns) or set(frame.columns) != set(RAW_COLUMNS):
        raise ValueError(f"unexpected {venue} kline columns: {frame.columns.tolist()}")
    frame = frame.loc[:, RAW_COLUMNS].copy()
    if frame.empty:
        raise ValueError(f"{venue} kline archive is empty")

    frame["open_time"] = _epoch_values_to_ms(frame["open_time"], field="open_time")
    frame["close_time"] = _epoch_values_to_ms(frame["close_time"], field="close_time")
    for column in RAW_COLUMNS[1:6] + RAW_COLUMNS[7:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.loc[:, RAW_COLUMNS].to_numpy(float)).all():
        raise ValueError(f"{venue} kline archive contains non-finite values")
    if not frame["open_time"].is_unique or not frame["open_time"].is_monotonic_increasing:
        raise ValueError(f"{venue} kline open times are not strictly increasing")

    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0.0).any().any():
        raise ValueError(f"{venue} kline archive contains non-positive prices")
    if (frame[["base_volume", "quote_notional", "trade_count"]] < 0.0).any().any():
        raise ValueError(f"{venue} kline archive contains negative activity")
    tolerance = 1e-8
    if (
        (frame["taker_buy_base"] < -tolerance).any()
        or (frame["taker_buy_quote"] < -tolerance).any()
        or (frame["taker_buy_base"] > frame["base_volume"] + tolerance).any()
        or (frame["taker_buy_quote"] > frame["quote_notional"] + tolerance).any()
    ):
        raise ValueError(f"{venue} kline taker-buy fields violate total-volume bounds")

    envelope_valid = (
        frame["high"].ge(frame[["open", "close", "low"]].max(axis=1))
        & frame["low"].le(frame[["open", "close", "high"]].min(axis=1))
    )
    frame["source_row_valid"] = (
        frame["open_time"].mod(60_000).eq(0)
        & frame["close_time"].sub(frame["open_time"]).eq(59_999)
        & frame["base_volume"].gt(0.0)
        & frame["quote_notional"].gt(0.0)
        & frame["trade_count"].gt(0)
        & envelope_valid
    )
    return frame


def _normalize_expected_minutes(
    spot: pd.DataFrame,
    um: pd.DataFrame,
    expected_minutes: pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if expected_minutes is None:
        observed_ms = pd.concat([spot["open_time"], um["open_time"]], ignore_index=True)
        observed = pd.to_datetime(observed_ms, unit="ms", utc=True).dt.tz_localize(None)
        start = observed.min().floor("5min")
        end = observed.max().floor("5min") + pd.Timedelta("5min")
        expected = pd.date_range(start, end, inclusive="left", freq="1min")
    else:
        expected = pd.DatetimeIndex(expected_minutes)
        if expected.tz is not None:
            expected = expected.tz_convert("UTC").tz_localize(None)
    if expected.empty:
        raise ValueError("expected minute grid is empty")
    canonical = pd.date_range(expected[0], periods=len(expected), freq="1min")
    if not expected.equals(canonical) or expected.has_duplicates:
        raise ValueError("expected minute grid must be unique contiguous UTC minutes")
    if expected[0] != expected[0].floor("5min") or len(expected) % 5 != 0:
        raise ValueError("expected minute grid must contain complete five-minute bins")
    return expected


def _indexed_source(frame: pd.DataFrame, expected: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    source = frame.copy()
    source.index = pd.DatetimeIndex(
        pd.to_datetime(source["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    )
    if not source.index.isin(expected).all():
        raise ValueError(f"{prefix} archive contains timestamps outside expected grid")
    source["present"] = True
    keep = [
        "open",
        "close",
        "quote_notional",
        "trade_count",
        "taker_buy_quote",
        "source_row_valid",
        "present",
    ]
    source = source.loc[:, keep].add_prefix(f"{prefix}_")
    return source.reindex(expected)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).divide(denominator.astype(float).replace(0.0, np.nan))


def _group_sum(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups, sort=True, observed=True).sum(min_count=1)


def _weighted_centroid(
    weights: pd.Series,
    positions: pd.Series,
    groups: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    denominator = _group_sum(weights, groups)
    numerator = _group_sum(weights * positions, groups)
    return _ratio(numerator, denominator).divide(4.0), denominator


def aggregate_cross_venue_five_minute(
    spot: pd.DataFrame,
    um: pd.DataFrame,
    *,
    expected_minutes: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Aggregate exact minute-order descriptors without crossing bar boundaries."""
    expected = _normalize_expected_minutes(spot, um, expected_minutes)
    spot_source = _indexed_source(spot, expected, "spot")
    um_source = _indexed_source(um, expected, "um")
    work = pd.concat([spot_source, um_source], axis=1)
    work.index.name = "minute_open_time"
    work["date"] = work.index.floor("5min")
    work["minute_position"] = (
        (work.index.to_series(index=work.index) - work["date"])
        .dt.total_seconds()
        .astype(int)
        // 60
    ).astype(float)

    for venue in ("spot", "um"):
        quote = work[f"{venue}_quote_notional"]
        signed_quote = 2.0 * work[f"{venue}_taker_buy_quote"] - quote
        work[f"{venue}_signed_quote"] = signed_quote
        work[f"{venue}_flow_fraction"] = _ratio(signed_quote, quote)
        prior_close = work.groupby("date", sort=False, observed=True)[
            f"{venue}_close"
        ].shift()
        anchor = prior_close.where(work["minute_position"].ne(0.0), work[f"{venue}_open"])
        work[f"{venue}_path_return"] = np.log(work[f"{venue}_close"] / anchor)

    inside_transition = work["minute_position"].lt(4.0)
    work["next_spot_return"] = work.groupby("date", sort=False, observed=True)[
        "spot_path_return"
    ].shift(-1).where(inside_transition)
    work["next_um_return"] = work.groupby("date", sort=False, observed=True)[
        "um_path_return"
    ].shift(-1).where(inside_transition)
    groups = work["date"]
    grouped = work.groupby("date", sort=True, observed=True)
    output = pd.DataFrame(index=pd.DatetimeIndex(sorted(groups.unique()), name="date"))

    for venue in ("spot", "um"):
        present = work[f"{venue}_present"].eq(True)
        valid = work[f"{venue}_source_row_valid"].eq(True)
        output[f"{venue}_rows"] = present.groupby(groups, observed=True).sum().astype(int)
        output[f"{venue}_missing_minutes"] = 5 - output[f"{venue}_rows"]
        output[f"{venue}_invalid_source_minutes"] = (
            (present & ~valid).groupby(groups, observed=True).sum().astype(int)
        )
        quote = work[f"{venue}_quote_notional"]
        signed_quote = work[f"{venue}_signed_quote"]
        flow_fraction = work[f"{venue}_flow_fraction"]
        path_return = work[f"{venue}_path_return"]
        output[f"{venue}_quote_notional"] = _group_sum(quote, groups)
        output[f"{venue}_trade_count"] = _group_sum(
            work[f"{venue}_trade_count"], groups
        )
        output[f"{venue}_signed_quote_notional"] = _group_sum(signed_quote, groups)
        output[f"{venue}_flow_fraction"] = _ratio(
            output[f"{venue}_signed_quote_notional"],
            output[f"{venue}_quote_notional"],
        )
        output[f"{venue}_flow_coherence"] = _ratio(
            _group_sum(signed_quote.abs(), groups), output[f"{venue}_quote_notional"]
        )
        first_open = grouped[f"{venue}_open"].first()
        last_close = grouped[f"{venue}_close"].last()
        output[f"{venue}_log_return_5m"] = np.log(last_close / first_open)
        output[f"{venue}_abs_path_return_bp"] = (
            _group_sum(path_return.abs(), groups) * 10_000.0
        )
        output[f"{venue}_activity_time_centroid"], activity_support = _weighted_centroid(
            quote, work["minute_position"], groups
        )
        output[f"{venue}_flow_time_centroid"], flow_support = _weighted_centroid(
            flow_fraction.abs(), work["minute_position"], groups
        )
        output[f"{venue}_return_time_centroid"], return_support = _weighted_centroid(
            path_return.abs(), work["minute_position"], groups
        )
        output[f"_{venue}_activity_support"] = activity_support
        output[f"_{venue}_flow_support"] = flow_support
        output[f"_{venue}_return_support"] = return_support

    output["um_minus_spot_activity_time_centroid"] = (
        output["um_activity_time_centroid"] - output["spot_activity_time_centroid"]
    )
    output["um_minus_spot_flow_time_centroid"] = (
        output["um_flow_time_centroid"] - output["spot_flow_time_centroid"]
    )
    output["um_minus_spot_return_time_centroid"] = (
        output["um_return_time_centroid"] - output["spot_return_time_centroid"]
    )

    lag_mask = (
        inside_transition
        & work["spot_flow_fraction"].notna()
        & work["um_flow_fraction"].notna()
        & work["next_spot_return"].notna()
        & work["next_um_return"].notna()
    )
    output["lagged_pair_count"] = lag_mask.groupby(groups, observed=True).sum().astype(int)
    spot_to_um = (work["spot_flow_fraction"] * work["next_um_return"]).where(lag_mask)
    um_to_spot = (work["um_flow_fraction"] * work["next_spot_return"]).where(lag_mask)
    spot_flow_support = _group_sum(
        work["spot_flow_fraction"].abs().where(lag_mask), groups
    )
    um_flow_support = _group_sum(work["um_flow_fraction"].abs().where(lag_mask), groups)
    output["spot_to_um_lagged_flow_response_bp"] = (
        _ratio(_group_sum(spot_to_um, groups), spot_flow_support) * 10_000.0
    )
    output["um_to_spot_lagged_flow_response_bp"] = (
        _ratio(_group_sum(um_to_spot, groups), um_flow_support) * 10_000.0
    )
    output["lagged_flow_response_diff_bp"] = (
        output["spot_to_um_lagged_flow_response_bp"]
        - output["um_to_spot_lagged_flow_response_bp"]
    )

    spot_direction_mask = lag_mask & work["spot_flow_fraction"].ne(0.0)
    um_direction_mask = lag_mask & work["um_flow_fraction"].ne(0.0)
    spot_direction_numerator = _group_sum(
        (np.sign(work["spot_flow_fraction"]) * work["next_um_return"]).where(
            spot_direction_mask
        ),
        groups,
    )
    um_direction_numerator = _group_sum(
        (np.sign(work["um_flow_fraction"]) * work["next_spot_return"]).where(
            um_direction_mask
        ),
        groups,
    )
    spot_direction_denominator = _group_sum(
        work["next_um_return"].abs().where(spot_direction_mask), groups
    )
    um_direction_denominator = _group_sum(
        work["next_spot_return"].abs().where(um_direction_mask), groups
    )
    output["spot_to_um_lagged_directional_alignment"] = _ratio(
        spot_direction_numerator, spot_direction_denominator
    )
    output["um_to_spot_lagged_directional_alignment"] = _ratio(
        um_direction_numerator, um_direction_denominator
    )
    output["lagged_directional_alignment_diff"] = 0.5 * (
        output["spot_to_um_lagged_directional_alignment"]
        - output["um_to_spot_lagged_directional_alignment"]
    )

    flow_transfer_denominator = _group_sum(
        spot_to_um.abs() + um_to_spot.abs(), groups
    )
    output["flow_transfer_asymmetry"] = _ratio(
        _group_sum(spot_to_um - um_to_spot, groups), flow_transfer_denominator
    )
    spot_return_lead = (work["spot_path_return"] * work["next_um_return"]).where(
        lag_mask
    )
    um_return_lead = (work["um_path_return"] * work["next_spot_return"]).where(lag_mask)
    return_lead_denominator = _group_sum(
        spot_return_lead.abs() + um_return_lead.abs(), groups
    )
    output["return_leadership_asymmetry"] = _ratio(
        _group_sum(spot_return_lead - um_return_lead, groups),
        return_lead_denominator,
    )

    simultaneous_flow_mask = (
        work["spot_flow_fraction"].notna()
        & work["um_flow_fraction"].notna()
        & work["spot_flow_fraction"].ne(0.0)
        & work["um_flow_fraction"].ne(0.0)
    )
    simultaneous_return_mask = (
        work["spot_path_return"].notna()
        & work["um_path_return"].notna()
        & work["spot_path_return"].ne(0.0)
        & work["um_path_return"].ne(0.0)
    )
    output["simultaneous_flow_pair_count"] = (
        simultaneous_flow_mask.groupby(groups, observed=True).sum().astype(int)
    )
    output["simultaneous_return_pair_count"] = (
        simultaneous_return_mask.groupby(groups, observed=True).sum().astype(int)
    )
    output["simultaneous_flow_sign_agreement"] = (
        (np.sign(work["spot_flow_fraction"]) * np.sign(work["um_flow_fraction"]))
        .where(simultaneous_flow_mask)
        .groupby(groups, observed=True)
        .mean()
    )
    output["simultaneous_return_sign_agreement"] = (
        (np.sign(work["spot_path_return"]) * np.sign(work["um_path_return"]))
        .where(simultaneous_return_mask)
        .groupby(groups, observed=True)
        .mean()
    )

    spot_open = grouped["spot_open"].first()
    spot_close = grouped["spot_close"].last()
    um_open = grouped["um_open"].first()
    um_close = grouped["um_close"].last()
    output["open_basis_bp"] = np.log(um_open / spot_open) * 10_000.0
    output["close_basis_bp"] = np.log(um_close / spot_close) * 10_000.0
    output["basis_change_bp"] = output["close_basis_bp"] - output["open_basis_bp"]
    output["log_spot_um_quote_ratio"] = np.log(
        output["spot_quote_notional"] / output["um_quote_notional"]
    )

    output["source_complete"] = (
        output["spot_rows"].eq(5)
        & output["um_rows"].eq(5)
        & output["spot_missing_minutes"].eq(0)
        & output["um_missing_minutes"].eq(0)
        & output["spot_invalid_source_minutes"].eq(0)
        & output["um_invalid_source_minutes"].eq(0)
    )
    denominator_valid = (
        output["_spot_activity_support"].gt(0.0)
        & output["_um_activity_support"].gt(0.0)
        & output["_spot_flow_support"].gt(0.0)
        & output["_um_flow_support"].gt(0.0)
        & output["_spot_return_support"].gt(0.0)
        & output["_um_return_support"].gt(0.0)
        & spot_flow_support.gt(0.0)
        & um_flow_support.gt(0.0)
        & spot_direction_denominator.gt(0.0)
        & um_direction_denominator.gt(0.0)
        & flow_transfer_denominator.gt(0.0)
        & return_lead_denominator.gt(0.0)
        & output["simultaneous_flow_pair_count"].gt(0)
        & output["simultaneous_return_pair_count"].gt(0)
        & output["lagged_pair_count"].eq(4)
    )
    finite_features = pd.Series(
        np.isfinite(output.loc[:, FEATURE_COLUMNS].to_numpy(float)).all(axis=1),
        index=output.index,
    )
    output["cross_venue_feature_valid"] = (
        output["source_complete"] & denominator_valid & finite_features
    )

    invalid = ~output["cross_venue_feature_valid"]
    output.loc[invalid, FEATURE_COLUMNS] = np.nan
    output["feature_available_time_utc"] = output.index + pd.Timedelta("5min")
    output["trade_earliest_time_utc"] = output["feature_available_time_utc"]
    output = output.drop(
        columns=[column for column in output.columns if column.startswith("_")]
    ).reset_index()
    output = output.loc[:, OUTPUT_COLUMNS]
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("cross-venue output timestamps are duplicate or unordered")
    accepted = output.loc[output["cross_venue_feature_valid"], FEATURE_COLUMNS]
    if not np.isfinite(accepted.to_numpy(float)).all():
        raise ValueError("accepted cross-venue rows contain non-finite descriptors")
    return output


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
    ):
        return False
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")
    spot_hash = expected_sha256(
        fetcher(
            _checksum_url(spot_archive_url(cfg.symbol, month)),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    )
    um_hash = expected_sha256(
        fetcher(
            _checksum_url(um_archive_url(cfg.symbol, month)),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    )
    return (
        spot_hash == metadata.get("spot_archive_sha256")
        and um_hash == metadata.get("um_archive_sha256")
    )


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    monthly_dir = Path(cfg.output_dir) / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_cross_venue_minute_leadership_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    source_payloads: dict[str, bytes] = {}
    source_hashes: dict[str, str] = {}
    for venue, archive in (
        ("spot", spot_archive_url(cfg.symbol, month)),
        ("um", um_archive_url(cfg.symbol, month)),
    ):
        checksum = expected_sha256(
            fetcher(
                _checksum_url(archive),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(
            archive,
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
        source_hashes[venue] = verify_sha256(payload, checksum)
        source_payloads[venue] = payload

    spot = read_archive(source_payloads.pop("spot"), venue="spot")
    um = read_archive(source_payloads.pop("um"), venue="um")
    month_start = pd.Timestamp(month)
    next_month = month_start + pd.offsets.MonthBegin(1)
    expected_minutes = pd.date_range(
        month_start, next_month, inclusive="left", freq="1min"
    )
    output = aggregate_cross_venue_five_minute(
        spot, um, expected_minutes=expected_minutes
    )
    expected_bars = pd.date_range(month_start, next_month, inclusive="left", freq="5min")
    if not output["date"].equals(pd.Series(expected_bars, name="date")):
        raise ValueError(f"cross-venue month {month:%Y-%m} has an invalid output grid")

    _write_gzip_csv(output, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "spot_archive_sha256": source_hashes["spot"],
        "um_archive_sha256": source_hashes["um"],
        "spot_raw_rows": int(len(spot)),
        "um_raw_rows": int(len(um)),
        "rows": int(len(output)),
        "source_complete_rows": int(output["source_complete"].sum()),
        "feature_valid_rows": int(output["cross_venue_feature_valid"].sum()),
        "first_date": str(output["date"].min()),
        "last_date": str(output["date"].max()),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def _coverage_by_year(
    frame: pd.DataFrame,
    expected: pd.DatetimeIndex,
) -> dict[str, dict[str, int]]:
    source_complete = frame["source_complete"].astype(bool)
    feature_valid = frame["cross_venue_feature_valid"].astype(bool)
    coverage: dict[str, dict[str, int]] = {}
    for year in sorted(set(expected.year)):
        expected_rows = int((expected.year == year).sum())
        mask = frame["date"].dt.year.eq(year)
        coverage[str(year)] = {
            "expected_rows": expected_rows,
            "source_complete_rows": int((mask & source_complete).sum()),
            "feature_valid_rows": int((mask & feature_valid).sum()),
            "quarantined_rows": int((mask & ~feature_valid).sum()),
        }
    return coverage


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start.day != 1 or end.day != 1:
        raise ValueError("cross-venue build boundaries must be month starts")
    if end > SEALED_END_EXCLUSIVE:
        raise ValueError("2024+ source data is sealed until pre-2024 validation passes")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")

    months = _month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(futures):
            month = futures[future]
            result = future.result()
            metadata.append(result)
            print(
                f"completed {month:%Y-%m}: rows={result['rows']} "
                f"valid={result['feature_valid_rows']}",
                flush=True,
            )
    metadata.sort(key=lambda item: item["month"])
    frames = [
        pd.read_csv(
            item["output"],
            compression="gzip",
            parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
        )
        for item in metadata
    ]
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    expected = pd.date_range(start, end, inclusive="left", freq="5min")
    if not combined["date"].equals(pd.Series(expected, name="date")):
        raise ValueError("combined cross-venue output has an invalid timestamp grid")
    if not combined["feature_available_time_utc"].equals(
        combined["date"] + pd.Timedelta("5min")
    ):
        raise ValueError("feature availability timestamp is inconsistent")
    if not combined["trade_earliest_time_utc"].equals(
        combined["feature_available_time_utc"]
    ):
        raise ValueError("earliest trade timestamp is inconsistent")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_month = date(
        end.year - (end.month == 1),
        12 if end.month == 1 else end.month - 1,
        1,
    )
    combined_path = output_dir / (
        f"{cfg.symbol}_cross_venue_minute_leadership_5m_"
        f"{start:%Y-%m}_{last_month:%Y-%m}.csv.gz"
    )
    _write_gzip_csv(combined, combined_path)
    source_complete = combined["source_complete"].astype(bool)
    feature_valid = combined["cross_venue_feature_valid"].astype(bool)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "sources": [
                "official Binance Spot monthly one-minute kline archives",
                "official Binance USD-M monthly one-minute kline archives",
            ],
            "archive_roots": {"spot": SPOT_BASE_URL, "um": UM_BASE_URL},
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "sealed_end_exclusive": SEALED_END_EXCLUSIVE.isoformat(),
            "join_key": "exact UTC one-minute open_time",
            "lag_edges": "only minute 0->1, 1->2, 2->3, 3->4 within each completed bar",
            "feature_available_time": "five-minute bar open time plus five minutes",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "source_complete_rows": int(source_complete.sum()),
        "feature_valid_rows": int(feature_valid.sum()),
        "quarantined_rows": int((~feature_valid).sum()),
        "coverage_by_year": _coverage_by_year(combined, expected),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=BuildConfig.symbol)
    parser.add_argument("--start", default=BuildConfig.start)
    parser.add_argument("--end", default=BuildConfig.end)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--workers", type=int, default=BuildConfig.workers)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--overwrite", action="store_true")
    manifest = build(BuildConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "combined_output",
                    "rows",
                    "source_complete_rows",
                    "feature_valid_rows",
                    "first_date",
                    "last_date",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
