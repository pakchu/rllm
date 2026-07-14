"""Build causal five-minute BTC microstructure features from Binance aggTrades.

The builder streams official daily USD-M Futures archives, verifies each
published checksum, aggregates trade-level order-flow structure, writes monthly
resume points, and finally emits one compact combined frame.  Raw archives are
kept in memory only so the build cannot push WSL disk usage above the project
budget.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
SCHEMA_VERSION = 1
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 6, "mtime": 0}
RAW_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)
OUTPUT_COLUMNS = (
    "date",
    "first_transact_time_ms",
    "last_transact_time_ms",
    "agg_trade_count",
    "underlying_trade_count",
    "base_volume",
    "quote_notional",
    "buy_quote_notional",
    "sell_quote_notional",
    "signed_quote_notional",
    "flow_coherence",
    "first_price",
    "last_price",
    "micro_log_return",
    "signed_price_response",
    "event_notional_mean",
    "event_notional_std",
    "event_notional_p50",
    "event_notional_p90",
    "event_notional_p99",
    "event_notional_max",
    "event_notional_hhi",
    "normalized_effective_event_count",
    "underlying_trades_per_agg_event",
    "signed_event_imbalance",
    "sign_flip_rate",
    "mean_same_sign_run_length",
    "max_same_sign_run_share",
    "interarrival_mean_ms",
    "interarrival_std_ms",
    "interarrival_burstiness",
    "buy_sell_event_size_log_ratio",
)


@dataclass(frozen=True)
class BuildConfig:
    symbol: str = "BTCUSDT"
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_um_aggtrade_microstructure_btc_2020_2023"
    workers: int = 4
    retries: int = 5
    timeout_seconds: int = 60
    overwrite: bool = False


def archive_url(symbol: str, day: date) -> str:
    stamp = day.isoformat()
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{stamp}.zip"


def checksum_url(symbol: str, day: date) -> str:
    return archive_url(symbol, day) + ".CHECKSUM"


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "rllm-aggtrade-builder/1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def expected_sha256(checksum_payload: bytes) -> str:
    fields = checksum_payload.decode("utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("invalid Binance checksum payload")
    int(fields[0], 16)
    return fields[0].lower()


def verify_sha256(payload: bytes, expected: str) -> str:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"archive checksum mismatch: expected {expected}, got {actual}")
    return actual


def read_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in archive, found {members}")
        with archive.open(members[0]) as handle:
            first_line = handle.readline()
        has_header = first_line.lower().startswith(b"agg_trade_id,")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(
                handle,
                header=0 if has_header else None,
                names=None if has_header else list(RAW_COLUMNS),
                dtype={
                    "agg_trade_id": "int64",
                    "price": "float64",
                    "quantity": "float64",
                    "first_trade_id": "int64",
                    "last_trade_id": "int64",
                    "transact_time": "int64",
                    "is_buyer_maker": "string",
                },
                low_memory=False,
            )
    if has_header:
        frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected aggTrades columns: {frame.columns.tolist()}")
    maker_text = frame["is_buyer_maker"].astype("string").str.lower()
    if not maker_text.isin(("true", "false")).all():
        raise ValueError("is_buyer_maker contains an unknown value")
    frame["is_buyer_maker"] = maker_text.eq("true")
    if frame.empty:
        raise ValueError("aggTrades archive is empty")
    if (
        not frame["agg_trade_id"].is_monotonic_increasing
        or not frame["agg_trade_id"].is_unique
    ):
        raise ValueError("aggregate trade ids are not strictly increasing")
    if not frame["transact_time"].is_monotonic_increasing:
        raise ValueError("aggregate trade timestamps are not monotonic")
    if not np.isfinite(frame[["price", "quantity"]].to_numpy(float)).all():
        raise ValueError("non-finite price or quantity in aggTrades")
    if (frame["price"] <= 0.0).any() or (frame["quantity"] <= 0.0).any():
        raise ValueError("non-positive price or quantity in aggTrades")
    if (frame["last_trade_id"] < frame["first_trade_id"]).any():
        raise ValueError("invalid underlying trade-id interval")
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).divide(denominator.astype(float).replace(0.0, np.nan))


def aggregate_five_minute(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = (
        pd.to_datetime(work["transact_time"], unit="ms", utc=True, errors="raise")
        .dt.floor("5min")
        .dt.tz_localize(None)
    )
    work["event_notional"] = work["price"] * work["quantity"]
    work["side"] = np.where(work["is_buyer_maker"].to_numpy(bool), -1.0, 1.0)
    work["signed_notional"] = work["event_notional"] * work["side"]
    work["buy_notional"] = np.where(work["side"] > 0.0, work["event_notional"], 0.0)
    work["sell_notional"] = np.where(work["side"] < 0.0, work["event_notional"], 0.0)
    work["underlying_count"] = work["last_trade_id"] - work["first_trade_id"] + 1
    work["notional_squared"] = work["event_notional"] ** 2

    grouped = work.groupby("date", sort=True, observed=True)
    output = grouped.agg(
        first_transact_time_ms=("transact_time", "first"),
        last_transact_time_ms=("transact_time", "last"),
        agg_trade_count=("agg_trade_id", "size"),
        underlying_trade_count=("underlying_count", "sum"),
        base_volume=("quantity", "sum"),
        quote_notional=("event_notional", "sum"),
        buy_quote_notional=("buy_notional", "sum"),
        sell_quote_notional=("sell_notional", "sum"),
        signed_quote_notional=("signed_notional", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        event_notional_mean=("event_notional", "mean"),
        event_notional_max=("event_notional", "max"),
        notional_squared_sum=("notional_squared", "sum"),
        signed_event_sum=("side", "sum"),
    )
    second_moment = _safe_divide(
        grouped["notional_squared"].sum(), output["agg_trade_count"]
    )
    output["event_notional_std"] = np.sqrt(
        np.maximum(second_moment - output["event_notional_mean"] ** 2, 0.0)
    )
    for quantile, name in ((0.50, "event_notional_p50"), (0.90, "event_notional_p90"), (0.99, "event_notional_p99")):
        output[name] = grouped["event_notional"].quantile(quantile)
    output["event_notional_hhi"] = _safe_divide(
        output.pop("notional_squared_sum"), output["quote_notional"] ** 2
    )
    output["normalized_effective_event_count"] = _safe_divide(
        1.0 / output["event_notional_hhi"], output["agg_trade_count"]
    )
    output["underlying_trades_per_agg_event"] = _safe_divide(
        output["underlying_trade_count"], output["agg_trade_count"]
    )
    output["flow_coherence"] = _safe_divide(
        output["signed_quote_notional"].abs(), output["quote_notional"]
    )
    output["signed_event_imbalance"] = _safe_divide(
        output.pop("signed_event_sum"), output["agg_trade_count"]
    )
    output["micro_log_return"] = np.log(output["last_price"] / output["first_price"])
    output["signed_price_response"] = np.sign(output["signed_quote_notional"]) * output["micro_log_return"]

    group_change = work["date"].ne(work["date"].shift())
    sign_change = work["side"].ne(work["side"].shift())
    flip = (~group_change & sign_change).astype(np.int8)
    flip_count = flip.groupby(work["date"], observed=True).sum()
    output["sign_flip_rate"] = _safe_divide(
        flip_count, (output["agg_trade_count"] - 1).clip(lower=1)
    )
    run_id = (group_change | sign_change).cumsum()
    run_lengths = work.groupby(["date", run_id], sort=False, observed=True).size()
    run_count = run_lengths.groupby(level=0).size()
    max_run = run_lengths.groupby(level=0).max()
    output["mean_same_sign_run_length"] = _safe_divide(output["agg_trade_count"], run_count)
    output["max_same_sign_run_share"] = _safe_divide(max_run, output["agg_trade_count"])

    interarrival = work.groupby("date", sort=False, observed=True)["transact_time"].diff()
    valid_interarrival = interarrival.notna()
    dt = interarrival.where(valid_interarrival, 0.0)
    dt_count = valid_interarrival.groupby(work["date"], observed=True).sum().astype(float)
    dt_sum = dt.groupby(work["date"], observed=True).sum()
    dt_square_sum = (dt * dt).groupby(work["date"], observed=True).sum()
    # A one-event bin has no observable spacing.  Encode that absence as zero
    # rather than leaking a NaN into downstream live features.
    dt_mean = _safe_divide(dt_sum, dt_count).fillna(0.0)
    dt_variance = (
        _safe_divide(dt_square_sum, dt_count).fillna(0.0) - dt_mean**2
    )
    dt_std = np.sqrt(np.maximum(dt_variance, 0.0))
    output["interarrival_mean_ms"] = dt_mean
    output["interarrival_std_ms"] = dt_std
    output["interarrival_burstiness"] = _safe_divide(
        dt_std - dt_mean, dt_std + dt_mean
    ).fillna(0.0)

    buy_count = (work["side"] > 0.0).groupby(work["date"], observed=True).sum()
    sell_count = (work["side"] < 0.0).groupby(work["date"], observed=True).sum()
    # No events on one side means zero observed mean size for that side.
    buy_mean = _safe_divide(output["buy_quote_notional"], buy_count).fillna(0.0)
    sell_mean = _safe_divide(output["sell_quote_notional"], sell_count).fillna(0.0)
    output["buy_sell_event_size_log_ratio"] = np.log((buy_mean + 1.0) / (sell_mean + 1.0))

    output = output.reset_index()
    output = output.loc[:, OUTPUT_COLUMNS]
    numeric = output.drop(columns="date")
    if not np.isfinite(numeric.to_numpy(float)).all():
        bad = numeric.columns[~np.isfinite(numeric.to_numpy(float)).all(axis=0)].tolist()
        raise ValueError(f"non-finite aggregated feature columns: {bad}")
    if output["date"].duplicated().any() or not output["date"].is_monotonic_increasing:
        raise ValueError("aggregated five-minute timestamps are invalid")
    return output


def _month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    months: list[date] = []
    while current < end:
        months.append(current)
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
    return months


def _month_days(month: date, start: date, end: date) -> list[date]:
    next_month = date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)
    current = max(start, month)
    limit = min(end, next_month)
    days: list[date] = []
    while current < limit:
        days.append(current)
        current += timedelta(days=1)
    return days


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-reproducible gzip CSV for stable resume hashes."""
    frame.to_csv(
        path,
        index=False,
        compression=GZIP_COMPRESSION,
        float_format="%.12g",
    )


def _resume_metadata_is_current(
    metadata: dict[str, Any],
    *,
    cfg: BuildConfig,
    month: date,
    expected_days: list[date],
    output_path: Path,
    fetcher: Callable[..., bytes],
) -> bool:
    expected_dates = [day.isoformat() for day in expected_days]
    archives = metadata.get("archives")
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("month") != f"{month:%Y-%m}"
        or metadata.get("symbol") != cfg.symbol
        or metadata.get("requested_dates") != expected_dates
        or not isinstance(archives, list)
        or [item.get("date") for item in archives] != expected_dates
    ):
        return False

    actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if actual != metadata.get("output_sha256"):
        raise ValueError(f"resume artifact hash mismatch: {output_path}")

    # Binance archives can be replaced. Re-fetch the small official checksum
    # sidecars so a resumed build cannot silently preserve superseded data.
    for day, archive in zip(expected_days, archives, strict=True):
        current = expected_sha256(
            fetcher(
                checksum_url(cfg.symbol, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        if current != archive.get("archive_sha256"):
            return False
    return True


def _process_month(
    month: date,
    cfg: BuildConfig,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    output_dir = Path(cfg.output_dir)
    monthly_dir = output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.symbol}_aggtrade_5m_{month:%Y-%m}"
    output_path = monthly_dir / f"{stem}.csv.gz"
    metadata_path = monthly_dir / f"{stem}.json"
    expected_days = _month_days(month, start, end)
    if output_path.exists() and metadata_path.exists() and not cfg.overwrite:
        metadata = json.loads(metadata_path.read_text())
        if _resume_metadata_is_current(
            metadata,
            cfg=cfg,
            month=month,
            expected_days=expected_days,
            output_path=output_path,
            fetcher=fetcher,
        ):
            return metadata

    frames: list[pd.DataFrame] = []
    archives: list[dict[str, Any]] = []
    for day in expected_days:
        checksum_payload = fetcher(
            checksum_url(cfg.symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds
        )
        expected = expected_sha256(checksum_payload)
        payload = fetcher(
            archive_url(cfg.symbol, day), retries=cfg.retries, timeout=cfg.timeout_seconds
        )
        archive_hash = verify_sha256(payload, expected)
        raw = read_archive(payload)
        aggregated = aggregate_five_minute(raw)
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta("1d")
        if not ((aggregated["date"] >= day_start) & (aggregated["date"] < day_end)).all():
            raise ValueError(f"archive contains timestamps outside {day}")
        frames.append(aggregated)
        archives.append(
            {
                "date": day.isoformat(),
                "archive_sha256": archive_hash,
                "agg_trade_rows": int(len(raw)),
                "five_minute_rows": int(len(aggregated)),
                "first_agg_trade_id": int(raw["agg_trade_id"].iloc[0]),
                "last_agg_trade_id": int(raw["agg_trade_id"].iloc[-1]),
                "first_underlying_trade_id": int(raw["first_trade_id"].iloc[0]),
                "last_underlying_trade_id": int(raw["last_trade_id"].iloc[-1]),
            }
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise ValueError(f"month {month:%Y-%m} has duplicate or unordered bins")
    _write_gzip_csv(combined, output_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "month": f"{month:%Y-%m}",
        "symbol": cfg.symbol,
        "requested_dates": [day.isoformat() for day in expected_days],
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "archives": archives,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build(cfg: BuildConfig) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    months = _month_starts(start, end)
    metadata: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        future_map = {executor.submit(_process_month, month, cfg): month for month in months}
        for future in as_completed(future_map):
            month = future_map[future]
            result = future.result()
            metadata.append(result)
            print(f"completed {month:%Y-%m}: rows={result['rows']}", flush=True)
    metadata.sort(key=lambda item: item["month"])

    monthly_frames = [pd.read_csv(item["output"], compression="gzip", parse_dates=["date"]) for item in metadata]
    combined = pd.concat(monthly_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any():
        raise ValueError("combined output has duplicate timestamps")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{cfg.symbol}_aggtrade_5m_{cfg.start}_{date.fromisoformat(cfg.end) - timedelta(days=1)}.csv.gz"
    _write_gzip_csv(combined, combined_path)
    manifest = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "source": "official Binance USD-M Futures daily aggTrades archives",
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "five_minute_bin": "UTC floor of aggregate-trade transaction timestamp",
            "buyer_maker_semantics": "true = buyer passive / seller aggressive, therefore signed side -1",
            "raw_archives_persisted": False,
            "outcomes_opened": False,
        },
        "combined_output": str(combined_path),
        "combined_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "first_date": str(combined["date"].min()),
        "last_date": str(combined["date"].max()),
        "columns": list(combined.columns),
        "months": metadata,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
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
    args = parser.parse_args()
    manifest = build(BuildConfig(**vars(args)))
    print(json.dumps({key: manifest[key] for key in ("combined_output", "rows", "first_date", "last_date")}, indent=2))


if __name__ == "__main__":
    main()
