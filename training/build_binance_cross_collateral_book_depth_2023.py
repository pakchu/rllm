"""Build a verified 2023 USD-M/COIN-M BTC book-depth panel.

The official Binance Vision daily archives contain cumulative depth snapshots
at +/-1..5 percent from the book. This builder verifies every available archive
against its checksum, reduces complete snapshots to causal five-minute medians,
and writes a physically pre-2024 panel. It never reads price outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.build_binance_aggtrade_microstructure import (
    _fetch_bytes,
    _write_gzip_csv,
    expected_sha256,
    verify_sha256,
)


BASE_URL = "https://data.binance.vision/data/futures"
RAW_COLUMNS = ("timestamp", "percentage", "depth", "notional")
PERCENTAGES = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
IGNORED_OPTIONAL_PERCENTAGES = (-0.2, 0.2)
VENUES = {
    "um": "BTCUSDT",
    "cm": "BTCUSD_PERP",
}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Config:
    start: str = "2023-01-01"
    end: str = "2024-01-01"
    output_dir: str = "data/binance_cross_collateral_book_depth_btc_2023"
    manifest: str = (
        "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
    )
    workers: int = 8
    retries: int = 5
    timeout_seconds: int = 60
    minimum_snapshots_per_bar: int = 8
    maximum_first_snapshot_offset_seconds: float = 60.0
    minimum_last_snapshot_offset_seconds: float = 240.0


def archive_url(venue: str, symbol: str, day: date) -> str:
    stem = f"{symbol}-bookDepth-{day:%Y-%m-%d}.zip"
    return f"{BASE_URL}/{venue}/daily/bookDepth/{symbol}/{stem}"


def checksum_url(venue: str, symbol: str, day: date) -> str:
    return archive_url(venue, symbol, day) + ".CHECKSUM"


def _days(start: date, end: date) -> list[date]:
    values: list[date] = []
    current = start
    while current < end:
        values.append(current)
        current += timedelta(days=1)
    return values


def read_archive(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one book-depth CSV, found {members}")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, low_memory=False)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected book-depth columns: {frame.columns.tolist()}")
    if frame.empty:
        raise ValueError("book-depth archive is empty")

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="raise",
    ).dt.tz_localize(None)
    percentage = pd.to_numeric(frame["percentage"], errors="raise")
    if not np.isfinite(percentage.to_numpy(float)).all():
        raise ValueError("book-depth percentage is non-finite")
    allowed = np.asarray(PERCENTAGES + IGNORED_OPTIONAL_PERCENTAGES, dtype=float)
    recognized = np.isclose(
        percentage.to_numpy(float)[:, None], allowed[None, :], rtol=0.0, atol=1e-12
    ).any(axis=1)
    if not recognized.all():
        unexpected = sorted(set(percentage.loc[~recognized].astype(float).tolist()))
        raise ValueError(f"book-depth percentage contains unsupported levels: {unexpected}")
    # Binance added +/-0.2% snapshots to USD-M archives in January 2026.
    # The frozen feature is defined only on +/-1..5%, so ignore the additive
    # levels rather than changing the selected feature's meaning.
    required = percentage.isin(PERCENTAGES)
    frame = frame.loc[required].copy()
    frame["percentage"] = percentage.loc[required].astype(np.int8)
    for column in ("depth", "notional"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["depth", "notional"]].to_numpy(float)).all():
        raise ValueError("book-depth archive contains non-finite values")
    if (frame[["depth", "notional"]] <= 0.0).any().any():
        raise ValueError("book-depth archive contains non-positive values")
    if frame.duplicated(["timestamp", "percentage"]).any():
        raise ValueError("book-depth archive contains duplicate snapshot levels")

    level_counts = (
        frame.groupby("timestamp", sort=True, observed=True)["percentage"]
        .agg(lambda values: tuple(sorted(int(value) for value in values)))
    )
    expected_levels = tuple(PERCENTAGES)
    if not level_counts.map(lambda values: values == expected_levels).all():
        raise ValueError("book-depth snapshot does not contain all +/-1..5 levels")
    if not level_counts.index.is_monotonic_increasing:
        raise ValueError("book-depth timestamps are not increasing")

    pivot = frame.pivot(index="timestamp", columns="percentage", values="depth")
    bid = pivot.loc[:, [-1, -2, -3, -4, -5]].to_numpy(float)
    ask = pivot.loc[:, [1, 2, 3, 4, 5]].to_numpy(float)
    if (np.diff(bid, axis=1) < 0.0).any() or (np.diff(ask, axis=1) < 0.0).any():
        raise ValueError("cumulative book depth is not monotonic with distance")
    return frame.sort_values(["timestamp", "percentage"]).reset_index(drop=True)


def aggregate_five_minute(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = work["timestamp"].dt.floor("5min")
    snapshots = work[["timestamp", "date"]].drop_duplicates()
    timing = snapshots.groupby("date", sort=True, observed=True).agg(
        snapshot_count=("timestamp", "size"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    timing["first_offset_seconds"] = (
        timing["first_timestamp"] - timing.index
    ).dt.total_seconds()
    timing["last_offset_seconds"] = (
        timing["last_timestamp"] - timing.index
    ).dt.total_seconds()
    valid = (
        timing["snapshot_count"].ge(cfg.minimum_snapshots_per_bar)
        & timing["first_offset_seconds"].le(
            cfg.maximum_first_snapshot_offset_seconds
        )
        & timing["last_offset_seconds"].ge(
            cfg.minimum_last_snapshot_offset_seconds
        )
    )

    depth = work.pivot_table(
        index="date",
        columns="percentage",
        values="depth",
        aggfunc="median",
        observed=True,
    ).reindex(columns=PERCENTAGES)
    depth.columns = [
        f"depth_{'m' if level < 0 else 'p'}{abs(level)}"
        for level in depth.columns
    ]
    output = depth.join(
        timing[
            [
                "snapshot_count",
                "first_offset_seconds",
                "last_offset_seconds",
            ]
        ],
        how="inner",
    )
    output = output.loc[valid].reset_index()
    if output.empty:
        return output
    depth_columns = [column for column in output if column.startswith("depth_")]
    if output[depth_columns].isna().any().any():
        raise ValueError("accepted book-depth bar contains a missing level")
    return output


def _empty_day(venue: str, symbol: str, day: date) -> dict[str, Any]:
    return {
        "venue": venue,
        "symbol": symbol,
        "date": day.isoformat(),
        "available": False,
        "reason": "official archive or checksum not published",
        "frame": pd.DataFrame(),
    }


def process_day(
    venue: str,
    symbol: str,
    day: date,
    cfg: Config,
    *,
    fetcher: Callable[..., bytes] = _fetch_bytes,
) -> dict[str, Any]:
    try:
        checksum = expected_sha256(
            fetcher(
                checksum_url(venue, symbol, day),
                retries=cfg.retries,
                timeout=cfg.timeout_seconds,
            )
        )
        payload = fetcher(
            archive_url(venue, symbol, day),
            retries=cfg.retries,
            timeout=cfg.timeout_seconds,
        )
    except FileNotFoundError:
        return _empty_day(venue, symbol, day)

    archive_hash = verify_sha256(payload, checksum)
    raw = read_archive(payload)
    day_start = pd.Timestamp(day)
    day_end = day_start + pd.Timedelta(days=1)
    if raw["timestamp"].lt(day_start).any() or raw["timestamp"].ge(day_end).any():
        raise ValueError(f"{venue} archive {day} contains another UTC date")
    bars = aggregate_five_minute(raw, cfg)
    return {
        "venue": venue,
        "symbol": symbol,
        "date": day.isoformat(),
        "available": True,
        "archive_sha256": archive_hash,
        "raw_rows": int(len(raw)),
        "snapshot_count": int(raw["timestamp"].nunique()),
        "accepted_bar_count": int(len(bars)),
        "first_timestamp": str(raw["timestamp"].min()),
        "last_timestamp": str(raw["timestamp"].max()),
        "frame": bars,
    }


def _public_record(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "frame"}


def _prefix_frame(frame: pd.DataFrame, venue: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{venue}_{column}" for column in frame if column != "date"}
    )


def build(cfg: Config) -> dict[str, Any]:
    start = date.fromisoformat(cfg.start)
    end = date.fromisoformat(cfg.end)
    if start >= end:
        raise ValueError("start must precede exclusive end")
    if start < date(2023, 1, 1) or end > date(2024, 1, 1):
        raise ValueError("book-depth build is physically bounded to calendar 2023")
    if cfg.workers < 1:
        raise ValueError("workers must be positive")
    if not 1 <= cfg.minimum_snapshots_per_bar <= 10:
        raise ValueError("minimum snapshots per bar must be in [1, 10]")
    if not 0.0 <= cfg.maximum_first_snapshot_offset_seconds < 300.0:
        raise ValueError("first snapshot offset bound is invalid")
    if not 0.0 <= cfg.minimum_last_snapshot_offset_seconds < 300.0:
        raise ValueError("last snapshot offset bound is invalid")

    days = _days(start, end)
    tasks = [
        (venue, symbol, day)
        for venue, symbol in VENUES.items()
        for day in days
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {
            executor.submit(process_day, venue, symbol, day, cfg): (
                venue,
                day,
            )
            for venue, symbol, day in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["venue"], item["date"]))

    venue_panels: dict[str, pd.DataFrame] = {}
    for venue in VENUES:
        frames = [
            _prefix_frame(item["frame"], venue)
            for item in results
            if item["venue"] == venue and item["available"]
        ]
        venue_panels[venue] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
        )

    full_grid = pd.DataFrame(
        {
            "date": pd.date_range(
                start,
                end,
                freq="5min",
                inclusive="left",
            )
        }
    )
    panel = full_grid
    for venue in VENUES:
        panel = panel.merge(
            venue_panels[venue],
            on="date",
            how="left",
            validate="one_to_one",
        )
    required_depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in VENUES
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    panel["source_complete"] = panel[required_depth].notna().all(axis=1)
    if panel["date"].duplicated().any() or not panel["date"].is_monotonic_increasing:
        raise ValueError("combined depth panel timestamps are invalid")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "BTC_cross_collateral_book_depth_5m_2023.csv.gz"
    _write_gzip_csv(panel, output)
    file_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    records = [_public_record(item) for item in results]
    missing = {
        venue: [
            item["date"]
            for item in records
            if item["venue"] == venue and not item["available"]
        ]
        for venue in VENUES
    }
    manifest = {
        "protocol": {
            "name": "Binance BTC cross-collateral book-depth 2023 panel",
            "outcomes_opened": False,
            "start_inclusive": str(pd.Timestamp(start)),
            "end_exclusive": str(pd.Timestamp(end)),
            "post_2023_rows_requested": False,
            "source": "official public Binance Vision daily archives",
        },
        "config": asdict(cfg),
        "venues": VENUES,
        "archive_root": BASE_URL,
        "missing_archive_dates": missing,
        "archives": records,
        "file": {
            "path": str(output),
            "sha256": file_hash,
            "rows": int(len(panel)),
            "source_complete_rows": int(panel["source_complete"].sum()),
            "first_date": str(panel["date"].min()),
            "last_date": str(panel["date"].max()),
        },
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=Config.workers)
    args = parser.parse_args()
    result = build(Config(workers=args.workers))
    print(
        json.dumps(
            {
                "outcomes_opened": result["protocol"]["outcomes_opened"],
                "missing_archive_dates": result["missing_archive_dates"],
                "file": result["file"],
                "manifest": Config.manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
