from __future__ import annotations

import io
import hashlib
import json
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_binance_cross_collateral_book_depth_2023 as builder


def _archive(frame: pd.DataFrame) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("depth.csv", frame.to_csv(index=False))
    return payload.getvalue()


def _raw_snapshots(count: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in range(count):
        timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(seconds=30 * snapshot)
        for level in builder.PERCENTAGES:
            distance = abs(level)
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": 100.0 * distance + snapshot,
                    "notional": 1_000.0 * distance + snapshot,
                }
            )
    return pd.DataFrame(rows)


def test_archive_urls_are_official_and_market_specific() -> None:
    day = date(2023, 1, 2)
    assert builder.archive_url("um", "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2023-01-02.zip"
    )
    assert builder.checksum_url("cm", "BTCUSD_PERP", day).endswith(
        "BTCUSD_PERP-bookDepth-2023-01-02.zip.CHECKSUM"
    )


def test_read_archive_requires_complete_monotone_cumulative_levels() -> None:
    raw = _raw_snapshots()
    parsed = builder.read_archive(_archive(raw))
    assert len(parsed) == 100
    assert parsed["timestamp"].nunique() == 10

    missing = raw.loc[~((raw["timestamp"] == raw["timestamp"].iloc[0]) & (raw["percentage"] == 5))]
    with pytest.raises(ValueError, match=r"all \+/-1..5 levels"):
        builder.read_archive(_archive(missing))

    broken = raw.copy()
    broken.loc[broken["percentage"] == -5, "depth"] = 1.0
    with pytest.raises(ValueError, match="not monotonic"):
        builder.read_archive(_archive(broken))


def test_five_minute_aggregation_requires_broad_snapshot_coverage() -> None:
    cfg = builder.Config()
    accepted = builder.aggregate_five_minute(
        builder.read_archive(_archive(_raw_snapshots())),
        cfg,
    )
    assert accepted["date"].tolist() == [pd.Timestamp("2023-01-01")]
    assert accepted.loc[0, "snapshot_count"] == 10
    assert accepted.loc[0, "first_offset_seconds"] == 0.0
    assert accepted.loc[0, "last_offset_seconds"] == 270.0
    assert accepted.loc[0, "depth_m1"] == np.median(np.arange(100.0, 110.0))

    too_few = builder.aggregate_five_minute(
        builder.read_archive(_archive(_raw_snapshots(7))),
        cfg,
    )
    assert too_few.empty


def test_process_day_records_missing_official_archive() -> None:
    def missing_fetcher(url: str, *, retries: int, timeout: int) -> bytes:
        del url, retries, timeout
        raise FileNotFoundError

    result = builder.process_day(
        "um",
        "BTCUSDT",
        date(2023, 2, 8),
        builder.Config(),
        fetcher=missing_fetcher,
    )
    assert result["available"] is False
    assert result["frame"].empty


def test_empty_accepted_frame_is_still_venue_prefixed() -> None:
    frame = pd.DataFrame(columns=["date", "depth_m1", "snapshot_count"])
    prefixed = builder._prefix_frame(frame, "cm")
    assert prefixed.columns.tolist() == [
        "date",
        "cm_depth_m1",
        "cm_snapshot_count",
    ]


def test_builder_rejects_post_2023_requests_before_network() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-01-02"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), start="2022-12-31"))


def test_frozen_manifest_matches_pre2024_depth_panel() -> None:
    manifest = json.loads(
        Path(
            "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
        ).read_text()
    )
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["post_2023_rows_requested"] is False
    assert manifest["missing_archive_dates"] == {
        "um": ["2023-02-08", "2023-02-09"],
        "cm": ["2023-09-25"],
    }
    item = manifest["file"]
    assert item["rows"] == 105_120
    assert item["source_complete_rows"] == 101_649
    path = Path(item["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    frame = pd.read_csv(path, compression="gzip", nrows=1)
    assert len(frame.columns) == 28
    assert not any(column.endswith(("_x", "_y")) for column in frame)
