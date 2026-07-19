from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_binance_um_book_centroid_2023 as builder


def _archive(frame: pd.DataFrame) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("depth.csv", frame.to_csv(index=False))
    return payload.getvalue()


def _checksum(payload: bytes) -> bytes:
    return f"{hashlib.sha256(payload).hexdigest()}  depth.zip\n".encode()


def _raw_snapshots(count: int = 10, *, bid_shift: float = 0.0, ask_shift: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for snapshot in range(count):
        timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(seconds=30 * snapshot)
        for level in (-5, -4, -3, -2, -1):
            distance = abs(level)
            depth = 100.0 * distance
            avg_quote = 100.0 / np.exp(0.01 * distance) + bid_shift * snapshot * distance
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": depth,
                    "notional": depth * avg_quote,
                }
            )
        for level in (1, 2, 3, 4, 5):
            depth = 100.0 * level
            avg_quote = 100.0 * np.exp(0.01 * level) + ask_shift * snapshot * level
            rows.append(
                {
                    "timestamp": timestamp,
                    "percentage": level,
                    "depth": depth,
                    "notional": depth * avg_quote,
                }
            )
    return pd.DataFrame(rows)


def test_archive_url_is_official_usd_m_btcusdt_book_depth() -> None:
    day = date(2023, 1, 2)
    assert builder.archive_url(day) == (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2023-01-02.zip"
    )
    assert builder.checksum_url(day).endswith("BTCUSDT-bookDepth-2023-01-02.zip.CHECKSUM")


def test_formula_direction_and_symmetry() -> None:
    skew = builder.snapshots_to_skew(builder.read_archive(_archive(_raw_snapshots(1))))
    for k in builder.SKEW_DISTANCES:
        assert skew.loc[0, f"skew_{k}"] == pytest.approx(0.0, abs=2e-15)

    ask_farther = _raw_snapshots(1)
    ask_farther.loc[ask_farther["percentage"] > 1, "notional"] *= 1.01
    skew = builder.snapshots_to_skew(builder.read_archive(_archive(ask_farther)))
    assert skew.loc[0, "skew_5"] > 0.0

    bid_farther = _raw_snapshots(1)
    bid_farther.loc[bid_farther["percentage"] < -1, "notional"] *= 0.99
    skew = builder.snapshots_to_skew(builder.read_archive(_archive(bid_farther)))
    assert skew.loc[0, "skew_5"] < 0.0


def test_monotonic_and_crossed_average_quotes_are_invalid() -> None:
    raw = _raw_snapshots(1)
    broken_bid = raw.copy()
    mask = broken_bid["percentage"].eq(-5)
    broken_bid.loc[mask, "notional"] = broken_bid.loc[mask, "depth"] * 100.0
    with pytest.raises(ValueError, match="bid average quote"):
        builder.snapshots_to_skew(builder.read_archive(_archive(broken_bid)))

    crossed = raw.copy()
    mask = crossed["percentage"].eq(1)
    crossed.loc[mask, "notional"] = crossed.loc[mask, "depth"] * 98.0
    with pytest.raises(ValueError, match="crossed"):
        builder.snapshots_to_skew(builder.read_archive(_archive(crossed)))


def test_aggregation_timing_path_and_efficiency_identities() -> None:
    cfg = builder.Config()
    raw = _raw_snapshots(10, ask_shift=0.02)
    bars = builder.aggregate_five_minute(builder.read_archive(_archive(raw)), cfg)
    assert bars["date"].tolist() == [pd.Timestamp("2023-01-01")]
    row = bars.iloc[0]
    assert row["snapshot_count"] == 10
    assert row["first_offset_seconds"] == 0.0
    assert row["last_offset_seconds"] == 270.0

    skew = builder.snapshots_to_skew(builder.read_archive(_archive(raw)))
    values = skew["skew_5"].to_numpy(float)
    expected_net = values[-1] - values[0]
    expected_path = np.abs(np.diff(values)).sum()
    assert row["skew_5_median"] == pytest.approx(np.median(values))
    assert row["skew_5_net"] == pytest.approx(expected_net)
    assert row["skew_5_path"] == pytest.approx(expected_path)
    assert row["skew_5_efficiency"] == pytest.approx(abs(expected_net) / expected_path)

    flat = builder.aggregate_five_minute(builder.read_archive(_archive(_raw_snapshots(10))), cfg)
    assert flat.loc[0, "skew_5_path"] == pytest.approx(0.0)
    assert flat.loc[0, "skew_5_efficiency"] == pytest.approx(0.0)


def test_physical_seal_rejects_non_2023_builds_before_network() -> None:
    with pytest.raises(ValueError, match="physically sealed"):
        builder.build(replace(builder.Config(), end="2024-01-02"))
    with pytest.raises(ValueError, match="physically sealed"):
        builder.build(replace(builder.Config(), start="2022-12-31"))


def test_process_day_verifies_checksum_and_replays_frozen_manifest_subset() -> None:
    payload = _archive(_raw_snapshots(10))
    expected_hash = hashlib.sha256(payload).hexdigest()
    reference = {
        "available": True,
        "archive_sha256": expected_hash,
        "raw_rows": 100,
        "snapshot_count": 10,
    }

    def fetcher(url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        if url.endswith(".CHECKSUM"):
            return _checksum(payload)
        return payload

    result = builder.process_day(date(2023, 1, 1), builder.Config(), reference, fetcher=fetcher)
    assert result["available"] is True
    assert result["archive_sha256"] == expected_hash
    assert result["raw_rows"] == 100
    assert result["snapshot_count"] == 10
    assert result["accepted_bar_count"] == 1

    mismatch = {**reference, "raw_rows": 90}
    with pytest.raises(ValueError, match="mismatches frozen"):
        builder.process_day(date(2023, 1, 1), builder.Config(), mismatch, fetcher=fetcher)


def test_missing_archive_replay_mismatch_is_fail_closed() -> None:
    def missing_fetcher(url: str, *, retries: int, timeout: int) -> bytes:
        del url, retries, timeout
        raise FileNotFoundError

    missing_reference = {"available": False}
    result = builder.process_day(date(2023, 2, 8), builder.Config(), missing_reference, fetcher=missing_fetcher)
    assert result["available"] is False

    available_reference = {"available": True, "archive_sha256": "x", "raw_rows": 1, "snapshot_count": 1}
    with pytest.raises(ValueError, match="mismatches frozen"):
        builder.process_day(date(2023, 2, 8), builder.Config(), available_reference, fetcher=missing_fetcher)


def test_frozen_manifest_has_um_subset_for_replay() -> None:
    path = Path(builder.Config.reference_manifest)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == builder.REFERENCE_MANIFEST_SHA256
    manifest = json.loads(path.read_text())
    um = [item for item in manifest["archives"] if item["venue"] == "um" and item["symbol"] == "BTCUSDT"]
    assert len(um) == 365
    assert manifest["protocol"]["outcomes_opened"] is False
    assert {item["date"] for item in um if not item["available"]} == {"2023-02-08", "2023-02-09"}


def test_reference_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"protocol":{"outcomes_opened":false},"archives":[]}\n')
    with pytest.raises(ValueError, match="reference manifest hash mismatch"):
        builder._load_reference_records(path)
