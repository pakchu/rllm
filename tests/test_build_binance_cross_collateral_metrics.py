from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_cross_collateral_metrics as builder


def _archive(rows: list[str], *, header: str | None = None) -> bytes:
    columns = header or ",".join(builder.RAW_COLUMNS)
    payload = (columns + "\n" + "\n".join(rows) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metrics.csv", payload)
    return buffer.getvalue()


def _row(timestamp: str, symbol: str = "BTCUSDT", taker: str = "0.8") -> str:
    return ",".join([timestamp, symbol, "100", "1000000", "1.1", "1.2", "1.3", taker])


def test_archive_urls_are_exact_daily_paths() -> None:
    day = date(2023, 1, 2)
    assert builder.archive_url("um", "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
        "BTCUSDT-metrics-2023-01-02.zip"
    )
    assert builder.checksum_url("cm", "BTCUSD_PERP", day).endswith(
        "BTCUSD_PERP-metrics-2023-01-02.zip.CHECKSUM"
    )


def test_read_archive_accepts_missing_ratios_and_quarantines_zero_taker() -> None:
    row = "2023-01-01 00:00:00,BTCUSD_PERP,100,2.5,,,,"
    frame, removed = builder.read_archive(_archive([row]), symbol="BTCUSD_PERP")
    assert removed == 0
    assert frame.loc[0, "sum_open_interest"] == 100.0
    assert pd.isna(frame.loc[0, "count_long_short_ratio"])
    assert pd.isna(frame.loc[0, "sum_taker_long_short_vol_ratio"])

    zero_taker = "2023-01-01 00:00:00,BTCUSD_PERP,100,2.5,,,,0"
    frame, _ = builder.read_archive(_archive([zero_taker]), symbol="BTCUSD_PERP")
    assert pd.isna(frame.loc[0, "sum_taker_long_short_vol_ratio"])


def test_read_archive_removes_only_exact_duplicate_rows() -> None:
    row = _row("2023-01-01 00:00:00")
    frame, removed = builder.read_archive(_archive([row, row]), symbol="BTCUSDT")
    assert len(frame) == 1
    assert removed == 1

    conflict = _row("2023-01-01 00:00:00", taker="1.2")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        builder.read_archive(_archive([row, conflict]), symbol="BTCUSDT")


def test_read_archive_rejects_schema_symbol_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="unexpected metrics columns"):
        builder.read_archive(
            _archive([_row("2023-01-01 00:00:00")], header="bad,column"),
            symbol="BTCUSDT",
        )
    with pytest.raises(ValueError, match="another symbol"):
        builder.read_archive(
            _archive([_row("2023-01-01 00:00:00", symbol="ETHUSDT")]),
            symbol="BTCUSDT",
        )
    unavailable = _row("2023-01-01 00:00:00").replace(",100,1000000,", ",0,1000000,")
    frame, _ = builder.read_archive(_archive([unavailable]), symbol="BTCUSDT")
    assert frame[list(builder.OPEN_INTEREST_COLUMNS)].isna().all(axis=None)

    invalid = _row("2023-01-01 00:00:00").replace(",100,1000000,", ",-1,1000000,")
    with pytest.raises(ValueError, match="invalid open interest"):
        builder.read_archive(_archive([invalid]), symbol="BTCUSDT")

    malformed = _row("2023-01-01 00:00:00").replace(",100,1000000,", ",BAD,1000000,")
    with pytest.raises(ValueError, match="malformed sum_open_interest"):
        builder.read_archive(_archive([malformed]), symbol="BTCUSDT")


def test_process_day_verifies_checksum_and_day_boundary() -> None:
    payload = _archive([_row("2023-01-01 00:00:00")])
    digest = hashlib.sha256(payload).hexdigest()

    def fetcher(url: str, **_: object) -> bytes:
        return (
            (digest + "  file.zip\n").encode() if url.endswith("CHECKSUM") else payload
        )

    result = builder.process_day(
        "um", "BTCUSDT", date(2023, 1, 1), builder.Config(), fetcher=fetcher
    )
    assert result["available"] is True
    assert result["archive_sha256"] == digest
    assert result["expected_archive_sha256"] == digest
    assert result["missing_five_minute_rows"] == 287

    def bad_checksum_fetcher(url: str, **_: object) -> bytes:
        return b"0" * 64 + b"  file.zip\n" if url.endswith("CHECKSUM") else payload

    with pytest.raises(ValueError, match="checksum"):
        builder.process_day(
            "um",
            "BTCUSDT",
            date(2023, 1, 1),
            builder.Config(),
            fetcher=bad_checksum_fetcher,
        )

    wrong_day = _archive([_row("2023-01-02 00:00:00")])
    wrong_digest = hashlib.sha256(wrong_day).hexdigest()

    def wrong_fetcher(url: str, **_: object) -> bytes:
        return (
            (wrong_digest + "  file.zip\n").encode()
            if url.endswith("CHECKSUM")
            else wrong_day
        )

    with pytest.raises(ValueError, match="another UTC date"):
        builder.process_day(
            "um",
            "BTCUSDT",
            date(2023, 1, 1),
            builder.Config(),
            fetcher=wrong_fetcher,
        )


def test_build_hard_cutoff_prevents_2024_access() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-01-02"))


def test_mocked_build_writes_deterministic_self_contained_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_process_day(
        venue: str,
        symbol: str,
        day: date,
        _: builder.Config,
    ) -> dict[str, object]:
        timestamps = pd.date_range(day, periods=288, freq="5min")
        frame = pd.DataFrame(
            {
                "create_time": timestamps,
                "symbol": symbol,
                "sum_open_interest": 100.0,
                "sum_open_interest_value": 1_000.0,
                "count_toptrader_long_short_ratio": 1.0,
                "sum_toptrader_long_short_ratio": 1.0,
                "count_long_short_ratio": 1.0,
                "sum_taker_long_short_vol_ratio": 1.0,
            }
        )
        digest = hashlib.sha256(f"{venue}-{day}".encode()).hexdigest()
        return {
            "venue": venue,
            "symbol": symbol,
            "date": day.isoformat(),
            "available": True,
            "archive_url": "https://example.invalid/archive.zip",
            "checksum_url": "https://example.invalid/archive.zip.CHECKSUM",
            "archive_sha256": digest,
            "expected_archive_sha256": digest,
            "checksum_payload_sha256": hashlib.sha256(b"checksum").hexdigest(),
            "rows": len(frame),
            "duplicate_rows_removed": 0,
            "missing_five_minute_rows": 0,
            "invalid_open_interest_rows": 0,
            "missing_taker_ratio_rows": 0,
            "first_time": str(timestamps.min()),
            "last_time": str(timestamps.max()),
            "frame": frame,
        }

    monkeypatch.setattr(builder, "process_day", fake_process_day)
    cfg = replace(
        builder.Config(),
        start="2021-07-08",
        end="2021-07-09",
        workers=2,
        output_dir=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
    )
    first = builder.build(cfg)
    first_bytes = Path(cfg.manifest).read_bytes()
    second = builder.build(cfg)
    assert Path(cfg.manifest).read_bytes() == first_bytes
    assert first == second
    for archive in first["archives"]:
        assert archive["archive_sha256"] == archive["expected_archive_sha256"]


def test_frozen_source_manifest_matches_local_panel() -> None:
    manifest = json.loads(Path(builder.Config.manifest).read_text())
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["post_2023_rows_requested"] is False
    assert manifest["file"]["rows"] == 261_216
    assert manifest["file"]["source_complete_rows"] == 220_370
    assert len(manifest["archives"]) == 1_814
    assert manifest["missing_archive_dates"]["um"] == []
    assert len(manifest["missing_archive_dates"]["cm"]) == 11

    path = Path(manifest["file"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["file"]["sha256"]
    panel = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    assert len(panel) == manifest["file"]["rows"]
    assert panel["date"].duplicated().sum() == 0
    assert panel["date"].diff().dropna().eq(pd.Timedelta(minutes=5)).all()
    required = [
        f"{venue}_{column}"
        for venue in builder.VENUES
        for column in builder.REQUIRED_NUMERIC
    ]
    expected_complete = panel[required].notna().all(axis=1)
    assert expected_complete.equals(panel["source_complete"].astype(bool))
    assert panel["date"].max() < pd.Timestamp("2024-01-01")
