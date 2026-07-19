from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import build_binance_um_premium_path as builder


def _archive(rows: list[list[object]], *, header: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        body = pd.DataFrame(
            rows, columns=cast(Any, list(builder.RAW_COLUMNS))
        ).to_csv(
            index=False, header=header
        )
        archive.writestr("BTCUSDT-1m.csv", body)
    return buffer.getvalue()


def _row(open_time: int, *, scale: int = 1) -> list[object]:
    close_time = open_time + 60_000 * scale - scale
    return [
        open_time,
        -0.0010,
        0.0005,
        -0.0015,
        0.0001,
        0,
        close_time,
        0,
        60,
        0,
        0,
        0,
    ]


def test_archive_urls_use_official_monthly_premium_index_path() -> None:
    month = date(2021, 1, 1)
    expected = (
        "https://data.binance.vision/data/futures/um/monthly/"
        "premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2021-01.zip"
    )
    assert builder.archive_url("BTCUSDT", "1m", month) == expected
    assert builder.checksum_url("BTCUSDT", "1m", month) == expected + ".CHECKSUM"


@pytest.mark.parametrize("header", [False, True])
def test_read_archive_is_causal_and_outcome_free(header: bool) -> None:
    start = 1_609_459_200_000
    frame = builder.read_archive(
        _archive([_row(start), _row(start + 60_000)], header=header)
    )
    assert frame.columns.tolist() == list(builder.OUTPUT_COLUMNS)
    assert frame["date"].tolist() == [
        pd.Timestamp("2021-01-01 00:00"),
        pd.Timestamp("2021-01-01 00:01"),
    ]
    assert frame["source_close_time"].iloc[0] == pd.Timestamp(
        "2021-01-01 00:00:59.999"
    )
    assert frame["feature_available_time"].iloc[0] == pd.Timestamp(
        "2021-01-01 00:01:01"
    )
    assert not any(
        forbidden in column
        for column in frame.columns
        for forbidden in ("btc_open", "btc_close", "return", "pnl", "funding")
    )


def test_read_archive_normalizes_microsecond_timestamps() -> None:
    start_us = 1_735_689_600_000_000
    frame = builder.read_archive(_archive([_row(start_us, scale=1_000)]))
    assert frame["date"].iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert frame["source_close_time"].iloc[0] == pd.Timestamp(
        "2025-01-01 00:00:59.999"
    )


def test_read_archive_rejects_mixed_units_and_bad_ohlc() -> None:
    millisecond = _row(1_609_459_200_000)
    microsecond = _row(1_609_459_260_000_000, scale=1_000)
    with pytest.raises(ValueError, match="mixes millisecond and microsecond"):
        builder.read_archive(_archive([millisecond, microsecond]))

    bad = _row(1_609_459_200_000)
    bad[2] = -0.0020
    with pytest.raises(ValueError, match="OHLC envelope"):
        builder.read_archive(_archive([bad]))


def test_process_month_verifies_checksum_and_marks_missing_rows_invalid() -> None:
    cfg = replace(
        builder.Config(),
        start="2021-01-01",
        end="2021-01-02",
        retries=1,
    )
    start = 1_609_459_200_000
    payload = _archive([_row(start), _row(start + 120_000)])
    digest = hashlib.sha256(payload).hexdigest()

    def fetcher(url: str, **_: object) -> bytes:
        return (digest + "  archive.zip\n").encode() if url.endswith(".CHECKSUM") else payload

    frame, metadata = builder.process_month(date(2021, 1, 1), cfg, fetcher=fetcher)
    assert len(frame) == 1_440
    assert frame["source_valid"].iloc[:3].tolist() == [True, False, True]
    assert frame.loc[1, list(builder.OUTPUT_COLUMNS[4:])].isna().all()
    assert metadata["archive_sha256"] == digest
    assert metadata["missing_rows"] == 1_438


def test_process_month_missing_archive_is_explicitly_invalid() -> None:
    cfg = replace(
        builder.Config(),
        start="2021-01-01",
        end="2021-01-02",
        retries=1,
    )

    def missing(_: str, **__: object) -> bytes:
        raise FileNotFoundError

    frame, metadata = builder.process_month(date(2021, 1, 1), cfg, fetcher=missing)
    assert metadata["available"] is False
    assert len(frame) == 1_440
    assert not bool(frame["source_valid"].any())
    assert frame.loc[:, list(builder.OUTPUT_COLUMNS[4:])].isna().all().all()


def test_build_manifest_remains_source_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = replace(
        builder.Config(),
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
        workers=1,
    )
    dates = pd.date_range(cfg.start, cfg.end, freq="1min", inclusive="left")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_close_time": dates + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "feature_available_time": dates + pd.Timedelta(minutes=1, seconds=1),
            "source_valid": True,
            "premium_open": np.zeros(len(dates)),
            "premium_high": np.ones(len(dates)),
            "premium_low": -np.ones(len(dates)),
            "premium_close": np.zeros(len(dates)),
        }
    )

    monkeypatch.setattr(
        builder,
        "process_month",
        lambda _month, _cfg: (
            frame,
            {
                "month": "2021-01",
                "available": True,
                "archive_sha256": "a" * 64,
                "rows": len(frame),
                "source_valid_rows": len(frame),
                "missing_rows": 0,
            },
        ),
    )
    manifest = builder.build(cfg)
    assert manifest["protocol"]["source_only"] is True
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["btc_execution_prices_retained"] is False
    assert manifest["protocol"]["returns_or_pnl_retained"] is False
    assert set(manifest["retained_columns"]) == set(builder.OUTPUT_COLUMNS)
    assert Path(manifest["file"]["path"]).exists()
