from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from training import build_coinm_usdm_liquidation_absorption_source as builder


def _row(timestamp: str, *, quote: float = 1_000.0, taker_buy: float = 650.0, trades: int = 20) -> list[object]:
    parsed = cast(pd.Timestamp, pd.Timestamp(timestamp, tz="UTC"))
    open_time = int(parsed.timestamp() * 1_000)
    return [
        open_time,
        100.0,
        101.0,
        99.0,
        100.5,
        10.0,
        open_time + 299_999,
        quote,
        trades,
        6.5,
        taker_buy,
        0,
    ]


def _day_rows(day: str, *, quote: float = 1_000.0, taker_buy: float = 650.0) -> list[list[object]]:
    stamps = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), inclusive="left", freq="5min")
    return [_row(str(stamp), quote=quote, taker_buy=taker_buy) for stamp in stamps]


def _archive(rows: list[list[object]], *, header: bool = True) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=pd.Index(builder.RAW_COLUMNS)).to_csv(
        text, index=False, header=header
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-5m-test.csv", text.getvalue())
    return output.getvalue()


class FakeFetcher:
    def __init__(self, payload_by_day: dict[str, bytes], *, bad_checksum_days: set[str] | None = None) -> None:
        self.payload_by_day = payload_by_day
        self.bad_checksum_days = bad_checksum_days or set()
        self.urls: list[str] = []

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        self.urls.append(url)
        day = next(key for key in self.payload_by_day if key in url)
        payload = self.payload_by_day[day]
        if url.endswith(".CHECKSUM"):
            digest = hashlib.sha256(payload).hexdigest()
            if day in self.bad_checksum_days:
                digest = "0" * 64
            return f"{digest}  BTCUSDT-5m-{day}.zip\n".encode()
        return payload


def test_official_usdm_daily_kline_url() -> None:
    assert builder.archive_url("BTCUSDT", "5m", date(2024, 10, 14)) == (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        "BTCUSDT/5m/BTCUSDT-5m-2024-10-14.zip"
    )


@pytest.mark.parametrize("header", [False, True])
def test_read_activity_archive_retains_only_activity_columns(header: bool) -> None:
    frame = builder.read_activity_archive(_archive([_row("2023-06-25")], header=header))
    assert tuple(frame.columns) == builder.OUTPUT_COLUMNS
    assert frame.loc[0, "date"] == pd.Timestamp("2023-06-25")
    assert frame.loc[0, "feature_available_time"] == pd.Timestamp("2023-06-25 00:05:01")
    assert frame.loc[0, "quote_asset_volume"] == 1_000.0
    assert frame.loc[0, "taker_buy_quote"] == 650.0
    assert frame.loc[0, "taker_sell_quote"] == 350.0
    assert frame.loc[0, "taker_imbalance"] == pytest.approx(0.3)
    assert "open" not in frame.columns
    assert "close" not in frame.columns


def test_read_activity_archive_rejects_taker_buy_above_quote() -> None:
    payload = _archive([_row("2023-06-25", quote=100.0, taker_buy=101.0)])
    with pytest.raises(ValueError, match="taker_buy_quote exceeds"):
        builder.read_activity_archive(payload)


def test_read_activity_archive_accepts_large_quote_roundoff() -> None:
    payload = _archive(
        [
            _row(
                "2023-06-25",
                quote=9_876_543_210.123457,
                taker_buy=6_543_210_987.765432,
            )
        ]
    )
    frame = builder.read_activity_archive(payload)
    assert frame.loc[0, "taker_sell_quote"] > 0.0
    assert np.isclose(
        frame.loc[0, "taker_buy_quote"] + frame.loc[0, "taker_sell_quote"],
        frame.loc[0, "quote_asset_volume"],
    )


def test_process_day_rejects_incomplete_grid() -> None:
    payload = _archive(_day_rows("2023-06-25")[:-1])
    fetcher = FakeFetcher({"2023-06-25": payload})
    cfg = builder.BuildConfig(start="2023-06-25", end="2023-06-26", workers=1)
    with pytest.raises(ValueError, match="incomplete timestamp coverage"):
        builder.process_day(date(2023, 6, 25), cfg, fetcher=fetcher)


def test_process_day_rejects_checksum_mismatch() -> None:
    payload = _archive(_day_rows("2023-06-25"))
    fetcher = FakeFetcher({"2023-06-25": payload}, bad_checksum_days={"2023-06-25"})
    cfg = builder.BuildConfig(start="2023-06-25", end="2023-06-26", workers=1)
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        builder.process_day(date(2023, 6, 25), cfg, fetcher=fetcher)


def test_build_writes_activity_file_and_results_manifest(tmp_path: Path) -> None:
    payloads = {
        "2023-06-25": _archive(_day_rows("2023-06-25", quote=1_000.0, taker_buy=750.0)),
        "2023-06-26": _archive(_day_rows("2023-06-26", quote=800.0, taker_buy=200.0)),
    }
    fetcher = FakeFetcher(payloads)
    manifest_path = tmp_path / "results" / "manifest.json"
    cfg = builder.BuildConfig(
        start="2023-06-25",
        end="2023-06-27",
        output_dir=str(tmp_path / "data"),
        manifest=str(manifest_path),
        workers=2,
    )
    manifest = builder.build(cfg, fetcher=fetcher)
    assert manifest_path.exists()
    assert manifest["protocol"]["archive_checksums_verified"] is True
    assert manifest["protocol"]["raw_archives_retained"] is False
    assert manifest["protocol"]["returns_pnl_or_signals_included"] is False
    assert manifest["validation"]["actual_rows"] == 576
    assert len(manifest["archives"]) == 2

    output = pd.read_csv(manifest["file"]["path"], compression="gzip", parse_dates=["date", "feature_available_time"])
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert len(output) == 576
    assert output["date"].iloc[0] == pd.Timestamp("2023-06-25")
    assert output["date"].iloc[-1] == pd.Timestamp("2023-06-26 23:55:00")
    assert output["feature_available_time"].iloc[-1] == pd.Timestamp("2023-06-27 00:00:01")
    assert output["taker_imbalance"].iloc[0] == pytest.approx(0.5)
    assert output["taker_imbalance"].iloc[-1] == pytest.approx(-0.5)
    assert all(url.endswith(".zip") or url.endswith(".zip.CHECKSUM") for url in fetcher.urls)
