from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_um_kline_reference as builder


def _archive(rows: list[list[object]], *, header: bool) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=builder.RAW_COLUMNS).to_csv(
        text,
        index=False,
        header=header,
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-5m-test.csv", text.getvalue())
    return output.getvalue()


def _row(timestamp: str, close: float = 101.0) -> list[object]:
    open_time = int(pd.Timestamp(timestamp, tz="UTC").timestamp() * 1_000)
    return [
        open_time,
        100.0,
        102.0,
        99.0,
        close,
        10.0,
        open_time + 299_999,
        1_005.0,
        20,
        6.0,
        603.0,
        0,
    ]


def _day_rows(day: str, first_close: float = 101.0) -> list[list[object]]:
    timestamps = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta("1d"), inclusive="left", freq="5min")
    return [
        _row(str(timestamp), first_close if position == 0 else 101.0)
        for position, timestamp in enumerate(timestamps)
    ]


class _FakeFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        if url.endswith(".CHECKSUM"):
            digest = hashlib.sha256(self.payload).hexdigest()
            return f"{digest}  archive.zip\n".encode()
        return self.payload


@pytest.mark.parametrize("header", [False, True])
def test_read_archive_supports_header_transition(header: bool) -> None:
    frame = builder.read_archive(_archive([_row("2021-01-01")], header=header))
    assert tuple(frame.columns) == builder.OUTPUT_COLUMNS
    assert frame.loc[0, "date"] == pd.Timestamp("2021-01-01")
    assert frame.loc[0, "quote_asset_volume"] == 1_005.0
    assert frame.loc[0, "number_of_trades"] == 20


def test_read_archive_rejects_duplicate_open_times() -> None:
    payload = _archive(
        [_row("2021-01-01"), _row("2021-01-01")],
        header=True,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.read_archive(payload)


def test_official_daily_url() -> None:
    assert builder.archive_url("BTCUSDT", "5m", date(2023, 11, 1)) == (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        "BTCUSDT/5m/BTCUSDT-5m-2023-11-01.zip"
    )


def test_partial_month_process_and_checksum_refresh(tmp_path: Path) -> None:
    rows = _day_rows("2021-01-01")
    fetcher = _FakeFetcher(_archive(rows, header=True))
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    metadata = builder._process_month(date(2021, 1, 1), cfg, fetcher=fetcher)
    assert metadata["rows"] == 288
    first_hash = metadata["output_sha256"]

    changed = _day_rows("2021-01-01", first_close=100.5)
    fetcher.payload = _archive(changed, header=True)
    rebuilt = builder._process_month(date(2021, 1, 1), cfg, fetcher=fetcher)
    assert rebuilt["output_sha256"] != first_hash
    output = pd.read_csv(rebuilt["output"], compression="gzip")
    assert output.loc[0, "close"] == 100.5


def test_process_rejects_missing_five_minute_slot(tmp_path: Path) -> None:
    fetcher = _FakeFetcher(_archive(_day_rows("2021-01-01")[:-1], header=True))
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    with pytest.raises(ValueError, match="incomplete timestamp coverage"):
        builder._process_month(date(2021, 1, 1), cfg, fetcher=fetcher)


def test_overwrite_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _day_rows("2021-01-01")
    fetcher = _FakeFetcher(_archive(rows, header=False))
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
        overwrite=True,
    )
    first = builder._process_month(date(2021, 1, 1), cfg, fetcher=fetcher)
    second = builder._process_month(date(2021, 1, 1), cfg, fetcher=fetcher)
    assert first["output_sha256"] == second["output_sha256"]
