from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_regional_fiat_cross_rate as builder


SYMBOLS = ("BTCUSDT", "BTCEUR", "BTCTRY", "BTCBRL")
CLOSE_COLUMNS = tuple(f"{symbol}_close" for symbol in SYMBOLS)


def _unit_scale(unit: str) -> int:
    if unit == "ms":
        return 1_000
    if unit == "us":
        return 1_000_000
    raise AssertionError(unit)


def _timestamp(value: pd.Timestamp, *, unit: str) -> int:
    return int(value.timestamp() * _unit_scale(unit))


def _rows(
    month: str = "2023-01",
    *,
    days: int | None = None,
    unit: str = "ms",
    close_start: float = 100.0,
) -> list[list[object]]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    next_month = start + pd.offsets.MonthBegin(1)
    count = int((next_month - start).days) if days is None else days
    rows: list[list[object]] = []
    for index in range(count):
        day = start + pd.Timedelta(days=index)
        open_time = _timestamp(day, unit=unit)
        close_time = open_time + (86_400_000 if unit == "ms" else 86_400_000_000) - 1
        close = close_start + index
        rows.append(
            [
                open_time,
                close - 1.0,
                close + 2.0,
                close - 3.0,
                close,
                10.0 + index,
                close_time,
                1000.0 + index,
                100 + index,
                4.0 + index / 10.0,
                400.0 + index,
                0.0,
            ]
        )
    return rows


def _archive(
    rows: list[list[object]],
    *,
    header: bool,
    symbol: str = "BTCUSDT",
) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=builder.RAW_COLUMNS).to_csv(
        text, index=False, header=header
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1d-test.csv", text.getvalue())
    return output.getvalue()


class _FakeFetcher:
    def __init__(
        self,
        payloads: dict[tuple[str, str], bytes],
        *,
        checksum_filename: str | None = None,
        checksum_hash: str | None = None,
    ) -> None:
        self.payloads = payloads
        self.checksum_filename = checksum_filename
        self.checksum_hash = checksum_hash

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        filename = url.removesuffix(".CHECKSUM").rsplit("/", 1)[-1]
        symbol, _, month_zip = filename.partition("-1d-")
        month = month_zip.removesuffix(".zip")
        payload = self.payloads[(symbol, month)]
        if url.endswith(".CHECKSUM"):
            digest = self.checksum_hash or hashlib.sha256(payload).hexdigest()
            checksum_name = self.checksum_filename or filename
            return f"{digest}  {checksum_name}\n".encode()
        return payload


def _payloads_for_month(month: str = "2023-01", *, unit: str = "ms") -> dict[tuple[str, str], bytes]:
    return {
        (symbol, month): _archive(
            _rows(month, unit=unit, close_start=100.0 + offset * 1000.0),
            header=True,
            symbol=symbol,
        )
        for offset, symbol in enumerate(SYMBOLS)
    }


@pytest.mark.parametrize("header", [False, True])
def test_read_archive_supports_header_and_headerless_files(header: bool) -> None:
    frame = builder.read_archive(_archive(_rows(days=2), header=header))

    assert tuple(frame.columns) == builder.RAW_COLUMNS
    assert frame.loc[0, "close"] == 100.0
    assert frame.loc[1, "trade_count"] == 101


def test_read_archive_accepts_2025_and_later_microsecond_timestamps() -> None:
    frame = builder.read_archive(_archive(_rows("2025-01", days=2, unit="us"), header=True))

    assert frame.loc[0, "open_time"] == _timestamp(pd.Timestamp("2025-01-01", tz="UTC"), unit="us")
    assert frame.loc[0, "close_time"] == (
        _timestamp(pd.Timestamp("2025-01-01", tz="UTC"), unit="us")
        + 86_400_000_000
        - 1
    )
    source = builder.source_row(frame.iloc[[0]], symbol="BTCUSDT").iloc[0].to_dict()
    assert pd.Timestamp(source["date"]) == pd.Timestamp("2025-01-01")
    assert source["close"] == 100.0
    assert bool(source["source_complete"]) is True
    assert not {"open", "high", "low", "base_volume", "trade_count"}.intersection(source)


@pytest.mark.parametrize(
    ("month", "unit", "message"),
    [
        ("2025-01", "ms", "microsecond"),
        ("2024-12", "us", "millisecond"),
    ],
)
def test_read_archive_rejects_wrong_timestamp_unit_for_transition(
    month: str, unit: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        builder.read_archive(
            _archive(_rows(month, days=1, unit=unit), header=True)
        )


def test_read_archive_requires_exact_utc_day_close() -> None:
    rows = _rows(days=1)
    rows[0][6] -= 1

    with pytest.raises(ValueError, match="exact UTC day|23:59:59.999|close"):
        builder.read_archive(_archive(rows, header=True))


def test_read_archive_enforces_ohlc_invariant() -> None:
    rows = _rows(days=1)
    rows[0][2] = rows[0][4] - 0.01

    with pytest.raises(ValueError, match="OHLC|high|low"):
        builder.read_archive(_archive(rows, header=True))


def test_read_archive_requires_positive_close_base_volume_and_trade_count() -> None:
    bad_cases = [
        ("price|close", 4, 0.0),
        ("base volume|empty|stale", 5, 0.0),
        ("trade count|empty|stale", 8, 0),
    ]
    for expected, column_index, value in bad_cases:
        rows = _rows(days=1)
        rows[0][column_index] = value
        with pytest.raises(ValueError, match=expected):
            builder.read_archive(_archive(rows, header=True))


def test_archive_urls_are_official_spot_monthly_daily_paths() -> None:
    assert builder.archive_url("BTCEUR", date(2023, 1, 1)) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCEUR/1d/BTCEUR-1d-2023-01.zip"
    )
    assert builder.checksum_url("BTCEUR", date(2023, 1, 1)).endswith(
        "/BTCEUR-1d-2023-01.zip.CHECKSUM"
    )


def test_process_archive_verifies_checksum_filename_and_hash() -> None:
    payload = _archive(_rows(), header=True, symbol="BTCEUR")
    cfg = builder.BuildConfig(
        symbols=SYMBOLS,
        start="2023-01-01",
        end="2023-02-01",
        workers=1,
    )

    panel, metadata = builder._process_archive(
        "BTCEUR",
        date(2023, 1, 1),
        cfg,
        fetcher=_FakeFetcher({("BTCEUR", "2023-01"): payload}),
    )
    assert len(panel) == 31
    assert metadata["archive_url"] == builder.archive_url("BTCEUR", date(2023, 1, 1))
    assert metadata["checksum_url"] == builder.checksum_url("BTCEUR", date(2023, 1, 1))
    assert metadata["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    expected_checksum = f"{metadata['archive_sha256']}  BTCEUR-1d-2023-01.zip\n".encode()
    assert metadata["checksum_response_sha256"] == hashlib.sha256(expected_checksum).hexdigest()
    assert metadata["published_archive_sha256"] == metadata["archive_sha256"]

    with pytest.raises(ValueError, match="checksum.*filename|filename.*checksum"):
        builder._process_archive(
            "BTCEUR",
            date(2023, 1, 1),
            cfg,
            fetcher=_FakeFetcher(
                {("BTCEUR", "2023-01"): payload},
                checksum_filename="wrong-file.zip",
            ),
        )

    with pytest.raises(ValueError, match="checksum.*filename|filename.*checksum"):
        builder._process_archive(
            "BTCEUR",
            date(2023, 1, 1),
            cfg,
            fetcher=_FakeFetcher(
                {("BTCEUR", "2023-01"): payload},
                checksum_filename="nested/BTCEUR-1d-2023-01.zip",
            ),
        )

    with pytest.raises(ValueError, match="sha256|checksum|hash"):
        builder._process_archive(
            "BTCEUR",
            date(2023, 1, 1),
            cfg,
            fetcher=_FakeFetcher(
                {("BTCEUR", "2023-01"): payload},
                checksum_hash="0" * 64,
            ),
        )


def test_process_archive_requires_exact_month_grid() -> None:
    cfg = builder.BuildConfig(
        symbols=SYMBOLS,
        start="2023-01-01",
        end="2023-02-01",
        workers=1,
    )
    short_payload = _archive(_rows("2023-01", days=30), header=True, symbol="BTCBRL")

    with pytest.raises(ValueError, match="exact.*month|daily grid|missing"):
        builder._process_archive(
            "BTCBRL",
            date(2023, 1, 1),
            cfg,
            fetcher=_FakeFetcher({("BTCBRL", "2023-01"): short_payload}),
        )


def test_build_writes_four_symbol_pivot_with_only_frozen_source_columns(tmp_path: Path) -> None:
    cfg = builder.BuildConfig(
        symbols=SYMBOLS,
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=2,
    )
    manifest = builder.build(
        cfg,
        fetcher=_FakeFetcher(_payloads_for_month()),
        _allow_partial_fixture=True,
    )
    output = pd.read_csv(manifest["combined_output"], compression="gzip")

    assert tuple(builder.OUTPUT_COLUMNS) == (
        "date",
        "source_available_not_before",
        *CLOSE_COLUMNS,
        "source_complete",
    )
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert len(output) == 31
    assert output.loc[0, "date"] == "2023-01-01"
    assert output.loc[0, "source_available_not_before"] == "2023-01-02T00:00:00Z"
    assert output.loc[0, "BTCUSDT_close"] == 100.0
    assert output.loc[0, "BTCEUR_close"] == 1100.0
    assert output.loc[0, "BTCTRY_close"] == 2100.0
    assert output.loc[0, "BTCBRL_close"] == 3100.0
    assert output["source_complete"].all()
    forbidden = {"symbol", "open", "high", "low", "volume", "trade_count", "quote_notional"}
    assert not forbidden.intersection(output.columns)
    assert manifest["protocol"]["outcomes_opened"] is False


def test_build_rejects_mismatched_close_calendars(tmp_path: Path) -> None:
    payloads = _payloads_for_month()
    bad_brl_rows = _rows("2023-01")
    del bad_brl_rows[3]
    payloads[("BTCBRL", "2023-01")] = _archive(bad_brl_rows, header=True, symbol="BTCBRL")
    cfg = builder.BuildConfig(
        symbols=SYMBOLS,
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=1,
    )

    with pytest.raises(ValueError, match="calendar|grid|missing|identical"):
        builder.build(
            cfg,
            fetcher=_FakeFetcher(payloads),
            _allow_partial_fixture=True,
        )


def test_build_is_byte_deterministic_for_output_and_manifest(tmp_path: Path) -> None:
    cfg = builder.BuildConfig(
        symbols=SYMBOLS,
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=4,
    )
    first = builder.build(
        cfg,
        fetcher=_FakeFetcher(_payloads_for_month()),
        _allow_partial_fixture=True,
    )
    first_output = Path(first["combined_output"]).read_bytes()
    first_manifest = (tmp_path / "build_manifest.json").read_bytes()

    second = builder.build(
        cfg,
        fetcher=_FakeFetcher(_payloads_for_month()),
        _allow_partial_fixture=True,
    )

    assert Path(second["combined_output"]).read_bytes() == first_output
    assert (tmp_path / "build_manifest.json").read_bytes() == first_manifest
    assert first["combined_sha256"] == second["combined_sha256"]
    assert json.loads(first_manifest)["columns"] == list(builder.OUTPUT_COLUMNS)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            builder.BuildConfig(
                symbols=SYMBOLS,
                start="2020-09-01",
                end="2023-02-01",
            ),
            "2020-10-01",
        ),
        (
            builder.BuildConfig(
                symbols=SYMBOLS,
                start="2020-10-02",
                end="2023-02-01",
            ),
            "month starts",
        ),
        (
            builder.BuildConfig(
                symbols=SYMBOLS,
                start="2020-10-01",
                end="2024-02-01",
            ),
            "2024-01-01",
        ),
        (
            builder.BuildConfig(
                symbols=SYMBOLS,
                start="2021-01-01",
                end="2024-01-01",
            ),
            "exact horizon",
        ),
        (
            builder.BuildConfig(
                symbols=("BTCUSDT", "BTCEUR", "BTCTRY"),
                start="2020-10-01",
                end="2023-02-01",
            ),
            "exactly these symbols",
        ),
        (
            builder.BuildConfig(
                symbols=("BTCUSDT", "BTCEUR", "BTCTRY", "BTCBRL", "BTCKRW"),
                start="2020-10-01",
                end="2023-02-01",
            ),
            "exactly these symbols",
        ),
        (
            builder.BuildConfig(
                symbols=("BTCUSDT", "BTCEUR", "BTCTRY", "btceur"),
                start="2020-10-01",
                end="2023-02-01",
            ),
            "unique|exactly these symbols",
        ),
    ],
    ids=[
        "before-first-common-month",
        "non-month-start",
        "after-frozen-source-cap",
        "partial-production-horizon",
        "missing-symbol",
        "extra-symbol",
        "duplicate-after-normalization",
    ],
)
def test_build_rejects_bad_boundaries_and_symbol_set(
    tmp_path: Path, cfg: builder.BuildConfig, message: str
) -> None:
    cfg = builder.BuildConfig(
        symbols=cfg.symbols,
        start=cfg.start,
        end=cfg.end,
        output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match=message):
        builder.build(
            cfg,
            fetcher=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("invalid config must fail before network fetch")
            ),
        )
