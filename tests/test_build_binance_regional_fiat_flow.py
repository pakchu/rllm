from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_binance_regional_fiat_flow as builder


def _rows(month: str = "2023-01", *, days: int | None = None) -> list[list[object]]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    next_month = start + pd.offsets.MonthBegin(1)
    count = int((next_month - start).days) if days is None else days
    rows: list[list[object]] = []
    for index in range(count):
        open_time = int((start + pd.Timedelta(days=index)).timestamp() * 1000)
        rows.append(
            [
                open_time,
                100.0,
                105.0,
                95.0,
                102.0,
                10.0 + index,
                open_time + 86_400_000 - 1,
                1000.0 + index,
                100 + index,
                4.0 + index / 10.0,
                400.0 + index,
                0.0,
            ]
        )
    return rows


def _archive(rows: list[list[object]], *, header: bool, symbol: str = "BTCEUR") -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=builder.RAW_COLUMNS).to_csv(
        text, index=False, header=header
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1d-test.csv", text.getvalue())
    return output.getvalue()


class _FakeFetcher:
    def __init__(self, payloads: dict[tuple[str, str], bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        filename = url.removesuffix(".CHECKSUM").rsplit("/", 1)[-1]
        symbol, _, month_zip = filename.partition("-1d-")
        month = month_zip.removesuffix(".zip")
        payload = self.payloads[(symbol, month)]
        if url.endswith(".CHECKSUM"):
            digest = hashlib.sha256(payload).hexdigest()
            return f"{digest}  {filename}\n".encode()
        return payload


@pytest.mark.parametrize("header", [False, True])
def test_read_archive_supports_header_transition(header: bool) -> None:
    frame = builder.read_archive(_archive(_rows(days=2), header=header))
    assert frame.loc[0, "base_volume"] == 10.0
    assert frame.loc[1, "trade_count"] == 101


def test_read_archive_rejects_bad_taker_bounds_and_daily_grid() -> None:
    rows = _rows(days=2)
    rows[0][9] = rows[0][5] + 1.0
    with pytest.raises(ValueError, match="taker-buy"):
        builder.read_archive(_archive(rows, header=True))

    rows = _rows(days=2)
    rows[1][0] = rows[0][0]
    rows[1][6] = rows[1][0] + 86_400_000 - 1
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.read_archive(_archive(rows, header=True))

    rows = _rows(days=2)
    rows[1][0] += 60_000
    rows[1][6] += 60_000
    with pytest.raises(ValueError, match="UTC day opens"):
        builder.read_archive(_archive(rows, header=True))


def test_read_archive_rejects_inexact_close_span_and_missing_day() -> None:
    rows = _rows(days=2)
    rows[0][6] -= 1
    with pytest.raises(ValueError, match="exact UTC days"):
        builder.read_archive(_archive(rows, header=True))

    rows = _rows(days=3)
    del rows[1]
    with pytest.raises(ValueError, match="missing or non-daily"):
        builder.read_archive(_archive(rows, header=True))


def test_source_panel_strips_all_price_and_quote_fields() -> None:
    frame = builder.read_archive(_archive(_rows(days=2), header=True))
    output = builder.source_panel(frame, symbol="BTCEUR")
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert output["source_complete"].all()
    assert not {"open", "high", "low", "close", "quote_notional"}.intersection(
        output.columns
    )
    assert np.isclose(output.loc[0, "taker_sell_base_btc"], 6.0)
    assert np.isclose(output.loc[0, "taker_buy_fraction"], 0.4)


def test_archive_url_is_official_spot_monthly_daily_path() -> None:
    assert builder.archive_url("BTCEUR", date(2023, 1, 1)) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCEUR/1d/BTCEUR-1d-2023-01.zip"
    )


def test_process_archive_requires_exact_month_and_verifies_checksum(tmp_path: Path) -> None:
    del tmp_path
    payload = _archive(_rows(), header=True)
    cfg = builder.BuildConfig(
        symbols=("BTCEUR",), start="2023-01-01", end="2023-02-01", workers=1
    )
    panel, metadata = builder._process_archive(
        "BTCEUR",
        date(2023, 1, 1),
        cfg,
        fetcher=_FakeFetcher({("BTCEUR", "2023-01"): payload}),
    )
    assert len(panel) == 31
    assert metadata["archive_sha256"] == hashlib.sha256(payload).hexdigest()

    short_payload = _archive(_rows(days=30), header=True)
    with pytest.raises(ValueError, match="exact UTC daily grid"):
        builder._process_archive(
            "BTCEUR",
            date(2023, 1, 1),
            cfg,
            fetcher=_FakeFetcher({("BTCEUR", "2023-01"): short_payload}),
        )


def test_build_is_byte_deterministic_and_has_full_date_symbol_grid(tmp_path: Path) -> None:
    symbols = ("BTCEUR", "BTCBRL")
    payloads = {
        (symbol, "2023-01"): _archive(_rows(), header=True, symbol=symbol)
        for symbol in symbols
    }
    cfg = builder.BuildConfig(
        symbols=symbols,
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=2,
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    first_panel = Path(first["combined_output"]).read_bytes()
    first_manifest = (tmp_path / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert Path(second["combined_output"]).read_bytes() == first_panel
    assert (tmp_path / "build_manifest.json").read_bytes() == first_manifest
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["rows"] == 62
    assert first["expected_rows"] == 62
    assert first["protocol"]["outcomes_opened"] is False
    assert json.loads(first_manifest)["protocol"]["price_fields_retained"] is False


def test_build_rejects_bad_boundaries_and_duplicate_symbols(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="month starts"):
        builder.build(
            builder.BuildConfig(
                symbols=("BTCEUR",),
                start="2023-01-02",
                end="2023-02-01",
                output_dir=str(tmp_path),
            )
        )
    with pytest.raises(ValueError, match="unique"):
        builder.build(
            builder.BuildConfig(
                symbols=("BTCEUR", "btceur"),
                start="2023-01-01",
                end="2023-02-01",
                output_dir=str(tmp_path),
            )
        )
