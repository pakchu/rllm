from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from training import build_binance_stablecoin_quote_flow as builder


def _rows(
    start: str,
    *,
    hours: int,
    unit: str = "ms",
) -> list[list[Any]]:
    first = cast(pd.Timestamp, pd.Timestamp(start, tz="UTC"))
    scale = 1_000 if unit == "ms" else 1_000_000
    step = 3_600 * scale
    rows: list[list[Any]] = []
    for index in range(hours):
        current = cast(pd.Timestamp, first + pd.Timedelta(hours=index))
        open_time = int(current.timestamp() * scale)
        rows.append(
            [
                open_time,
                30_000.0,
                30_100.0,
                29_900.0,
                30_050.0,
                10.0 + index / 100.0,
                open_time + step - 1,
                300_000.0 + index,
                100 + index,
                4.0 + index / 1_000.0,
                120_000.0 + index,
                0.0,
            ]
        )
    return rows


def _archive(rows: list[list[Any]], *, header: bool, symbol: str) -> bytes:
    text = io.StringIO()
    frame = pd.DataFrame(
        {
            column: [row[index] for row in rows]
            for index, column in enumerate(builder.RAW_COLUMNS)
        }
    )
    frame.to_csv(text, index=False, header=header)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{symbol}-1h-test.csv", text.getvalue())
    return output.getvalue()


class _FakeFetcher:
    def __init__(self, payloads: dict[tuple[str, str], bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        filename = url.removesuffix(".CHECKSUM").rsplit("/", 1)[-1]
        symbol, _, month_zip = filename.partition("-1h-")
        month = month_zip.removesuffix(".zip")
        payload = self.payloads[(symbol, month)]
        if url.endswith(".CHECKSUM"):
            digest = hashlib.sha256(payload).hexdigest()
            return f"{digest}  {filename}\n".encode()
        return payload


@pytest.mark.parametrize("header", [False, True])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_read_archive_supports_headers_and_timestamp_transition(
    header: bool, unit: str
) -> None:
    raw, observed_unit = builder.read_archive(
        _archive(
            _rows("2025-01-01", hours=2, unit=unit),
            header=header,
            symbol="BTCUSDT",
        )
    )
    assert observed_unit == unit
    assert raw.loc[0, "base_volume"] == 10.0
    assert raw.loc[1, "trade_count"] == 101


def test_read_archive_rejects_bad_bounds_alignment_and_gaps() -> None:
    rows = _rows("2024-01-01", hours=3)
    rows[0][9] = rows[0][5] + 1.0
    with pytest.raises(ValueError, match="taker-buy"):
        builder.read_archive(_archive(rows, header=True, symbol="BTCUSDT"))

    rows = _rows("2024-01-01", hours=3)
    rows[1][0] += 60_000
    rows[1][6] += 60_000
    with pytest.raises(ValueError, match="UTC hour opens"):
        builder.read_archive(_archive(rows, header=True, symbol="BTCUSDT"))

    rows = _rows("2024-01-01", hours=3)
    del rows[1]
    with pytest.raises(ValueError, match="missing or non-hourly"):
        builder.read_archive(_archive(rows, header=True, symbol="BTCUSDT"))


def test_source_panel_strips_prices_and_quote_fields() -> None:
    raw, unit = builder.read_archive(
        _archive(
            _rows("2024-01-01", hours=2), header=True, symbol="BTCUSDC"
        )
    )
    output = builder.source_panel(raw, symbol="BTCUSDC", unit=unit)
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS
    assert bool(output["source_complete"].all())
    assert not {
        "open",
        "high",
        "low",
        "close",
        "quote_notional",
        "taker_buy_quote",
    }.intersection(output.columns)
    assert np.isclose(output.loc[0, "taker_sell_base_btc"], 6.0)
    assert np.isclose(output.loc[0, "signed_taker_flow_btc"], -2.0)


def test_fdusd_activation_is_the_only_allowed_partial_month() -> None:
    expected = builder._expected_hours("BTCFDUSD", date(2023, 8, 1))
    assert expected[0] == pd.Timestamp("2023-08-04T08:00:00Z")
    assert expected[-1] == pd.Timestamp("2023-08-31T23:00:00Z")
    assert len(builder._expected_hours("BTCFDUSD", date(2023, 7, 1))) == 0
    assert len(builder._expected_hours("BTCUSDC", date(2023, 7, 1))) == 31 * 24


def test_process_archive_requires_exact_active_grid_and_checksum() -> None:
    hours = len(builder._expected_hours("BTCFDUSD", date(2023, 8, 1)))
    payload = _archive(
        _rows("2023-08-04 08:00", hours=hours),
        header=True,
        symbol="BTCFDUSD",
    )
    cfg = builder.BuildConfig(
        symbols=("BTCFDUSD",), start="2023-08-01", end="2023-09-01", workers=1
    )
    panel, metadata = builder._process_archive(
        "BTCFDUSD",
        date(2023, 8, 1),
        cfg,
        fetcher=_FakeFetcher({("BTCFDUSD", "2023-08"): payload}),
    )
    assert len(panel) == hours
    assert metadata["archive_sha256"] == hashlib.sha256(payload).hexdigest()

    short = _archive(
        _rows("2023-08-04 08:00", hours=hours - 1),
        header=True,
        symbol="BTCFDUSD",
    )
    with pytest.raises(ValueError, match="exact active UTC-hour grid"):
        builder._process_archive(
            "BTCFDUSD",
            date(2023, 8, 1),
            cfg,
            fetcher=_FakeFetcher({("BTCFDUSD", "2023-08"): short}),
        )


def test_build_is_byte_deterministic_and_respects_launch_grid(tmp_path: Path) -> None:
    full_hours = 31 * 24
    fdusd_hours = len(builder._expected_hours("BTCFDUSD", date(2023, 8, 1)))
    payloads = {
        ("BTCUSDT", "2023-08"): _archive(
            _rows("2023-08-01", hours=full_hours), header=True, symbol="BTCUSDT"
        ),
        ("BTCUSDC", "2023-08"): _archive(
            _rows("2023-08-01", hours=full_hours), header=True, symbol="BTCUSDC"
        ),
        ("BTCFDUSD", "2023-08"): _archive(
            _rows("2023-08-04 08:00", hours=fdusd_hours),
            header=True,
            symbol="BTCFDUSD",
        ),
    }
    cfg = builder.BuildConfig(
        start="2023-08-01",
        end="2023-09-01",
        output_dir=str(tmp_path),
        workers=3,
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    panel_bytes = Path(first["combined_output"]).read_bytes()
    manifest_bytes = (tmp_path / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert Path(second["combined_output"]).read_bytes() == panel_bytes
    assert (tmp_path / "build_manifest.json").read_bytes() == manifest_bytes
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["rows"] == 2 * full_hours + fdusd_hours
    assert first["rows"] == first["expected_rows"] == first["complete_rows"]
    assert first["protocol"]["outcomes_opened"] is False
    assert json.loads(manifest_bytes)["protocol"]["price_fields_retained"] is False


def test_build_rejects_bad_boundaries_and_symbols(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="month starts"):
        builder.build(
            builder.BuildConfig(
                start="2023-08-02", end="2023-09-01", output_dir=str(tmp_path)
            )
        )
    with pytest.raises(ValueError, match="unique"):
        builder.build(
            builder.BuildConfig(
                symbols=("BTCUSDC", "btcusdc"),
                start="2023-08-01",
                end="2023-09-01",
                output_dir=str(tmp_path),
            )
        )
    with pytest.raises(ValueError, match="frozen quote basket"):
        builder.build(
            builder.BuildConfig(
                symbols=("BTCBUSD",),
                start="2023-08-01",
                end="2023-09-01",
                output_dir=str(tmp_path),
            )
        )


def test_archive_url_is_official_spot_monthly_hourly_path() -> None:
    assert builder.archive_url("BTCFDUSD", date(2025, 1, 1)) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCFDUSD/1h/BTCFDUSD-1h-2025-01.zip"
    )
