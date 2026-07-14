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

from training import build_binance_spot_kline_microstructure as builder


def _archive(rows: list[list[object]], *, header: bool) -> bytes:
    text = io.StringIO()
    pd.DataFrame(rows, columns=builder.RAW_COLUMNS).to_csv(
        text, index=False, header=header
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-test.csv", text.getvalue())
    return output.getvalue()


def _rows(start: str = "2023-01-01 00:00:00") -> list[list[object]]:
    rows: list[list[object]] = []
    for index in range(5):
        open_time = int((pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=index)).timestamp() * 1000)
        close_time = open_time + 59_999
        price = 100.0 + index
        rows.append(
            [
                open_time,
                price,
                price + 1.0,
                price - 1.0,
                price + 0.5,
                10.0,
                close_time,
                1000.0,
                10,
                4.0 if index < 3 else 6.0,
                400.0 if index < 3 else 600.0,
                0.0,
            ]
        )
    return rows


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
    frame = builder.read_archive(_archive(_rows(), header=header))
    assert frame.loc[0, "open"] == 100.0
    assert frame.loc[0, "trade_count"] == 10


def test_read_archive_rejects_bad_taker_totals_and_duplicate_minutes() -> None:
    rows = _rows()
    rows[0][9] = 11.0
    with pytest.raises(ValueError, match="taker-buy"):
        builder.read_archive(_archive(rows, header=True))

    rows = _rows()
    rows[1][0] = rows[0][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.read_archive(_archive(rows, header=True))


def test_archive_url_is_official_spot_monthly_path() -> None:
    assert builder.archive_url("BTCUSDT", date(2023, 1, 1)) == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1m/BTCUSDT-1m-2023-01.zip"
    )


def test_contiguous_ranges_groups_only_adjacent_five_minute_slots() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 00:05:00",
            "2023-01-01 00:20:00",
        ]
    )
    assert builder._contiguous_ranges(timestamps) == [
        {
            "start": "2023-01-01 00:00:00",
            "end": "2023-01-01 00:05:00",
            "rows": 2,
        },
        {
            "start": "2023-01-01 00:20:00",
            "end": "2023-01-01 00:20:00",
            "rows": 1,
        },
    ]


def test_five_minute_aggregation_reconstructs_spot_auction_observables() -> None:
    frame = builder.read_archive(_archive(_rows(), header=True))
    output = builder.aggregate_five_minute(frame)
    assert len(output) == 1
    row = output.iloc[0]
    assert row["date"] == pd.Timestamp("2023-01-01 00:00:00")
    assert row["spot_rows"] == 5
    assert row["source_complete"] == np.bool_(True)
    assert row["quote_notional"] == 5000.0
    assert row["taker_buy_quote"] == 2400.0
    assert row["taker_sell_quote"] == 2600.0
    assert row["signed_quote_notional"] == -200.0
    assert np.isclose(row["flow_coherence"], 0.04)
    assert np.isclose(row["buyer_execution_centroid"], 100.0)
    assert np.isclose(row["seller_execution_centroid"], 100.0)
    assert row["minute_flow_sign_flip_rate"] == 0.25
    assert 0.0 < row["minute_price_path_efficiency"] <= 1.0
    assert 0.0 < row["minute_flow_path_efficiency"] <= 1.0
    assert tuple(output.columns) == builder.OUTPUT_COLUMNS


def test_incomplete_five_minute_group_fails_closed() -> None:
    frame = builder.read_archive(_archive(_rows()[:4], header=True))
    output = builder.aggregate_five_minute(frame)
    assert output.loc[0, "source_complete"] == np.bool_(False)
    assert output.loc[0, "invalid_source_minute_count"] == 0


def test_zero_activity_or_bad_close_time_is_preserved_but_quarantined() -> None:
    rows = _rows()
    rows[2][5] = 0.0
    rows[2][7] = 0.0
    rows[2][8] = 0
    rows[2][9] = 0.0
    rows[2][10] = 0.0
    rows[2][6] = rows[2][0] - 1
    frame = builder.read_archive(_archive(rows, header=True))
    assert frame.loc[2, "source_row_valid"] == np.bool_(False)
    output = builder.aggregate_five_minute(frame)
    assert output.loc[0, "source_complete"] == np.bool_(False)
    assert output.loc[0, "invalid_source_minute_count"] == 1


def test_process_month_resume_rechecks_checksum_and_is_deterministic(tmp_path: Path) -> None:
    payload = _archive(_rows(), header=True)
    fetcher = _FakeFetcher(payload)
    cfg = builder.BuildConfig(
        start="2023-01-01",
        end="2023-02-01",
        output_dir=str(tmp_path),
        workers=1,
    )
    first = builder._process_month(date(2023, 1, 1), cfg, fetcher=fetcher)
    second = builder._process_month(date(2023, 1, 1), cfg, fetcher=fetcher)
    assert first["output_sha256"] == second["output_sha256"]
    assert first["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    metadata_path = next((tmp_path / "monthly").glob("*.json"))
    assert json.loads(metadata_path.read_text())["schema_version"] == builder.SCHEMA_VERSION


def test_build_requires_month_boundaries() -> None:
    with pytest.raises(ValueError, match="month starts"):
        builder.build(
            builder.BuildConfig(start="2023-01-02", end="2023-02-01", workers=1)
        )
