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

from training import build_binance_btcdom_premium_decomposition_source as builder


RAW_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _archive(rows: list[list[object]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        body = pd.DataFrame(rows, columns=cast(Any, list(RAW_COLUMNS))).to_csv(index=False)
        archive.writestr("source.csv", body)
    return buffer.getvalue()


def _row(open_time: int, close: float) -> list[object]:
    return [
        open_time,
        close,
        close + 0.001,
        close - 0.001,
        close,
        0,
        open_time + 3_600_000 - 1,
        0,
        720,
        0,
        0,
        0,
    ]


def _symbol_frame(symbol: str, dates: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "source_close_time": dates + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "feature_available_time": dates + pd.Timedelta(hours=1, seconds=1),
            "premium_close": values,
        }
    )


def test_inventory_is_bound_to_frozen_source_decision() -> None:
    inventory = builder.load_inventory()
    assert inventory["source_decision_sha256"] == builder.SOURCE_DECISION_SHA256
    assert len(inventory["records"]) == 60
    assert inventory["post_2023_rows_requested"] is False


def test_process_archive_verifies_published_and_archive_hashes() -> None:
    start = int(pd.Timestamp("2023-01-01T00:00:00Z").timestamp() * 1000)
    payload = _archive([_row(start, -0.0002)])
    digest = hashlib.sha256(payload).hexdigest()
    record = {
        "symbol": "BTCDOMUSDT",
        "month": "2023-01",
        "interval": "1h",
        "archive_url": "https://example/archive.zip",
        "checksum_url": "https://example/archive.zip.CHECKSUM",
        "archive_sha256": digest,
    }

    def fetcher(url: str, **_: object) -> bytes:
        return f"{digest} archive.zip\n".encode() if url.endswith("CHECKSUM") else payload

    frame, metadata = builder.process_archive(
        "BTCDOMUSDT",
        date(2023, 1, 1),
        record,
        replace(builder.BuildConfig(), retries=1),
        fetcher=fetcher,
    )
    assert frame["premium_close"].tolist() == [-0.0002]
    assert frame["feature_available_time"].iloc[0] == pd.Timestamp(
        "2023-01-01 01:00:01"
    )
    assert metadata["archive_sha256"] == digest

    def changed(url: str, **_: object) -> bytes:
        return b"0" * 64 + b" archive.zip\n" if url.endswith("CHECKSUM") else payload

    with pytest.raises(ValueError, match="published DLPD checksum changed"):
        builder.process_archive(
            "BTCDOMUSDT",
            date(2023, 1, 1),
            record,
            replace(builder.BuildConfig(), retries=1),
            fetcher=changed,
        )


def test_pair_panel_preserves_missing_legs_without_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "START", pd.Timestamp("2023-01-01 00:00:00"))
    monkeypatch.setattr(builder, "END", pd.Timestamp("2023-01-01 03:00:00"))
    dates = pd.date_range(builder.START, builder.END, freq="1h", inclusive="left")
    btc = _symbol_frame("BTCUSDT", dates, [0.1, 0.2, 0.3])
    dom = _symbol_frame("BTCDOMUSDT", dates.delete(1), [-0.1, -0.3])
    panel = builder.pair_panel([btc, dom])
    assert panel["btcusdt_valid"].tolist() == [True, True, True]
    assert panel["btcdomusdt_valid"].tolist() == [True, False, True]
    assert panel["source_valid"].tolist() == [True, False, True]
    assert np.isnan(panel.loc[1, "btcdomusdt_premium_close"])
    assert panel.loc[1, "btcusdt_premium_close"] == pytest.approx(0.2)


def test_pair_panel_is_outcome_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "START", pd.Timestamp("2023-01-01 00:00:00"))
    monkeypatch.setattr(builder, "END", pd.Timestamp("2023-01-01 01:00:00"))
    dates = pd.date_range(builder.START, builder.END, freq="1h", inclusive="left")
    panel = builder.pair_panel(
        [
            _symbol_frame("BTCUSDT", dates, [0.1]),
            _symbol_frame("BTCDOMUSDT", dates, [-0.2]),
        ]
    )
    assert panel.columns.tolist() == list(builder.OUTPUT_COLUMNS)
    forbidden = ("return", "label", "pnl", "funding", "contract_open", "contract_high")
    assert not any(token in column for token in forbidden for column in panel.columns)


def test_manifest_writer_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    builder._write_frozen_json(path, {"source_only": True, "outcomes_opened": False})
    first = path.read_bytes()
    builder._write_frozen_json(path, {"source_only": True, "outcomes_opened": False})
    assert path.read_bytes() == first
    with pytest.raises(FileExistsError, match="differs"):
        builder._write_frozen_json(path, {"source_only": False})
