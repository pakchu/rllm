from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_clbr_execution_sources as builder


def _kline_archive(day: date) -> bytes:
    rows = []
    for timestamp in pd.date_range(day, periods=288, freq="5min"):
        open_ms = int(timestamp.tz_localize("UTC").timestamp() * 1_000)
        rows.append(
            ",".join(
                map(
                    str,
                    [
                        open_ms,
                        100,
                        102,
                        99,
                        101,
                        1,
                        open_ms + 299_999,
                        100,
                        10,
                        0.5,
                        50,
                        0,
                    ],
                )
            )
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("klines.csv", "\n".join(rows) + "\n")
    return buffer.getvalue()


def test_market_day_verifies_checksum_and_exact_grid() -> None:
    payload = _kline_archive(date(2023, 6, 25))
    digest = hashlib.sha256(payload).hexdigest()

    def fetcher(url: str, **_: object) -> bytes:
        return (digest + "  x\n").encode() if url.endswith("CHECKSUM") else payload

    result = builder.process_market_day(
        date(2023, 6, 25), builder.Config(), fetcher=fetcher
    )
    assert result["archive_sha256"] == digest
    assert result["rows"] == 288
    assert result["frame"]["date"].iloc[-1] == pd.Timestamp("2023-06-25 23:55:00")

    def bad(url: str, **_: object) -> bytes:
        return b"0" * 64 + b"  x\n" if url.endswith("CHECKSUM") else payload

    with pytest.raises(ValueError, match="checksum"):
        builder.process_market_day(date(2023, 6, 25), builder.Config(), fetcher=bad)


def test_funding_pagination_is_bounded_and_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    start_ms = builder.START_MS
    monkeypatch.setattr(builder, "END_MS", start_ms + builder.MARK_STEP_MS)
    second = start_ms + builder.MARK_STEP_MS
    pages = [
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": start_ms,
                "fundingRate": "0.0001",
                "markPrice": "",
            }
        ],
        [],
    ]

    def opener(url: str) -> object:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert int(query["endTime"][0]) < builder.END_MS
        return pages.pop(0)

    frame, count = builder.download_funding(builder.Config(), opener=opener)
    assert count == 1
    assert frame["funding_time_ms"].tolist() == [start_ms]
    assert second > start_ms


def test_funding_rejects_duplicate_and_missing_canonical_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_ms = builder.START_MS
    monkeypatch.setattr(builder, "END_MS", start_ms + 3 * builder.MARK_STEP_MS)

    def event(timestamp: int) -> dict[str, object]:
        return {
            "symbol": "BTCUSDT",
            "fundingTime": timestamp,
            "fundingRate": "0.0001",
            "markPrice": "100",
        }

    duplicate = [
        event(start_ms),
        event(start_ms),
        event(start_ms + builder.MARK_STEP_MS),
        event(start_ms + 2 * builder.MARK_STEP_MS),
    ]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        builder.download_funding(builder.Config(), opener=lambda _: duplicate)

    missing = [event(start_ms), event(start_ms + 2 * builder.MARK_STEP_MS)]
    with pytest.raises(ValueError, match="incomplete canonical 8h grid"):
        builder.download_funding(builder.Config(), opener=lambda _: missing)


@pytest.mark.parametrize("mark_price", ["not-a-number", "NaN", "0", "-1"])
def test_funding_rejects_nonempty_invalid_recorded_mark(
    monkeypatch: pytest.MonkeyPatch, mark_price: str
) -> None:
    start_ms = builder.START_MS
    monkeypatch.setattr(builder, "END_MS", start_ms + builder.MARK_STEP_MS)
    payload = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": start_ms,
            "fundingRate": "0.0001",
            "markPrice": mark_price,
        }
    ]
    with pytest.raises(ValueError, match="invalid recorded mark"):
        builder.download_funding(builder.Config(), opener=lambda _: payload)


def test_mark_grid_and_funding_composition() -> None:
    start_ms = builder.START_MS
    periods = (builder.END_MS - builder.START_MS) // builder.MARK_STEP_MS
    rows = [
        [start_ms + i * builder.MARK_STEP_MS, "100", "0", "0", "0", "0", 0]
        for i in range(periods)
    ]

    def opener(_: str) -> object:
        return rows

    marks, pages = builder.download_mark_klines(builder.Config(), opener=opener)
    assert pages == 1
    funding = pd.DataFrame(
        {
            "funding_time_ms": [start_ms + 8, start_ms + builder.MARK_STEP_MS + 9],
            "funding_rate": [0.0001, -0.0001],
            "recorded_mark_price": [100.0, float("nan")],
        }
    )
    output, quality = builder.compose_funding(funding, marks)
    assert output.loc[0, "settlement_mark_price"] == 100.0
    assert output.loc[0, "mark_source"] == "funding_history_recorded_mark"
    assert output.loc[1, "settlement_mark_price"] == 100.0
    assert output.loc[1, "mark_source"] == "binance_8h_mark_price_kline_open"
    assert output.loc[0, "funding_time_offset_ms"] == 8
    assert quality["recorded_mark_overlap_events"] == 1
    assert quality["mark_proxy_events"] == 1


def test_mark_history_rejects_duplicate_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_ms = builder.START_MS
    monkeypatch.setattr(builder, "END_MS", start_ms + 2 * builder.MARK_STEP_MS)
    rows = [
        [start_ms, "100", "0", "0", "0", "0", 0],
        [start_ms, "101", "0", "0", "0", "0", 0],
        [start_ms + builder.MARK_STEP_MS, "100", "0", "0", "0", "0", 0],
    ]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        builder.download_mark_klines(builder.Config(), opener=lambda _: rows)


def test_mocked_build_writes_physically_separate_deterministic_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_day(day: date, _: builder.Config) -> dict[str, object]:
        dates = pd.date_range(day, periods=288, freq="5min")
        frame = pd.DataFrame(
            {"date": dates, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
        )
        return {
            "date": day.isoformat(),
            "archive_url": "https://example.invalid/a.zip",
            "checksum_url": "https://example.invalid/a.zip.CHECKSUM",
            "archive_sha256": "a" * 64,
            "expected_archive_sha256": "a" * 64,
            "checksum_payload_sha256": "b" * 64,
            "rows": 288,
            "frame": frame,
        }

    monkeypatch.setattr(builder, "process_market_day", fake_day)
    monkeypatch.setattr(
        builder,
        "download_funding",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "funding_time_ms": [
                        builder.START_MS
                    ],
                    "funding_rate": [0.0001],
                    "recorded_mark_price": [100.0],
                }
            ),
            1,
        ),
    )
    monkeypatch.setattr(
        builder,
        "download_mark_klines",
        lambda *args, **kwargs: (
            pd.DataFrame(
                {
                    "mark_open_time_ms": [
                        builder.START_MS
                    ],
                    "settlement_mark_price": [100.0],
                }
            ),
            1,
        ),
    )
    cfg = replace(
        builder.Config(),
        output_dir=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
        workers=2,
    )
    first = builder.build(cfg)
    first_bytes = Path(cfg.manifest).read_bytes()
    second = builder.build(cfg)
    assert first == second
    assert Path(cfg.manifest).read_bytes() == first_bytes
    assert set(first["files"]) == {"train", "test", "eval"}
    for split, files in first["files"].items():
        market = pd.read_csv(files["market"]["path"], compression="gzip")
        assert len(market) == files["market"]["rows"]
        if split != "train":
            assert files["funding"]["rows"] == 0
    loaded = json.loads(Path(cfg.manifest).read_text())
    assert loaded["protocol"]["outcomes_opened"] is False
    assert loaded["protocol"]["clbr_clocks_loaded"] is False


def test_range_is_immutable() -> None:
    with pytest.raises(ValueError, match="range is immutable"):
        builder.build(replace(builder.Config(), end="2024-10-16 00:00:00"))
