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

from training import build_binance_coinm_liquidation_snapshot as builder


def _row(
    timestamp: int = 1_687_656_471_926,
    *,
    side: str = "BUY",
    quantity: int | float = 3,
    price: float = 30_741.3,
    average_price: float = 30_631.6,
) -> str:
    return ",".join(
        map(
            str,
            (
                timestamp,
                side,
                "LIMIT",
                "IOC",
                quantity,
                price,
                average_price,
                "FILLED",
                quantity,
                quantity,
            ),
        )
    )


def _archive(rows: list[str], *, header: str | None = None) -> bytes:
    columns = header or ",".join(builder.RAW_COLUMNS)
    payload = (columns + "\n" + "\n".join(rows) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("liquidation.csv", payload)
    return buffer.getvalue()


def test_urls_match_official_daily_tree() -> None:
    day = date(2023, 6, 25)
    assert builder.archive_url(day) == (
        "https://data.binance.vision/data/futures/cm/daily/liquidationSnapshot/"
        "BTCUSD_PERP/BTCUSD_PERP-liquidationSnapshot-2023-06-25.zip"
    )
    assert builder.checksum_url(day).endswith(".zip.CHECKSUM")


def test_read_archive_removes_only_exact_duplicate_rows() -> None:
    row = _row()
    distinct_same_ms = _row(quantity=2)
    frame, removed = builder.read_archive(_archive([row, row, distinct_same_ms]))
    assert len(frame) == 2
    assert removed == 1
    assert frame["time"].nunique() == 1


def test_read_archive_rejects_invalid_contract_semantics() -> None:
    with pytest.raises(ValueError, match="unexpected liquidation columns"):
        builder.read_archive(_archive([_row()], header="bad,column"))
    with pytest.raises(ValueError, match="unknown side"):
        builder.read_archive(_archive([_row(side="HOLD")]))
    with pytest.raises(ValueError, match="BUY average price"):
        builder.read_archive(_archive([_row(price=30_000.0, average_price=30_100.0)]))
    with pytest.raises(ValueError, match="contract quantities must be integers"):
        builder.read_archive(_archive([_row(quantity=3.5)]))


def test_aggregate_maps_sell_to_long_and_buy_to_short_liquidation() -> None:
    buy = _row(timestamp=1_687_656_471_926, side="BUY", quantity=3)
    sell = _row(
        timestamp=1_687_656_472_926,
        side="SELL",
        quantity=2,
        price=30_500.0,
        average_price=30_600.0,
    )
    events, _ = builder.read_archive(_archive([buy, sell]))
    output = builder.aggregate_day(events, date(2023, 6, 25))
    row = output.loc[output["event_count"].eq(2)].iloc[0]
    assert row["short_liquidation_usd"] == 300.0
    assert row["long_liquidation_usd"] == 200.0
    assert row["signed_liquidation_usd"] == 100.0
    assert row["liquidation_imbalance"] == pytest.approx(0.2)
    assert row["event_notional_hhi"] == pytest.approx(0.52)
    assert row["first_snapshot_average_price"] == pytest.approx(30_631.6)
    assert row["last_snapshot_average_price"] == pytest.approx(30_600.0)
    assert row["min_snapshot_average_price"] == pytest.approx(30_600.0)
    assert row["max_snapshot_average_price"] == pytest.approx(30_631.6)
    assert row["snapshot_price_closing_location"] == pytest.approx(0.0)
    assert row["feature_available_time"] == row["date"] + pd.Timedelta(
        minutes=5, seconds=1
    )
    assert output.loc[output["event_count"].eq(0), "source_valid"].all()


def test_process_day_verifies_checksum_and_fails_missing_day_closed() -> None:
    payload = _archive([_row()])
    digest = hashlib.sha256(payload).hexdigest()

    def fetcher(url: str, **_: object) -> bytes:
        return (digest + "  archive.zip\n").encode() if url.endswith("CHECKSUM") else payload

    result = builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=fetcher)
    assert result["available"] is True
    assert result["archive_sha256"] == digest
    assert result["snapshot_rows"] == 1

    def missing(_: str, **__: object) -> bytes:
        raise FileNotFoundError

    absent = builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=missing)
    assert absent["available"] is False
    assert not absent["frame"]["source_valid"].any()
    assert absent["frame"]["event_count"].isna().all()


def test_process_day_rejects_wrong_date_and_checksum() -> None:
    wrong = _archive([_row(timestamp=1_687_742_871_926)])
    digest = hashlib.sha256(wrong).hexdigest()

    def wrong_day(url: str, **_: object) -> bytes:
        return (digest + "  x\n").encode() if url.endswith("CHECKSUM") else wrong

    with pytest.raises(ValueError, match="another UTC date"):
        builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=wrong_day)

    def wrong_checksum(url: str, **_: object) -> bytes:
        return b"0" * 64 + b"  x\n" if url.endswith("CHECKSUM") else wrong

    with pytest.raises(ValueError, match="checksum"):
        builder.process_day(date(2023, 6, 25), builder.Config(), fetcher=wrong_checksum)


def test_build_is_deterministic_and_records_missing_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_process_day(day: date, _: builder.Config) -> dict[str, object]:
        if day == date(2023, 6, 26):
            return builder._missing_day(day)
        frame = builder._day_grid(day, source_valid=True)
        frame.loc[0, "event_count"] = 1
        return {
            "date": day.isoformat(),
            "available": True,
            "archive_url": "https://example.invalid/a.zip",
            "checksum_url": "https://example.invalid/a.zip.CHECKSUM",
            "archive_sha256": "a" * 64,
            "expected_archive_sha256": "a" * 64,
            "checksum_payload_sha256": "b" * 64,
            "raw_rows": 2,
            "snapshot_rows": 1,
            "duplicate_rows_removed": 1,
            "event_bars": 1,
            "first_time_ms": 1,
            "last_time_ms": 1,
            "frame": frame,
        }

    monkeypatch.setattr(builder, "process_day", fake_process_day)
    cfg = replace(
        builder.Config(),
        start="2023-06-25",
        end="2023-06-27",
        workers=2,
        output_dir=str(tmp_path / "data"),
        manifest=str(tmp_path / "manifest.json"),
    )
    first = builder.build(cfg)
    first_bytes = Path(cfg.manifest).read_bytes()
    second = builder.build(cfg)
    assert first == second
    assert Path(cfg.manifest).read_bytes() == first_bytes
    assert first["missing_archive_dates"] == ["2023-06-26"]
    assert first["file"]["rows"] == 576
    assert first["file"]["source_valid_rows"] == 288
    assert hashlib.sha256(Path(first["file"]["path"]).read_bytes()).hexdigest() == first[
        "file"
    ]["sha256"]
    loaded = json.loads(Path(cfg.manifest).read_text())
    assert loaded["protocol"]["outcomes_opened"] is False


def test_build_enforces_physical_archive_bounds() -> None:
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), start="2023-06-24"))
    with pytest.raises(ValueError, match="physically bounded"):
        builder.build(replace(builder.Config(), end="2024-10-16"))
